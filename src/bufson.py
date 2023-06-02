from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Self
from urllib.parse import parse_qs

import requests

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

import build
import pydantic
import pypi_simple
from packaging.utils import NormalizedName, canonicalize_name
from packaging.version import InvalidVersion, Version
from yarl import URL


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)10s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


log = _get_logger()


PathType = str | os.PathLike[str]


class PipGripDep(pydantic.BaseModel):
    name: str
    version: str
    dependencies: list[PipGripDep] = []


class DepID(NamedTuple):
    name: NormalizedName
    version: Version | URL

    def __str__(self) -> str:
        if isinstance(self.version, Version):
            return f"{self.name}=={self.version}"
        return f"{self.name} @ {self.version}"

    def __lt__(self, other: DepID) -> bool:
        return (self.name, str(self.version)) < (other.name, str(other.version))

    @classmethod
    def parse(cls, name: str, raw_version: str) -> Self:
        normalized_name = canonicalize_name(name)
        try:
            version = Version(raw_version)
        except InvalidVersion:
            version = URL(raw_version)

        return cls(normalized_name, version)


def to_dep_id(dep: PipGripDep) -> DepID:
    return DepID.parse(dep.name, dep.version)


class _DirectDeps(NamedTuple):
    runtime: list[DepID]
    build: list[DepID]


DepGraph = dict[DepID, _DirectDeps]


class Cache:
    def __init__(self, root: PathType) -> None:
        self._root = Path(root)

    @property
    def pipgrip_cache(self) -> Path:
        return self._root / "pipgrip"

    def get_sdist_cache(self) -> Path:
        return self._mkdir("sdists")

    def _mkdir(self, relpath: PathType) -> Path:
        path = self._root / relpath
        path.mkdir(exist_ok=True, parents=True)
        return path


@dataclass
class Config:
    allow_wheels: bool = False
    request_timeout: int = 60


class Bufson:
    def __init__(
        self,
        cache: Cache,
        config: Config,
        pypi_client: pypi_simple.PyPISimple,
    ) -> None:
        self._cache = cache
        self._config = config
        self._pypi_client = pypi_client

    def resolve_deps(self, requirements: Iterable[str]) -> list[PipGripDep]:
        requirements = list(requirements)
        if not requirements:
            return []

        cmd = [
            "pipgrip",
            "--cache-dir",
            self._cache.pipgrip_cache,
            "--tree",
            "--json",
            *requirements,
        ]

        output = _run_cmd(cmd)
        return list(map(PipGripDep.parse_obj, json.loads(output)))

    def build_depgraph(self, deps: Iterable[PipGripDep]) -> DepGraph:
        graph: DepGraph = {}
        self._populate_depgraph(graph, deps)
        return graph

    def _populate_depgraph(self, graph: DepGraph, deps: Iterable[PipGripDep]) -> None:
        def dedup(x: Iterable[DepID]) -> list[DepID]:
            return list(dict.fromkeys(x))

        for dep in deps:
            dep_id = to_dep_id(dep)

            if dep_id not in graph:
                builddeps = self._get_builddeps(dep_id)
                graph[dep_id] = _DirectDeps(
                    runtime=dedup(map(to_dep_id, dep.dependencies)),
                    build=dedup(map(to_dep_id, builddeps)),
                )
                self._populate_depgraph(graph, dep.dependencies)
                self._populate_depgraph(graph, builddeps)

    def _get_builddeps(self, dep_id: DepID) -> list[PipGripDep]:
        log.info("getting build dependencies for %s", dep_id)
        if isinstance(dep_id.version, URL):
            return self._get_builddeps_for_url_dep(dep_id.version)
        return self._get_builddeps_for_pypi_dep(dep_id.name, dep_id.version)

    def _get_builddeps_for_pypi_dep(
        self, name: NormalizedName, version: Version
    ) -> list[PipGripDep]:
        page = self._pypi_client.get_project_page(name)
        dists = [
            pkg
            for pkg in page.packages
            if pkg.version and Version(pkg.version) == version
        ]
        if self._config.allow_wheels and any(
            dist.package_type == "wheel" for dist in dists
        ):
            return []

        try:
            sdist = next(dist for dist in dists if dist.package_type == "sdist")
        except StopIteration:
            return []

        sdist_path = self._cache.get_sdist_cache() / sdist.filename
        if not sdist_path.exists():
            self._pypi_client.download_package(sdist, sdist_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.unpack_archive(sdist_path, tmpdir)
            project_dir = _find_project_dir(tmpdir)
            return self._get_builddeps_from_project_dir(project_dir)

    def _get_builddeps_for_url_dep(self, url: URL) -> list[PipGripDep]:
        if url.scheme == "file":
            return self._get_builddeps_for_file_dep(Path(url.path))
        if url.scheme == "git" or url.scheme.startswith("git+"):
            return self._get_builddeps_for_git_dep(url)
        if url.scheme in ("http", "https"):
            return self._get_builddeps_for_http_dep(url)
        raise ValueError(f"unsupported scheme in url: {url}")

    def _get_builddeps_for_file_dep(self, path: Path) -> list[PipGripDep]:
        if path.is_dir():
            return self._get_builddeps_from_project_dir(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.unpack_archive(path, tmpdir)
            project_dir = _find_project_dir(tmpdir)
            return self._get_builddeps_from_project_dir(project_dir)

    def _get_builddeps_for_git_dep(self, url: URL) -> list[PipGripDep]:
        scheme = url.scheme.removeprefix("git+")
        path, _, ref = url.path.partition("@")
        subdirectory = parse_qs(url.fragment).get("subdirectory", ["."])[0]
        repo_url = url.with_scheme(scheme).with_path(path).with_fragment("")

        with tempfile.TemporaryDirectory() as tmpdir:
            _run_cmd(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    str(repo_url),
                    tmpdir,
                ]
            )
            _run_cmd(["git", "checkout", ref or "HEAD"], cwd=tmpdir)
            project_dir = Path(tmpdir) / subdirectory
            return self._get_builddeps_from_project_dir(project_dir)

    def _get_builddeps_for_http_dep(self, url: URL) -> list[PipGripDep]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive_path = tmp_path / "archive.tar.gz"
            unpack_dir = tmp_path / "unpacked"

            self._download_file(str(url), archive_path)
            shutil.unpack_archive(archive_path, unpack_dir)
            project_dir = _find_project_dir(unpack_dir)
            return self._get_builddeps_from_project_dir(project_dir)

    def _get_builddeps_from_project_dir(self, project_dir: Path) -> list[PipGripDep]:
        def subprocess_runner(
            cmd: Sequence[PathType],
            cwd: PathType | None = None,
            extra_environ: Mapping[str, str] | None = None,
        ) -> None:
            _run_cmd(cmd, cwd, extra_environ, logger=None)

        try:
            with build.env.DefaultIsolatedEnv() as env:
                builder = build.ProjectBuilder.from_isolated_env(
                    env, project_dir, subprocess_runner
                )
                builddeps = builder.build_system_requires

                env.install(builddeps)
                builddeps.update(builder.get_requires_for_build("wheel"))
        except build.BuildException:
            return []

        return self.resolve_deps(builddeps)

    def _download_file(self, url: str, path: Path) -> None:
        with requests.get(
            url, stream=True, timeout=self._config.request_timeout
        ) as resp, path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)


def _find_project_dir(unpack_dir: PathType) -> Path:
    unpack_dir = Path(unpack_dir)
    if len(files := list(unpack_dir.iterdir())) == 1 and files[0].is_dir():
        return unpack_dir / files[0]
    return unpack_dir


def _run_cmd(
    cmd: Sequence[PathType],
    cwd: PathType | None = None,
    extra_environ: Mapping[str, str] | None = None,
    logger: logging.Logger | None = log,
) -> str:
    env = os.environ.copy()
    if extra_environ:
        env.update(extra_environ)

    if logger:
        logger.debug("running %s", list(map(str, cmd)))
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        log.error(
            "`%s` failed.\nSTDOUT:\n%s\nSTDERR:%s",
            " ".join(map(str, cmd)),
            p.stdout,
            p.stderr,
        )
    p.check_returncode()
    return p.stdout


_DirectDependents = _DirectDeps


def invert_graph(graph: DepGraph) -> dict[DepID, _DirectDependents]:
    dependents: dict[DepID, _DirectDependents] = defaultdict(
        lambda: _DirectDependents(runtime=[], build=[])
    )
    for dependent, dependencies in graph.items():
        for runtime_dep in dependencies.runtime:
            dependents[runtime_dep].runtime.append(dependent)
        for build_dep in dependencies.build:
            dependents[build_dep].build.append(dependent)

    return dependents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=".")
    ap.add_argument("--allow-wheels", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    cache = Cache(Path("~/.cache/bufson").expanduser())
    config = Config(allow_wheels=args.allow_wheels)
    pypi_client = pypi_simple.PyPISimple()

    bufson = Bufson(cache=cache, config=config, pypi_client=pypi_client)

    with contextlib.chdir(args.path):
        resolved = bufson.resolve_deps(["."])

    resolved[0].name = "root"
    resolved[0].version = Path(args.path).resolve().as_uri()

    graph = bufson.build_depgraph(resolved)

    def print_graph(
        root: DepID, graph: DepGraph, depth: int = 0, prefix: str = ""
    ) -> None:
        indent = " " * 2 * depth
        print(indent, prefix, root, sep="")
        for d in graph[root].runtime:
            print_graph(d, graph, depth + 1, "(R) ")
        for d in graph[root].build:
            print_graph(d, graph, depth + 1, "(B) ")

    def print_pins(dependents_map: dict[DepID, _DirectDependents]) -> None:
        for dep, dependents in dependents_map.items():
            print(dep)
            dependents_repr = [str(rtd) for rtd in dependents.runtime]
            dependents_repr.extend(f"build({bd})" for bd in dependents.build)

            if not dependents_repr:
                via = None
            if len(dependents_repr) == 1:
                via = f"    # via {dependents_repr[0]}"
            else:
                via = "\n".join(
                    ["    # via", *(f"    #   {d}" for d in dependents_repr)]
                )

            if via:
                print(via)

    print_pins(invert_graph(graph))


if __name__ == "__main__":
    main()
