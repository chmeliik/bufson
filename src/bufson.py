from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, NamedTuple, NewType, Self, cast
from urllib.parse import parse_qs

import build
import pydantic
import pypi_simple
import requests
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import NormalizedName, canonicalize_name
from packaging.version import InvalidVersion, Version
from yarl import URL

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)7s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


log = _get_logger()


PathType = str | os.PathLike[str]

NormalizedReqStr = NewType("NormalizedReqStr", str)


def _normalize_requirement_string(req_str: str) -> NormalizedReqStr:
    try:
        req = Requirement(req_str)
    except InvalidRequirement:
        return NormalizedReqStr(req_str)

    req.name = canonicalize_name(req.name)
    return NormalizedReqStr(str(req))


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

    def __lt__(self, other: tuple[NormalizedName, Version | URL]) -> bool:
        other_name, other_version = other
        return (self.name, str(self.version)) < (other_name, str(other_version))

    @classmethod
    def parse(cls, name: str, raw_version: str) -> Self:
        normalized_name = canonicalize_name(name)
        try:
            version: Version | URL = Version(raw_version)
        except InvalidVersion:
            version = URL(raw_version)

        return cls(normalized_name, version)


def to_dep_id(dep: PipGripDep) -> DepID:
    return DepID.parse(dep.name, dep.version)


class _DirectDeps(NamedTuple):
    runtime: list[DepID]
    build: list[DepID]


DepGraph = dict[DepID, _DirectDeps]


class Hash(NamedTuple):
    algorithm: str
    digest: str

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.digest}"


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
        self._resolution_cache: dict[frozenset[NormalizedReqStr], list[PipGripDep]] = {}
        self._expected_hashes: dict[DepID, list[Hash]] = {}

    def resolve_deps(self, requirements: Iterable[str]) -> list[PipGripDep]:
        normalized_requirements = list(map(_normalize_requirement_string, requirements))
        if not normalized_requirements:
            return []
        return self._cached_resolve_deps(normalized_requirements)

    def _cached_resolve_deps(
        self, requirements: list[NormalizedReqStr]
    ) -> list[PipGripDep]:
        key = frozenset(requirements)
        if (resolved := self._resolution_cache.get(key)) is not None:
            log.debug(
                "already resolved runtime dependencies for %s", ", ".join(requirements)
            )
            return resolved

        log.debug("resolving runtime dependencies for %s", ", ".join(requirements))
        cmd: list[str | PathType] = [
            "pipgrip",
            "--cache-dir",
            self._cache.pipgrip_cache,
            "--tree",
            "--json",
            *requirements,
        ]
        output = _run_cmd(cmd)

        resolved = list(map(PipGripDep.parse_obj, json.loads(output)))
        self._resolution_cache[key] = resolved
        return resolved

    def build_depgraph(self, deps: Iterable[PipGripDep]) -> DepGraph:
        graph: DepGraph = {}
        self._populate_depgraph(graph, deps)
        return graph

    def get_expected_hashes(self) -> dict[DepID, list[Hash]]:
        return self._expected_hashes

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
            return self._get_builddeps_for_url_dep(dep_id)
        return self._get_builddeps_for_pypi_dep(dep_id)

    def _get_builddeps_for_pypi_dep(self, dep_id: DepID) -> list[PipGripDep]:
        name, version = dep_id
        page = self._pypi_client.get_project_page(name)
        dists = [
            pkg
            for pkg in page.packages
            if pkg.version and Version(pkg.version) == version
        ]
        self._expected_hashes[dep_id] = [
            Hash(algorithm, digest)
            for dist in dists
            for algorithm, digest in dist.digests.items()
        ]
        if self._config.allow_wheels and any(
            dist.package_type == "wheel" for dist in dists
        ):
            return []

        try:
            sdist = next(dist for dist in dists if dist.package_type == "sdist")
        except StopIteration:
            log.warning(
                "%s does not distribute sdist, could not get builddeps",
                DepID(name, version),
            )
            return []

        sdist_path = self._cache.get_sdist_cache() / sdist.filename
        if not sdist_path.exists():
            self._pypi_client.download_package(sdist, sdist_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.unpack_archive(sdist_path, tmpdir)
            project_dir = _find_project_dir(tmpdir)
            return self._get_builddeps_from_project_dir(project_dir)

    def _get_builddeps_for_url_dep(self, dep_id: DepID) -> list[PipGripDep]:
        url = cast(URL, dep_id.version)
        if url.scheme == "file":
            return self._get_builddeps_for_file_dep(dep_id)
        if url.scheme == "git" or url.scheme.startswith("git+"):
            return self._get_builddeps_for_git_dep(dep_id)
        if url.scheme in ("http", "https"):
            return self._get_builddeps_for_http_dep(dep_id)
        raise ValueError(f"unsupported scheme in url: {url}")

    def _get_builddeps_for_file_dep(self, dep_id: DepID) -> list[PipGripDep]:
        url = cast(URL, dep_id.version)
        path = Path(url.path)
        if path.is_dir():
            return self._get_builddeps_from_project_dir(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            with path.open("rb") as f:
                sha256 = hashlib.sha256()
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
                self._expected_hashes[dep_id] = [Hash("sha256", sha256.hexdigest())]

            shutil.unpack_archive(path, tmpdir)
            project_dir = _find_project_dir(tmpdir)
            return self._get_builddeps_from_project_dir(project_dir)

    def _get_builddeps_for_git_dep(self, dep_id: DepID) -> list[PipGripDep]:
        url = cast(URL, dep_id.version)
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

    def _get_builddeps_for_http_dep(self, dep_id: DepID) -> list[PipGripDep]:
        url = cast(URL, dep_id.version)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            archive_path = tmp_path / "archive.tar.gz"
            unpack_dir = tmp_path / "unpacked"

            sha256 = self._download_file(str(url), archive_path)
            self._expected_hashes[dep_id] = [sha256]

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

        builddeps = set()
        try:
            with build.env.DefaultIsolatedEnv() as env:
                builder = build.ProjectBuilder.from_isolated_env(
                    env, project_dir, subprocess_runner
                )
                builddeps = builder.build_system_requires
                env.install(builddeps)
                builddeps.update(builder.get_requires_for_build("wheel"))
        except build.BuildException as e:
            log.warning("failed to get build dependencies from pyproject.toml: %s", e)
            log.warning(
                "cannot attempt to get extra build dependencies from get_requires_for_build_wheel"
            )
        except build.BuildBackendException as e:
            log.warning(
                "failed to get extra build dependencies from get_requires_for_build_wheel: %s",
                e,
            )

        return self.resolve_deps(builddeps)

    def _download_file(self, url: str, path: Path) -> Hash:
        sha256 = hashlib.sha256()

        with requests.get(
            url, stream=True, timeout=self._config.request_timeout
        ) as resp, path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                sha256.update(chunk)

        return Hash("sha256", sha256.hexdigest())


def _find_project_dir(unpack_dir: PathType) -> Path:
    unpack_dir = Path(unpack_dir)
    if len(files := list(unpack_dir.iterdir())) == 1 and files[0].is_dir():
        return unpack_dir / files[0]
    return unpack_dir


def _run_cmd(
    cmd: Sequence[str | PathType],
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


DependentsGraph = NewType("DependentsGraph", dict[DepID, _DirectDeps])


def invert_graph(graph: DepGraph) -> DependentsGraph:
    dependents = DependentsGraph(defaultdict(lambda: _DirectDeps(runtime=[], build=[])))

    for dependent, dependencies in graph.items():
        for runtime_dep in dependencies.runtime:
            dependents[runtime_dep].runtime.append(dependent)
        for build_dep in dependencies.build:
            dependents[build_dep].build.append(dependent)

    return dependents


# requirements file: subset of dependents graph such that each *name*
# (not name+version) appears only once


def _find_runtime_deps(deps: DepGraph, root: DepID) -> set[DepID]:
    stack = [root]
    runtime_deps = set()

    while stack:
        pkg = stack.pop()
        runtime_deps.add(pkg)
        stack.extend(dep for dep in deps[pkg].runtime if dep not in runtime_deps)

    runtime_deps.remove(root)
    return runtime_deps


def _split_on_name_conflict(graph: DependentsGraph) -> list[DependentsGraph]:
    graphs: list[DependentsGraph] = []
    names_in_graph: list[set[NormalizedName]] = []

    def first_nonconflicting(name: NormalizedName) -> int | None:
        return next(
            (i for i in range(len(graphs)) if name not in names_in_graph[i]), None
        )

    for dep_id, dependents in graph.items():
        if (i := first_nonconflicting(dep_id.name)) is None:
            graphs.append(DependentsGraph({}))
            names_in_graph.append(set())
            i = len(graphs) - 1

        graphs[i][dep_id] = dependents
        names_in_graph[i].add(dep_id.name)

    return graphs


def print_requirements_file(
    graph: DependentsGraph, expected_hashes: dict[DepID, list[Hash]], f: IO[str]
) -> None:
    for dep, dependents in graph.items():
        hashes = expected_hashes.get(dep, [])
        hash_lines = [f"    --hash={h}" for h in sorted(hashes)]
        dep_str = " \\\n".join([str(dep), *hash_lines])
        print(dep_str, file=f)

        dependents_repr = [str(rtd) for rtd in dependents.runtime]
        dependents_repr.extend(f"build({bd})" for bd in dependents.build)

        if not dependents_repr:
            via = None
        if len(dependents_repr) == 1:
            via = f"    # via {dependents_repr[0]}"
        else:
            via = "\n".join(["    # via", *(f"    #   {d}" for d in dependents_repr)])

        if via:
            print(via, file=f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=".")
    ap.add_argument("--allow-wheels", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-hashes", action="store_true")
    args = ap.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    cache = Cache(Path("~/.cache/bufson").expanduser())
    config = Config(allow_wheels=args.allow_wheels)
    pypi_client = pypi_simple.PyPISimple()

    bufson = Bufson(cache=cache, config=config, pypi_client=pypi_client)

    log.info("processing %s", args.path)
    path = Path(args.path).resolve()

    with contextlib.chdir(path):
        resolved = bufson.resolve_deps(["."])

    resolved[0].name = path.name
    resolved[0].version = path.as_uri()

    graph = bufson.build_depgraph(resolved)
    dependents_graph = invert_graph(graph)

    root = next(node for node in graph if node not in dependents_graph)
    runtime_deps = _find_runtime_deps(graph, root)

    runtime_graph = DependentsGraph(
        {
            dep: dependents
            for dep, dependents in dependents_graph.items()
            if dep in runtime_deps
        }
    )
    build_graph = DependentsGraph(
        {
            dep: dependents
            for dep, dependents in dependents_graph.items()
            if dep not in runtime_deps
        }
    )

    hashes = bufson.get_expected_hashes() if not args.no_hashes else {}

    log.info("writing requirements.txt")
    with Path("requirements.txt").open("w") as f:
        print_requirements_file(runtime_graph, hashes, f)

    for i, req_file in enumerate(_split_on_name_conflict(build_graph)):
        file_name = (
            "requirements-build.txt"
            if i == 0
            else f"requirements-build-conflict{i}.txt"
        )
        log.info("writing %s", file_name)

        with Path(file_name).open("w") as f:
            print_requirements_file(req_file, hashes, f)


if __name__ == "__main__":
    main()
