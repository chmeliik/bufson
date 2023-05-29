from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Self
from urllib.parse import SplitResult as URL
from urllib.parse import parse_qs, urlsplit

if TYPE_CHECKING:
    from collections.abc import Iterable


import build
import pydantic
import pypi_simple
from packaging.utils import NormalizedName, canonicalize_name
from packaging.version import InvalidVersion, Version

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

logging.getLogger("build").disabled = True

pypi_client = pypi_simple.PyPISimple()

cfg = {
    "allow_wheels": False,
}


class PipGripDep(pydantic.BaseModel):
    name: str
    version: str
    dependencies: list[PipGripDep] = []


def resolve_deps(requirements: Iterable[str]) -> list[PipGripDep]:
    requirements = list(requirements)
    if not requirements:
        return []
    cmd = ["pipgrip", "--tree", "--json", *requirements]
    log.info("running %s", cmd)
    p = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    return list(map(PipGripDep.parse_obj, json.loads(p.stdout)))


class DepID(NamedTuple):
    name: NormalizedName
    version: Version | URL

    def __str__(self) -> str:
        if isinstance(self.version, Version):
            return f"{self.name}=={self.version}"
        return f"{self.name} @ {self.version.geturl()}"

    @classmethod
    def parse(cls, name: str, raw_version: str) -> Self:
        normalized_name = canonicalize_name(name)
        try:
            version = Version(raw_version)
        except InvalidVersion:
            version = urlsplit(raw_version)

        return cls(normalized_name, version)


def to_dep_id(d: PipGripDep) -> DepID:
    return DepID.parse(d.name, d.version)


class _DirectDeps(NamedTuple):
    runtime: list[DepID]
    build: list[DepID]


DepGraph = dict[DepID, _DirectDeps]


def to_depgraph(pipgrip_deps: Iterable[PipGripDep]) -> DepGraph:
    root = DepID.parse("-root-", "file:.")
    graph: DepGraph = {
        root: _DirectDeps(runtime=[to_dep_id(d) for d in pipgrip_deps], build=[])
    }
    _populate_depgraph(graph, pipgrip_deps)
    return graph


def _populate_depgraph(graph: DepGraph, pipgrip_deps: Iterable[PipGripDep]) -> None:
    def dedup(x: Iterable[DepID]) -> list[DepID]:
        return list(dict.fromkeys(x))

    for dep in pipgrip_deps:
        dep_id = to_dep_id(dep)

        if dep_id not in graph:
            builddeps = _get_builddeps(dep_id)
            graph[dep_id] = _DirectDeps(
                runtime=dedup(map(to_dep_id, dep.dependencies)),
                build=dedup(map(to_dep_id, builddeps)),
            )
            _populate_depgraph(graph, dep.dependencies)
            _populate_depgraph(graph, builddeps)


def _get_builddeps(dep_id: DepID) -> list[PipGripDep]:
    if isinstance(dep_id.version, Version):
        page = pypi_client.get_project_page(dep_id.name)
        dists = [
            pkg
            for pkg in page.packages
            if pkg.version and Version(pkg.version) == dep_id.version
        ]
        if cfg["allow_wheels"] and any(dist.package_type == "wheel" for dist in dists):
            return []

        try:
            sdist = next(dist for dist in dists if dist.package_type == "sdist")
        except StopIteration:
            return []

        unpack_dir = Path(sdist.filename).with_suffix(".unpacked")
        if not unpack_dir.exists():
            pypi_client.download_package(sdist, sdist.filename)
            shutil.unpack_archive(sdist.filename, unpack_dir)

        project_dir = next(unpack_dir.iterdir())

    elif (url := dep_id.version).scheme.startswith("git"):
        scheme = url.scheme.removeprefix("git+")
        path, _, ref = url.path.lstrip("/").rpartition("@")
        repo_dir = Path(url.hostname or ".") / path

        if not repo_dir.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    url._replace(scheme=scheme, path=path, fragment="").geturl(),
                    repo_dir,
                ],
                check=True,
            )
            subprocess.run(["git", "checkout", ref], cwd=repo_dir, check=True)

        fragments = parse_qs(url.fragment)
        subdirectory = fragments.get("subdirectory", ["."])[0]
        project_dir = repo_dir / subdirectory

    else:
        filename = Path(url.path).name
        unpack_dir = Path(filename).with_suffix(".unpacked")
        if not unpack_dir.exists():
            subprocess.run(["curl", "-fsS", "-o", filename, url.geturl()], check=True)
            shutil.unpack_archive(filename, unpack_dir)

        if len(files := list(unpack_dir.iterdir())) == 1 and files[0].is_dir():
            project_dir = unpack_dir / files[0]
        else:
            project_dir = unpack_dir

    try:
        with build.env.DefaultIsolatedEnv() as env:
            builder = build.ProjectBuilder.from_isolated_env(env, project_dir)
            build_deps = builder.build_system_requires

            env.install(build_deps)
            build_deps.update(builder.get_requires_for_build("wheel"))
    except build.BuildException:
        return []

    return resolve_deps(build_deps)


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


class Hash(NamedTuple):
    algorithm: str
    digest: str


def main() -> None:
    resolved = resolve_deps(["aiohttp", "idna"])
    graph = to_depgraph(resolved)

    def print_graph(
        root: DepID, graph: DepGraph, depth: int = 0, prefix: str = ""
    ) -> None:
        indent = " " * 2 * depth
        print(indent, prefix, root, sep="")
        for d in graph[root].runtime:
            print_graph(d, graph, depth + 1, "(R) ")
        for d in graph[root].build:
            print_graph(d, graph, depth + 1, "(B) ")

    for root in resolved:
        print_graph(to_dep_id(root), graph)

    def print_pins(dependents_map: dict[DepID, _DirectDependents]) -> None:
        for dep, dependents in dependents_map.items():
            print(dep)
            dependents_repr = [str(rtd) for rtd in dependents.runtime]
            dependents_repr.extend(f"build({bd})" for bd in dependents.build)

            if not dependents_repr:
                via = None
            if len(dependents_repr) == 1:
                via = f"  # via {dependents_repr[0]}"
            else:
                via = "\n".join(["  # via", *(f"  #   {d}" for d in dependents_repr)])

            if via:
                print(via)

    print_pins(invert_graph(graph))


if __name__ == "__main__":
    main()
