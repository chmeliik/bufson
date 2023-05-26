from __future__ import annotations

import json
import logging
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from typing import NamedTuple, Self
from urllib.parse import SplitResult as URL
from urllib.parse import urlsplit

import pydantic
from packaging.utils import NormalizedName, canonicalize_name
from packaging.version import InvalidVersion, Version

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)


class PipGripDep(pydantic.BaseModel):
    name: str
    version: str
    dependencies: list[PipGripDep] = []


def resolve_deps(requirements: Iterable[str]) -> list[PipGripDep]:
    cmd = ["pipgrip", "--tree", "--json", *requirements]
    log.debug("running %s", cmd)
    p = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    return list(map(PipGripDep.parse_obj, json.loads(p.stdout)))


class DepID(NamedTuple):
    name: NormalizedName
    version: Version | URL

    def __str__(self) -> str:
        if isinstance(self.version, Version):
            return f"{self.name}=={self.version}"
        else:
            return f"{self.name} @ {self.version}"

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


def _populate_depgraph(graph: DepGraph, pipgrip_deps: Iterable[PipGripDep]) -> None:
    for dep in pipgrip_deps:
        dep_id = to_dep_id(dep)
        if dep_id not in graph:
            builddeps = _get_builddeps(dep_id)
            graph[dep_id] = _DirectDeps(
                runtime=list(map(to_dep_id, dep.dependencies)),
                build=list(map(to_dep_id, builddeps)),
            )
            _populate_depgraph(graph, dep.dependencies)
            _populate_depgraph(graph, builddeps)


def _get_builddeps(dep_id: DepID) -> list[PipGripDep]:
    # TODO
    return []


def to_depgraph(pipgrip_deps: Iterable[PipGripDep]) -> DepGraph:
    graph: DepGraph = {}
    _populate_depgraph(graph, pipgrip_deps)
    return graph


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


if __name__ == "__main__":
    main()
