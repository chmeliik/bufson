"""Find transitive PEP517 dependencies."""

import argparse
import contextlib
import logging
import pathlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, NewType

import build
import build.env
import pypi_simple
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

pypi_client = pypi_simple.PyPISimple()


class NormalizedRequirement(Requirement):
    """A Requirement, but with its name canonicalized."""

    def __init__(self, requirement_string: str) -> None:
        """Init a Requirement and canonicalize its name."""
        super().__init__(requirement_string)
        self.name = canonicalize_name(self.name)


PinnedRequirement = NewType("PinnedRequirement", NormalizedRequirement)


_venvs: dict[frozenset[NormalizedRequirement], build.env.DefaultIsolatedEnv] = {}


def _get_direct_builddeps_and_their_transitive_runtime_deps(
    project_dir: Path,
    exit_stack: contextlib.ExitStack,
) -> list[PinnedRequirement]:
    build_system_requires = build.ProjectBuilder(project_dir).build_system_requires

    buildreqs = frozenset(NormalizedRequirement(req) for req in build_system_requires)
    buildsys_env = _venvs.get(buildreqs)
    if not buildsys_env:
        buildsys_env = exit_stack.enter_context(build.env.DefaultIsolatedEnv())
        log.debug("installing %s", build_system_requires)
        buildsys_env.install(build_system_requires)

    builder = build.ProjectBuilder.from_isolated_env(buildsys_env, project_dir)
    try:
        requires_for_build_wheel = builder.get_requires_for_build("wheel")
    except Exception:
        log.exception("failed to get backend-specific build requirements")
        log.warning("assuming there are no extra requirements")
        requires_for_build_wheel = []

    wheelreqs = (
        frozenset(NormalizedRequirement(req) for req in requires_for_build_wheel)
        | buildreqs
    )

    if wheelreqs == buildreqs:
        wheel_env = buildsys_env
    elif (wheel_env := _venvs.get(wheelreqs)) is None:
        wheel_env = exit_stack.enter_context(build.env.DefaultIsolatedEnv())
        wheel_requires = sorted(str(req) for req in wheelreqs)
        log.debug("installing %s", wheel_requires)
        wheel_env.install(wheel_requires)
    _venvs[wheelreqs] = wheel_env

    return _get_installed(wheel_env)


def _get_transitive_builddeps(
    project_dir: Path,
    tmpdir: Path,
    exit_stack: contextlib.ExitStack,
    requirements_file: str | None = None,
) -> list[PinnedRequirement]:
    def search_deps(
        dep: PinnedRequirement,
        seen: set[PinnedRequirement],
    ) -> "Iterator[PinnedRequirement]":
        if dep in seen:
            return
        seen.add(dep)
        log.debug("processing %s", dep)
        dep_dir = _download_and_unpack(dep, tmpdir)
        for dep in _get_direct_builddeps_and_their_transitive_runtime_deps(
            dep_dir, exit_stack
        ):
            yield dep
            yield from search_deps(dep, seen)

    seen = set()
    builddeps = set()

    for direct_builddep in _get_direct_builddeps_and_their_transitive_runtime_deps(
        project_dir, exit_stack
    ):
        builddeps.add(direct_builddep)
        builddeps.update(search_deps(direct_builddep, seen))

    if requirements_file:
        runtime_deps_env = exit_stack.enter_context(build.env.DefaultIsolatedEnv())
        _runpy(runtime_deps_env, ["-m", "pip", "install", "-r", requirements_file])
        for runtime_dep in _get_installed(runtime_deps_env):
            builddeps.update(search_deps(runtime_dep, seen))

    return sorted(builddeps, key=str)


def _download_and_unpack(dep: PinnedRequirement, tmpdir: Path) -> Path:
    project_page = pypi_client.get_project_page(dep.name)
    matches = [
        dist
        for dist in project_page.packages
        if dist.package_type == "sdist"
        and dist.version
        and dep.specifier.contains(dist.version)
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple sdists match {dep}: {matches}")
    if not matches:
        raise RuntimeError(f"sdist not found for {dep}")
    sdist = matches[0]
    archive = tmpdir / sdist.filename
    unpack_dir = archive.with_suffix(".unpacked")
    if not unpack_dir.exists():
        pypi_client.download_package(sdist, archive)
        shutil.unpack_archive(archive, unpack_dir)
    return next(unpack_dir.iterdir())


def _get_installed(env: build.env.DefaultIsolatedEnv) -> list[PinnedRequirement]:
    return [
        PinnedRequirement(requirement)
        for line in _runpy(env, ["-m", "pip", "freeze", "--local", "--all"])
        if (requirement := NormalizedRequirement(line)).name != "pip"
    ]


def _runpy(env: build.env.DefaultIsolatedEnv, args: list[str]) -> list[str]:
    return _run([env.python_executable, "-I", *args])


def _run(cmd: list[str]) -> list[str]:
    p = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    return p.stdout.splitlines()


def main() -> None:
    """Run the CLI."""
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir")
    ap.add_argument("-r", "--requirements")
    args = ap.parse_args()

    project_dir = pathlib.Path(args.project_dir)

    with contextlib.ExitStack() as exit_stack:
        tmpdir = exit_stack.enter_context(
            tempfile.TemporaryDirectory(prefix="bufson-sdists-")
        )
        deps = _get_transitive_builddeps(
            project_dir, Path(tmpdir), exit_stack, args.requirements
        )
        for dep in deps:
            print(dep)

        for breqs, env in _venvs.items():
            print("build-reqs =", ", ".join(sorted(map(str, breqs))) or "N/A")
            for dep in _get_installed(env):
                print("   ", dep)
