# Bufson

**Bu**ild **f**rom **so**urce, <sup>**n**o?</sup>

## Usage

```shell
bufson .
```

Generate requirements.txt, requirements-build.txt for the python project in the
current directory. The project must declare its dependencies in a pip-compatible
way (i.e. `pip install .` would install the dependencies).

:warning: this may overwrite existing requirements.txt and requirements-build\*.txt
files.

Note that bufson may generate additional requirements-build-conflictN.txt files
for conflicting versions of build dependencies.
