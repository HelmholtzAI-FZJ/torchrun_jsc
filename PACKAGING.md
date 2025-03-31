# Packaging `torchrun_jsc`

## Setting up

May have to adapt `venv` sourcing for your operating system.

```shell
python3 -m venv ./env
source env/bin/activate
python -m pip install -U pip
python -m pip install -U build twine
```

## Packaging and pushing a new version

```shell
source env/bin/activate
python -m build
# Optionally with `--skip-existing`.
python -m twine upload dist/*
```
