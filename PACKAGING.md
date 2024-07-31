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
python -m build
python -m twine upload dist/*
```
