# Building

## Build

```console
python -m pip install -U build
python -m build
```

## Publish

_Get token from https://pypi.org/manage/account/token/ _

```console
python -m pip install -U twine

python -m build
python -m twine upload dist/*
```

_Add `--repository testpypi` to upload to test repository_


See https://bootphon.github.io/phonemizer/install.html