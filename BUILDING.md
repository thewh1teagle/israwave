# Building

## Build

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation)

```console
uv sync
uv build
```

## Publish

_Get token from https://pypi.org/manage/account/token/ _

```console
UV_PUBLISH_TOKEN="your pypi token" uv publish
```

_Add `--repository testpypi` to upload to test repository_

## Run example

```console
uv sync
uv run examples/play.py
```


See https://bootphon.github.io/phonemizer/install.html

_Enable Logging_

Set `LOG_LEVEL=DEBUG` environment variable.

## Gotchas

On Linux you will need PortAudio and LibSound packages

```console
sudo apt install -y libasound-dev libportaudio2
```