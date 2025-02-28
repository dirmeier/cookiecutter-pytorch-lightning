# {{ cookiecutter.project_title }}

[![ci](https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}/actions/workflows/ci.yaml/badge.svg)](https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}/actions/workflows/ci.yaml)

> {{ cookiecutter.project_short_description }}

## About

TODO

## Development and installation

First install [`uv`](https://docs.astral.sh/uv/) and call
```shell
uv sync
```

You can then add/remove dependencies using
```shell
uv add/remove <dependency>
```

## Example usage

A denoising diffusion model implementation is provided in `experiments/mnist` that can be trained via

```bash
uv run python experiments/mnist/main.py \
  --workdir=workdir \
  --config=experiments/mnist/config.p
``` 


This will create lock files that can be used to reproduce your experimental results..

## Author

{{cookiecutter.full_name}} <a href="mailto:{{cookiecutter.email}}">{{cookiecutter.email}}</a>
