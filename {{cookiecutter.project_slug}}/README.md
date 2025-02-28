# {{ cookiecutter.project_title }}

[![ci](https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}/actions/workflows/ci.yaml/badge.svg)](https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}/actions/workflows/ci.yaml)

> {{ cookiecutter.project_short_description }}

## About

TODO

## Example usage

TODO

## Installation

To install the latest GitHub <RELEASE>, just call the following on the
command line:

```bash
pip install git+https://github.com/{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}@<RELEASE>
```

## Development

First install [`uv`](https://docs.astral.sh/uv/) and call
```shell
uv sync
```

You can then add/remove dependencies using
```shell
uv add/remove <dependency>
```

This will create lock files that can be used to reproduce your experimental results..

## Author

{{cookiecutter.full_name}} <a href="mailto:{{cookiecutter.email}}">{{cookiecutter.email}}</a>
