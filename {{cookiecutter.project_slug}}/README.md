# {{ cookiecutter.project_title }}

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
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

First install [`rye`](https://rye.astral.sh/) and call
```shell
rye sync
```

You can then add/remove dependencies using
```shell
rye add/remove <dependency>
```

This will create lock files that can be used to reproduce your experimental results..


## Author

{{cookiecutter.full_name}} <a href="mailto:{{cookiecutter.email}}">{{cookiecutter.email}}</a>
