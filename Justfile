set shell := ["bash", "-cu"]

default:
    @just --list

install:
    uv sync --group dev --frozen
    uv run --frozen pre-commit install

docs-serve: install
    uv run --frozen sphinx-autobuild docs docs/_build/html --open-browser

docs-build: install docs-clean
    uv run --frozen sphinx-build -b html docs docs/_build/html

docs-clean:
    rm -rf docs/_build docs/jupyter_execute

test:
    uv run --frozen pytest --cov=./ --cov-report=xml

coverage-xml:
    uv run --frozen pytest --cov=highway_env --cov-report=xml

coverage-total:
    uv run --frozen pytest --cov=highway_env --cov-report=term --cov-fail-under=85

# 差异覆盖率：默认对比 origin/main；可按位置传入远程名称和分支。
coverage-diff remote="origin" branch="main": coverage-xml
    uv run --frozen diff-cover coverage.xml --compare-branch={{remote}}/{{branch}} --fail-under=80

coverage: coverage-total coverage-diff
