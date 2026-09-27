# 根据 PEP 517/518，本文件原则上可由 setup.cfg 和 pyproject.toml 替代。
# 原实现说明：当时 pip 的可编辑安装模式 `pip install -e` 仍需要本文件。
# 参考 https://stackoverflow.com/a/60885212

import pathlib

from setuptools import setup


CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """获取 highway-env 的版本号。"""
    path = CWD / "highway_env" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


setup(version=get_version())
