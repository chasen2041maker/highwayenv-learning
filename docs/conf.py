# Sphinx 文档构建器的配置文件。
#
# 本文件只包含部分常用选项。完整选项列表
# 请参阅文档：
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- 路径设置 --------------------------------------------------------------

# 如果扩展模块或需要 autodoc 生成文档的模块位于其他目录，
# 请在这里将其目录加入 sys.path。若路径相对于文档根目录，
# 可像下面的示例一样，使用 os.path.abspath 将其转换为绝对路径。
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- 项目信息 -----------------------------------------------------
import os
from typing import Any, Dict

import highway_env  # noqa: F401


project = "HighwayEnv"
copyright = "2023 Farama Foundation"
author = "Farama Foundation"

# 完整版本号，包含 alpha、beta、rc 标签
# release = highway_env.__version__
release = ""

# -- 常规配置 ---------------------------------------------------

# 在这里以字符串形式添加 Sphinx 扩展模块名称。
# 可以使用 Sphinx 自带的扩展（名称为 'sphinx.ext.*'），
# 也可以使用自定义扩展。
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
    "jupyter_sphinx",
    "myst_parser",
    "sphinx_github_changelog",
]

autodoc_default_flags = [
    "members",
    "private-members",
    "undoc-members",
    "special-members",
]
autodoc_member_order = "bysource"

# 在这里添加包含模板的路径，路径相对于当前目录。
templates_path = ["_templates"]

# 匹配需要忽略的文件和目录的模式列表，
# 查找源文件时使用，路径相对于源目录。
# 这些模式也会影响 html_static_path 和 html_extra_path。
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Napoleon 扩展设置
# napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
# 参考 https://github.com/sphinx-doc/sphinx/issues/9119
napoleon_custom_sections = [("Returns", "params_style")]

# Autodoc 自动生成文档设置
autoclass_content = "both"
autodoc_preserve_defaults = True

# -- 自动章节标签配置 ----------------------------------------
autosectionlabel_prefix_document = True

# -- MyST 配置 -----------------------------------------------------
myst_enable_extensions = [
    "dollarmath",
]

# -- HTML 输出选项 -------------------------------------------------

# 用于 HTML 和 HTML 帮助页面的主题。
# 内置主题列表请参阅文档。
#
html_theme = "celshast"
html_title = "HighwayEnv Documentation"
html_baseurl = "https://highway-env.farama.org"
html_copy_source = False
html_favicon = "_static/img/highway-favicon.png"
html_theme_options = {
    "light_logo": "img/highway.svg",
    "dark_logo": "img/highway-white.svg",
    "image": "img/highway-github.png",
    "gtag": "G-6H9C8TWXZ8",
    "description": "A collection of environments for autonomous driving and tactical decision-making tasks",
    "versioning": True,
    "source_repository": "https://github.com/Farama-Foundation/HighwayEnv/",
    "source_branch": "main",
    "source_directory": "docs/",
}
html_context: Dict[str, Any] = {}
html_context["conf_py_path"] = "/docs/"
html_context["display_github"] = False
html_context["github_user"] = "Farama-Foundation"
html_context["github_repo"] = "HighwayEnv"
html_context["github_version"] = "main"
html_context["slug"] = "HighwayEnv"

html_static_path = ["_static"]
html_css_files = []


# -- BibTeX 参考文献设置 -------------------------------------------------------------

bibtex_bibfiles = ["bibliography/biblio.bib"]
bibtex_encoding = "latin"
bibtex_default_style = "alpha"

# -- 生成更新日志 -------------------------------------------------

sphinx_github_changelog_token = os.environ.get("SPHINX_GITHUB_CHANGELOG_TOKEN")
