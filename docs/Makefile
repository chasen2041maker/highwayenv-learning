# 用于 Sphinx 文档的最简 Makefile
#

# 可以从命令行设置这些变量，
# 前两个变量也可以通过环境变量设置。
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# 放在最前面，使不带参数的 "make" 等同于 "make help"。
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# 通用目标：使用新的 "make mode" 选项，
# 将所有未知目标交给 Sphinx；$(O) 是 $(SPHINXOPTS) 的简写。
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
