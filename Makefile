readme:
	jupyter nbconvert --clear-output --stdout notebook.ipynb | \
		jupyter nbconvert --to markdown --output README.md --stdin

init:
	git config --local include.path ../.gitconfig
	
.PHONY: readme init