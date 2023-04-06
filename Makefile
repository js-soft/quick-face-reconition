init:
	# Sets up the git filter to strip output from jupyter notebooks
	git config --local include.path ../.gitconfig
	
.PHONY: init