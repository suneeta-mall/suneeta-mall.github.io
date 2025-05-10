.PHONY: install serve fmt clean
.DEFAULT_GOAL := serve

install:
	pip install -r requirements.txt

serve:
	mkdocs serve

fmt:
	ruff format .
	ruff check --fix .

clean:
	git clean -Xdf
