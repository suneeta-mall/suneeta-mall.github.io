.PHONY: lock install fmt serve clean uv upgrade
.DEFAULT_GOAL := serve

uv:
	@python -m pip install -Uq uv --disable-pip-version-check

lock: uv
	uv pip compile -o requirements.txt \
		--no-emit-index-url \
		--generate-hashes \
		--extra dev \
		--extra docs \
		pyproject.toml
	@echo "Please check in the generated <requirements.txt> to the repository"

upgrade: uv
	uv pip compile -o requirements.txt \
		--upgrade \
		--no-emit-index-url \
		--generate-hashes \
		--extra dev \
		--extra docs \
		pyproject.toml

install: uv
	uv pip install \
		--no-compile \
		--no-deps \
		-r requirements.txt
	uv pip install \
		--no-deps \
		--editable .
	uv pip check

deploy:
	mkdocs build #--strict
	
serve:
	mkdocs serve --dev-addr=127.0.0.1:8080 --livereload

fmt:
	ruff format .
	ruff check --fix .

clean:
	git clean -Xdf
