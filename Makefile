.PHONY: install validate sample clean

install:  ## Sync the venv with all dependency groups
	uv sync --all-groups

validate: install  ## Type-check every file in the project
	uv run pyrefly check

sample: install  ## Regenerate the checked-in sample scene
	uv run sitegen generate --seed 1 --duration 60 --out samples/site_seed1.mcap

clean:
	rm -rf .venv .pyrefly_cache
