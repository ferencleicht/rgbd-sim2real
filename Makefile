.PHONY: setup parts render dataset vis all clean summary figures fmt lint test experiment analysis noise-sensitivity

setup:
	uv sync
	uv run blenderproc pip install pyyaml

parts:
	uv run python src/parts.py

render: parts
	for i in $$(seq 0 $$(( $(or $(N),1) - 1 ))); do \
		uv run blenderproc run src/render.py --scene-id $$i || exit 1; \
	done

dataset: parts
	$(MAKE) render N=100

vis:
	uv run blenderproc vis hdf5 data/renders/scene_$(or $(SCENE),000)/$(or $(VIEW),0).hdf5 \
		--depth_max $(or $(DMAX),1)

summary:
	uv run python src/summary.py

figures:
	uv run python src/figures.py --scene $(or $(SCENE),0) --view $(or $(VIEW),0)

fmt:
	uv run ruff check --select I --fix src tests
	uv run ruff format src tests

lint:
	uv run ruff check src tests

all: dataset

clean:
	rm -rf data/ output/

test:
	uv run pytest -v

experiment:
	uv run python src/experiment.py

analysis:
	uv run python src/analysis.py

noise-sensitivity:
	for c in 1 4 8; do \
		uv run python src/experiment.py --noise-corr-px $$c \
			--conditions ideal,full,only_noise --tag noise$$c || exit 1; \
	done
