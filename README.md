# Simulating Intel RealSense D435 depth on aluminium parts

This project renders synthetic RGB-D images of aluminium T-slot profiles and L-brackets,
adds the depth errors a real Intel RealSense D435 would show on them, and then checks how
much those errors affect estimating the brackets' pose with ICP.

## Requirements

- Tested on macOS (Apple Silicon) with Python 3.11 and [uv](https://docs.astral.sh/uv/)
- macOS only: `brew install libusb` (needed by the Open3D wheel)
- About 1 GB of disk space, since BlenderProc downloads Blender on first use

## Setup

```bash
make setup
make test   # some tests are skipped until scene 0 has been rendered
```

## Run the pipeline

```bash
make parts        # builds the meshes and data/meshes/manifest.json
make dataset      # renders 100 scenes
make summary      # creates a summary of the dataset
make test         # unit tests: geometry conventions, camera model vs Intel's formulas
make figures      # camera model figure for scene 0, view 0
make experiment   # ICP on ideal vs simulated depth, 16 conditions
make analysis     # paired tests and result figures
```

Quick try: `make render N=3` renders three scenes, and `make vis SCENE=001` opens a frame.

Optional: `make noise-sensitivity` repeats the experiment with three assumed noise correlation lengths.

## Layout

```
configs/default.yaml    all parameters: Intel datasheet values, model assumptions, scene and experiment settings
src/parts.py            procedural T-slot profiles and brackets
src/render.py           BlenderProc scene generation
src/dataset.py          frame loading, Blender to OpenCV conventions
src/camera_model.py     D435 depth model
src/experiment.py       ICP pose estimation
src/analysis.py         statistical tests and result figures
src/figures.py          camera model figure
src/summary.py          dataset statistics
tests/                  pytest checks
data/, output/          generated
```

## Notes

- Every number used by the code is in `configs/default.yaml`. Each rendered scene stores a copy of the config and its seed in `meta.json`, so scenes are reproducible.
- Generated data is not included in the repo, `make dataset` recreates it.
