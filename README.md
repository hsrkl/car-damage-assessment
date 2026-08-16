# Damage Assessment

Car damage assessment library using YOLO-based part detection and geometric reconstruction. Detects car parts, identifies missing parts via geometric models, runs damage detection (severity classification), and scores damage per part.

## How it works

The pipeline has three stages:

1. **Part detection** — A YOLO segmentation model detects 19 car parts (bumpers, doors, lights, etc.) and produces per-part polygons.

2. **Reconstruction** — A geometric model determines the camera viewpoint (FL/FR/BL/BR) from detected parts, then predicts the positions of missing parts using trained spatial relationships between parts.

3. **Damage scoring** — Two YOLO detectors (severity and damage) find damage polygons. Each detected/predicted part is scored 0–1 based on overlap and proximity to damage regions.

## Project structure

```
damage-assessment/
├── assessment_lib/              # Core library
│   ├── yolo_loader.py           # YoloLoader — YOLO model wrapper
│   ├── reconstruction.py        # Reconstruction — viewpoint + missing parts
│   └── assessment.py            # Assessment — damage scoring
├── mock_assessment_lambda.py    # AWS Lambda entrypoint
├── visualizer.py                # CLI visualization script
├── training_scripts/
│   └── stat_trainer.py          # Trains geometric stat models from labels
├── weights/
│   ├── yolo26_parts_weights.pt  # Part detection model
│   ├── yolo26_severity_weights.pt  # Severity classification model
│   ├── yolo26_damage_weights.pt # Damage detection model
│   └── stat_weights/            # Geometric models (model_FL.json, etc.)
└── pyproject.toml
```

## Installation

Requires Python 3.13+.

```bash
uv sync
```

## Usage

### As a library

```python
from assessment_lib import Reconstruction, Assessment
from assessment_lib.yolo_loader import YoloLoader

# Load models
recon = Reconstruction("weights/stat_weights")
parts_yolo = YoloLoader("weights/yolo26_parts_weights.pt", confidence=0.4)
damage_yolo = YoloLoader("weights/yolo26_severity_weights.pt", confidence=0.2)
assess = Assessment(img_shape=(640, 640))

# Run pipeline
parts_out = parts_yolo.process("car.jpg")
reconstruction = recon.process_image(parts_out)
damage_out = damage_yolo.process("car.jpg")
scores = assess.assess(damage_out, reconstruction)
# scores = {"front_bumper": 0.767, "hood": 0.0, ...}
```

### Visualizer

```bash
uv run python visualizer.py car.jpg -o output/vis.jpg
```

Draws detected parts (green), predicted missing parts (blue), and damage polygons (yellow/orange/red by severity) with a legend.

### Lambda

`mock_assessment_lambda.py` exposes a `lambda_handler(event, context)` function. Set the `image_path` key in the event payload:

```json
{"image_path": "s3://bucket/car.jpg"}
```

Returns a JSON response with `severely_damaged`, `minor_damaged`, and a summary.

## Classes

| Class | Module | Purpose |
|---|---|---|
| `YoloLoader` | `yolo_loader` | Wraps a YOLO model. `process(source)` returns `{"detected": [{"class", "conf", "polygon"}]}`. |
| `Reconstruction` | `reconstruction` | Loads geometric stat models. `process_image(yolo_output)` returns `{"viewpoint", "detected", "missing_predicted"}`. |
| `Assessment` | `assessment` | Scores damage per part. `assess(damage_result, reconstruction_result)` returns `{part: score}`. |

## Training

The geometric stat models are trained from labelled YOLO datasets using `training_scripts/stat_trainer.py`. It computes average spatial offsets between part pairs for each viewpoint and saves them as JSON.
