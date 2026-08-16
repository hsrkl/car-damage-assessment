import json
import cv2

from assessment_lib import Reconstruction, Assessment
from assessment_lib.yolo_loader import YoloLoader

# Weights
YOLO_PARTS = "weights/yolo26_parts_weights.pt"
YOLO_SEVERITY = "weights/yolo26_severity_weights.pt"
YOLO_MINOR = "weights/yolo26_damage_weights.pt"
GEO_DIR = "weights/stat_weights"

# Thresholds
SEVERE_THRESHOLD = 0.6
MINOR_THRESHOLD = 0.2

# Loaded once at cold-start
_reconstruction = Reconstruction(GEO_DIR)
_parts_yolo = YoloLoader(YOLO_PARTS, confidence=0.4)
_severe = YoloLoader(YOLO_SEVERITY, confidence=0.2)
_minor = YoloLoader(YOLO_MINOR, confidence=0.2)
_assessment = Assessment(img_shape=(640, 640))


def _run_assessment(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    parts_out = _parts_yolo.process(img)
    parts = _reconstruction.process_image(parts_out)
    severe_result = _severe.process(img)
    minor_result = _minor.process(img)

    severe_scores = _assessment.assess(severe_result, parts)
    minor_scores = _assessment.assess(minor_result, parts)

    all_parts = set(severe_scores) | set(minor_scores)
    combined = {
        part: max(severe_scores.get(part, 0.0), minor_scores.get(part, 0.0))
        for part in all_parts
    }

    severe = {
        part: round(combined[part], 4)
        for part, score in severe_scores.items()
        if score > SEVERE_THRESHOLD
    }
    minor = {
        part: round(combined[part], 4)
        for part, score in minor_scores.items()
        if score > MINOR_THRESHOLD and part not in severe
    }

    return {
        "image": image_path,
        "summary": {
            "total_parts_assessed": len(combined),
            "severely_damaged_count": len(severe),
            "minor_damaged_count": len(minor),
        },
        "severely_damaged": severe,
        "minor_damaged": minor,
    }


def lambda_handler(event, context):
    image_path = event.get("image_path", "data/test-images/dam.jpeg")
    try:
        result = _run_assessment(image_path)
        return {"statusCode": 200, "body": json.dumps(result, indent=2)}
    except FileNotFoundError as exc:
        return {"statusCode": 404, "body": json.dumps({"error": str(exc)})}
    except Exception as exc:
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}


if __name__ == "__main__":
    response = lambda_handler({"image_path": "damaged-car.jpg"}, context=None)
    print(f"\nStatus: {response['statusCode']}")
    print(response["body"])
