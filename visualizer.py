#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from assessment_lib import Reconstruction, Assessment
from assessment_lib.yolo_loader import YoloLoader

SEVERITY_COLORS = {
    "minor": (0, 255, 255),
    "moderate": (0, 165, 255),
    "severe": (0, 0, 255),
}
PART_COLOR = (0, 255, 0)
MISSING_COLOR = (255, 100, 0)


def _centroid(polygon):
    if not polygon:
        return None
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _to_px(pt, w, h):
    return int(pt[0] * w), int(pt[1] * h)


def _draw_part(img, name, norm_data, score, w, h):
    if isinstance(norm_data, list) and norm_data:
        pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in norm_data], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], PART_COLOR)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
        cv2.polylines(img, [pts], True, PART_COLOR, 2)
        cx, cy = pts.mean(axis=0).astype(int)
    else:
        if isinstance(norm_data, tuple):
            cx, cy = _to_px(norm_data, w, h)
        else:
            return
        cv2.circle(img, (cx, cy), 6, PART_COLOR, -1)

    label = f"{name} {score:.0%}" if score else name
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (cx - tw // 2 - 2, cy - th - 6), (cx + tw // 2 + 2, cy - 2), (0, 0, 0), -1)
    cv2.putText(img, label, (cx - tw // 2, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_missing(img, name, pt, w, h):
    px, py = _to_px(pt, w, h)
    cv2.circle(img, (px, py), 7, MISSING_COLOR, 2)
    cv2.circle(img, (px, py), 2, MISSING_COLOR, -1)
    label = f"? {name}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(img, (px + 8, py - th - 2), (px + 10 + tw, py + 2), (0, 0, 0), -1)
    cv2.putText(img, label, (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MISSING_COLOR, 1, cv2.LINE_AA)


def _draw_damage(img, det, w, h):
    polygon = det.get("polygon", [])
    cls = det.get("class", "damage")
    conf = det.get("conf", 0.0)
    color = SEVERITY_COLORS.get(cls, (0, 140, 255))

    if polygon:
        pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in polygon], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
        cv2.polylines(img, [pts], True, color, 2)
        cx, cy = pts.mean(axis=0).astype(int)
    else:
        return

    label = f"{cls} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (cx - tw // 2 - 2, cy + 4), (cx + tw // 2 + 2, cy + th + 8), color, -1)
    cv2.putText(img, label, (cx - tw // 2, cy + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_legend(img):
    items = [
        ("Detected part", PART_COLOR),
        ("Missing part (predicted)", MISSING_COLOR),
        ("Minor damage", SEVERITY_COLORS["minor"]),
        ("Moderate damage", SEVERITY_COLORS["moderate"]),
        ("Severe damage", SEVERITY_COLORS["severe"]),
    ]
    x, y = 10, 20
    for label, color in items:
        cv2.circle(img, (x + 6, y), 5, color, -1)
        cv2.putText(img, label, (x + 16, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18


def visualize(image_path, output_path, parts_weights, damage_weights, geo_dir, conf):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    h, w = img.shape[:2]

    recon = Reconstruction(geo_dir)
    parts_yolo = YoloLoader(parts_weights, confidence=conf)
    damage_yolo = YoloLoader(damage_weights, confidence=conf)
    assessment = Assessment(img_shape=(h, w))

    parts_out = parts_yolo.process(img)
    damage_out = damage_yolo.process(img)
    reconstruction = recon.process_image(parts_out)
    scores = assessment.assess(damage_out, reconstruction)

    for det in damage_out.get("detected", []):
        _draw_damage(img, det, w, h)

    for name, norm_data in reconstruction.get("detected", {}).items():
        score = scores.get(name, 0.0)
        _draw_part(img, name, norm_data, score, w, h)

    for name, pt in reconstruction.get("missing_predicted", {}).items():
        _draw_missing(img, name, pt, w, h)

    vp = reconstruction.get("viewpoint")
    if vp:
        cv2.putText(img, f"Viewpoint: {vp}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    _draw_legend(img)

    cv2.imwrite(output_path, img)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize damage assessment results")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-o", "--output", default="output/visualized.jpg", help="Output image path")
    parser.add_argument("--parts-weights", default="weights/yolo26_parts_weights.pt")
    parser.add_argument("--damage-weights", default="weights/yolo26_damage_weights.pt")
    parser.add_argument("--geo-dir", default="weights/stat_weights")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    visualize(args.image, args.output, args.parts_weights, args.damage_weights, args.geo_dir, args.conf)


if __name__ == "__main__":
    main()
