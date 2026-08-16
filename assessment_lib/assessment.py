import cv2
import numpy as np


class Assessment:
    MAX_DIST = 0.05
    DECAY_RATE = 3.0
    OVERLAP_THRESHOLD = 0.05

    def __init__(self, img_shape=(640, 640)):
        self.H, self.W = img_shape[:2]

    def _to_pixels(self, data):
        if isinstance(data, list):
            return np.array(
                [[int(p[0] * self.W), int(p[1] * self.H)] for p in data],
                dtype=np.int32,
            )
        return (int(data[0] * self.W), int(data[1] * self.H))

    def _decay(self, distance):
        max_dist = self.W * self.MAX_DIST
        if distance > max_dist:
            return 0.0
        if distance <= 0:
            return 1.0
        return float(np.exp(-self.DECAY_RATE * (distance / max_dist)))

    def _overlap_score(self, part_poly, dmg_poly):
        mask_part = np.zeros((self.H, self.W), dtype=np.uint8)
        mask_dmg = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.fillPoly(mask_part, [part_poly], 255)
        cv2.fillPoly(mask_dmg, [dmg_poly], 255)

        area_part = cv2.countNonZero(mask_part)
        if area_part == 0:
            return 0.0

        area_inter = cv2.countNonZero(cv2.bitwise_and(mask_part, mask_dmg))
        if area_inter == 0:
            return 0.0

        pct = area_inter / area_part
        if pct < self.OVERLAP_THRESHOLD:
            return (pct / self.OVERLAP_THRESHOLD) * 0.6
        return min(1.0, 0.6 + ((pct - self.OVERLAP_THRESHOLD) / (1.0 - self.OVERLAP_THRESHOLD)) * 0.4)

    def _min_distance(self, part_poly, dmg_poly):
        x, y, w, h = cv2.boundingRect(part_poly)
        dx, dy, dw, dh = cv2.boundingRect(dmg_poly)
        x_gap = max(0, dx - (x + w), x - (dx + dw))
        y_gap = max(0, dy - (y + h), y - (dy + dh))
        if x_gap > self.W * self.MAX_DIST or y_gap > self.H * self.MAX_DIST:
            return float("inf")

        min_dist = float("inf")
        for pt in part_poly:
            dist = cv2.pointPolygonTest(dmg_poly, (int(pt[0]), int(pt[1])), True)
            if dist >= 0:
                return 0.0
            min_dist = min(min_dist, abs(dist))
        return min_dist

    def assess(self, damage_result, reconstruction_result):
        damage_polys = [
            self._to_pixels(det["polygon"])
            for det in damage_result.get("detected", [])
            if det.get("polygon")
        ]

        all_parts = list(reconstruction_result.get("detected", {}).keys()) + list(
            reconstruction_result.get("missing_predicted", {}).keys()
        )

        if not damage_polys:
            return {part: 0.0 for part in all_parts}

        scores = {}

        for part, norm_data in reconstruction_result.get("detected", {}).items():
            if not isinstance(norm_data, list):
                continue
            part_poly = self._to_pixels(norm_data)
            best = 0.0
            for dmg in damage_polys:
                ov = self._overlap_score(part_poly, dmg)
                best = max(best, ov if ov > 0 else self._decay(self._min_distance(part_poly, dmg)))
            scores[part] = round(best, 3)

        for part, norm_pt in reconstruction_result.get("missing_predicted", {}).items():
            px, py = self._to_pixels(norm_pt)
            best = 0.0
            for dmg in damage_polys:
                dist = cv2.pointPolygonTest(dmg, (px, py), True)
                if dist >= 0:
                    best = 1.0
                    break
                best = max(best, min(1.0, self._decay(abs(dist)) * 1.15))
            scores[part] = round(best, 3)

        return scores
