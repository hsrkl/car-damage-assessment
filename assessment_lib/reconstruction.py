import json
import os


class Reconstruction:
    def __init__(self, stat_model_dir):
        self.classes = {
            0: "boot",
            1: "front_bumper",
            2: "front_door",
            3: "front_fender",
            4: "front_grill",
            5: "front_windshield",
            6: "head_light",
            7: "hood",
            8: "quarter_panel",
            9: "rear_bumper",
            10: "rear_door",
            11: "rear_number_plate",
            12: "rear_windshield",
            13: "roof",
            14: "running_board",
            15: "side_view_mirror",
            16: "tail_lamp",
            17: "tyre",
            18: "under_chassis",
        }
        self._name_to_id = {name: cid for cid, name in self.classes.items()}

        self.anchors = {
            "FRONT": [1, 4, 5, 6, 7],
            "REAR": [0, 9, 11, 12, 16],
            "SIDE": [2, 3, 8, 10, 14, 15],
        }

        self.stat_models = {}
        for view in ("FL", "FR", "BL", "BR"):
            path = os.path.join(stat_model_dir, f"model_{view}.json")
            if os.path.exists(path):
                with open(path) as f:
                    self.stat_models[view] = json.load(f)
            else:
                self.stat_models[view] = {}

    @staticmethod
    def _centroid_from_polygon(polygon):
        if not polygon:
            return None
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _extract_centroids(self, detections):
        named = {}
        id_centroids = {}
        for det in detections:
            name = det["class"]
            polygon = det.get("polygon", [])
            pt = self._centroid_from_polygon(polygon)
            if pt is None:
                continue
            named[name] = pt
            cid = self._name_to_id.get(name)
            if cid is not None:
                id_centroids[cid] = pt
        return id_centroids, named

    def determine_viewpoint(self, detections_id_centroids):
        detected_ids = list(detections_id_centroids.keys())

        f_score = sum(1 for i in detected_ids if i in self.anchors["FRONT"])
        r_score = sum(1 for i in detected_ids if i in self.anchors["REAR"])

        if f_score >= r_score and f_score > 0:
            main_view = "FRONT"
            end_ids = [i for i in detected_ids if i in self.anchors["FRONT"]]
        elif r_score > f_score:
            main_view = "REAR"
            end_ids = [i for i in detected_ids if i in self.anchors["REAR"]]
        else:
            return None

        side_ids = [i for i in detected_ids if i in self.anchors["SIDE"]]
        if not side_ids or not end_ids:
            return None

        avg_end_x = sum(detections_id_centroids[i][0] for i in end_ids) / len(end_ids)
        avg_side_x = sum(detections_id_centroids[i][0] for i in side_ids) / len(side_ids)

        if main_view == "FRONT":
            return "FL" if avg_side_x > avg_end_x else "FR"
        else:
            return "BL" if avg_side_x > avg_end_x else "BR"

    def predict_missing(self, detections_named_centroids, viewpoint):
        if viewpoint not in self.stat_models:
            return {}

        model = self.stat_models[viewpoint]
        predictions = {}

        for target in model.keys():
            if target in detections_named_centroids:
                continue

            votes_x, votes_y = [], []
            for ref_part, ref_coords in detections_named_centroids.items():
                if ref_part in model and target in model[ref_part]:
                    dx, dy = model[ref_part][target]
                    votes_x.append(ref_coords[0] + dx)
                    votes_y.append(ref_coords[1] + dy)

            if votes_x:
                predictions[target] = (
                    float(sum(votes_x) / len(votes_x)),
                    float(sum(votes_y) / len(votes_y)),
                )

        return predictions

    def process_image(self, yolo_output):
        detections = yolo_output.get("detected", [])
        id_centroids, named_centroids = self._extract_centroids(detections)

        viewpoint = self.determine_viewpoint(id_centroids)
        predicted = self.predict_missing(named_centroids, viewpoint) if viewpoint else {}

        detected = {}
        for det in detections:
            name = det["class"]
            polygon = det.get("polygon", [])
            if polygon:
                detected[name] = polygon
            else:
                pt = self._centroid_from_polygon(polygon)
                if pt is not None:
                    detected[name] = pt

        return {
            "viewpoint": viewpoint,
            "detected": detected,
            "missing_predicted": predicted,
        }
