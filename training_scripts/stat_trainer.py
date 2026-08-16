import os
import json
import glob
from collections import defaultdict

# --- CONFIGURATION ---
DATA_ROOT = './data'
VIEWPOINTS = ['FL', 'FR', 'BL', 'BR']
OUTPUT_DIR = 'weights/stat-weights'

# Yolo26 Class Map
CLASSES = {
    0: 'boot', 1: 'front_bumper', 2: 'front_door', 3: 'front_fender',
    4: 'front_grill', 5: 'front_windshield', 6: 'head_light', 7: 'hood',
    8: 'quarter_panel', 9: 'rear_bumper', 10: 'rear_door', 11: 'rear_number_plate',
    12: 'rear_windshield', 13: 'roof', 14: 'running_board', 15: 'side_view_mirror',
    16: 'tail_lamp', 17: 'tyre', 18: 'under_chassis'
}

# --- HARDCODED VISIBILITY MASKS ---
# Only these parts are allowed to be predicted for these viewpoints.
# We exclude parts that are physically on the other side or hidden.
VISIBLE_PARTS = {
    'FL': [
        'front_bumper', 'front_door', 'front_fender', 'front_grill', 
        'front_windshield', 'head_light', 'hood', 'roof', 
        'running_board', 'side_view_mirror', 'tyre', 'under_chassis',
        'rear_door'
    ],
    'FR': [
        'front_bumper', 'front_door', 'front_fender', 'front_grill', 
        'front_windshield', 'head_light', 'hood', 'roof', 
        'running_board', 'side_view_mirror', 'tyre', 'under_chassis',
        'rear_door',
    ],
    'BL': [
        'rear_bumper', 'rear_door', 'quarter_panel', 'rear_windshield', 
        'boot', 'tail_lamp', 'rear_number_plate', 'roof', 
        'running_board', 'side_view_mirror', 'tyre', 'under_chassis',
        'front_door'
    ],
    'BR': [
        'rear_bumper', 'rear_door', 'quarter_panel', 'rear_windshield', 
        'boot', 'tail_lamp', 'rear_number_plate', 'roof', 
        'running_board', 'side_view_mirror', 'tyre', 'under_chassis',
        'front_door'
    ]
}

def train_viewpoint(viewpoint_name, viewpoint_dir):
    """
    Calculates relationships only for parts allowed in VISIBLE_PARTS[viewpoint_name]
    """
    relationships = defaultdict(lambda: defaultdict(list))
    
    # Get allowed list for this specific view
    allowed_names = set(VISIBLE_PARTS.get(viewpoint_name, []))
    
    files = glob.glob(os.path.join(viewpoint_dir, '*', 'labels', '*.txt'))
    print(f"  - Processing {len(files)} files for {viewpoint_name}...")

    for fpath in files:
        current_parts = {} 
        try:
            with open(fpath, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) >= 3:
                        cid = int(data[0])
                        cname = CLASSES[cid]
                        
                        # FILTER 1: If the input data has a part that shouldn't be there 
                        # (e.g. noise), ignore it completely.
                        if cname not in allowed_names:
                            continue
                            
                        cx = float(data[1])
                        cy = float(data[2])
                        current_parts[cid] = (cx, cy)
        except Exception:
            continue

        # Calculate geometric relationships
        ids = list(current_parts.keys())
        for source_id in ids:
            for target_id in ids:
                if source_id == target_id: continue
                
                sx, sy = current_parts[source_id]
                tx, ty = current_parts[target_id]
                
                # We record the relationship. 
                # Note: We don't need to check target_id against allowed_names here 
                # because we already filtered the input list `current_parts` above.
                relationships[source_id][target_id].append((tx - sx, ty - sy))

    # Compress to Averages
    model_json = {}
    
    for src_id, targets in relationships.items():
        src_name = CLASSES[src_id]
        model_json[src_name] = {}
        
        for tgt_id, deltas in targets.items():
            tgt_name = CLASSES[tgt_id]
            
            avg_dx = sum(d[0] for d in deltas) / len(deltas)
            avg_dy = sum(d[1] for d in deltas) / len(deltas)
            
            model_json[src_name][tgt_name] = [round(avg_dx, 5), round(avg_dy, 5)]

    return model_json

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Starting Training...")

    for vp in VIEWPOINTS:
        vp_path = os.path.join(DATA_ROOT, vp)
        if not os.path.exists(vp_path):
            print(f"Skipping {vp} (Folder not found)")
            continue
            
        print(f"Training Model: {vp}")
        # Pass the viewpoint name to access the hardcoded list
        model_data = train_viewpoint(vp, vp_path)
        
        with open(os.path.join(OUTPUT_DIR, f'model_{vp}.json'), 'w') as f:
            json.dump(model_data, f)
            
    print(f"\nDone! Models saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()