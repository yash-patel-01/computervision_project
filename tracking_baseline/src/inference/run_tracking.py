import json
import yaml
from pathlib import Path
from typing import List
from ..tracking.tracker import Tracker, Detection, format_tracks


def load_detections(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['frames']


def main(detections_json: str, config_yaml: str, out_path: str):
    with open(config_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    tracker = Tracker(cfg)
    frames = load_detections(detections_json)
    results = []
    for frame in frames:
        frame_id = frame['frame_id']
        det_objs: List[Detection] = []
        for det in frame['detections']:
            det_objs.append(Detection(bbox=det['bbox'], embedding=det.get('embedding'), class_id=det['class_id']))
        tracker.predict()
        tracker.update(det_objs)
        results.append({
            'frame_id': frame_id,
            'tracks': format_tracks(tracker.active_tracks())
        })
    with open(out_path, 'w') as f:
        json.dump({'frames': results}, f)
    print(f"Saved tracking results to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--detections', required=True, help='Path to detections JSON')
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[2] / 'configs' / 'tracker.yaml'))
    parser.add_argument('--out', default='tracks.json')
    args = parser.parse_args()
    main(args.detections, args.config, args.out)
