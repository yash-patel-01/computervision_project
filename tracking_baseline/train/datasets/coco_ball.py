"""Minimal COCO-style dataset loader for player/ball detection.
Loads COCO-format annotations (coco_train.json or coco_test.json) and returns
images + target dicts compatible with torchvision detection models.
"""
from __future__ import annotations
from pathlib import Path
import json
from PIL import Image
import torch

class CocoBallDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path | str, coco_json: Path | str, transforms=None, limit=None):
        self.root = Path(root)
        self.coco_path = Path(coco_json)
        with open(self.coco_path, 'r') as f:
            data = json.load(f)
        # Remap potentially string image ids to contiguous integer indices
        self.orig_to_new = {}
        self.id_to_file = {}
        images = data.get('images', [])
        for idx, img in enumerate(images):
            new_id = idx  # zero-based contiguous id
            self.orig_to_new[img['id']] = new_id
            self.id_to_file[new_id] = img['file_name']
        # Group annotations by new image id
        anns_by_img: dict[int, list] = {}
        for ann in data.get('annotations', []):
            orig_id = ann['image_id']
            new_id = self.orig_to_new.get(orig_id)
            if new_id is None:
                continue
            anns_by_img.setdefault(new_id, []).append(ann)
        self.img_ids = sorted(self.id_to_file.keys())
        if limit:
            self.img_ids = self.img_ids[:limit]
        self.anns_by_img = anns_by_img
        self.transforms = transforms
        self.categories = {cat['id']: cat['name'] for cat in data['categories']}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = int(self.img_ids[idx])
        rel_file = self.id_to_file[img_id]
        # Determine sequence directory if path pattern includes seq name
        # Our file_name currently 'seq/img1/000123.jpg' or just '000123.jpg'
        parts = rel_file.split('/')
        if len(parts) > 1:
            # Use root as the base directory for tracking-2023
            img_path = self.root / rel_file
        else:
            img_path = self.root / rel_file
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert('RGB')
        anns = self.anns_by_img.get(img_id, [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        track_ids = []
        for a in anns:
            x,y,w,h = a['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(a['category_id'])
            areas.append(a['area'])
            iscrowd.append(a.get('iscrowd', 0))
            track_ids.append(a.get('track_id', -1))
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        target = {
            'image_id': torch.tensor(img_id, dtype=torch.int64),
            'orig_image_id': self._find_orig_id(img_id),
            'boxes': boxes,
            'labels': labels,
            'area': areas,
            'iscrowd': iscrowd,
            'track_ids': torch.tensor(track_ids, dtype=torch.int64)
        }
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def _find_orig_id(self, new_id: int):
        # Reverse lookup for original image id string (optional debugging aid)
        for orig, nid in self.orig_to_new.items():
            if nid == new_id:
                return orig
        return None


def collate_fn(batch):
    return tuple(zip(*batch))
