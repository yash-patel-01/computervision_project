"""
Basic tests to verify the project structure and code validity.
These tests don't require dependencies to be installed.
"""

import os
import json
import yaml
import ast


def test_project_structure():
    """Test that all required files and directories exist."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    required_files = [
        'README.md',
        'requirements.txt',
        'setup.py',
        'example_usage.py',
        'configs/default_config.yaml',
        'data/train/annotations.json',
        'data/val/annotations.json',
        'src/__init__.py',
        'src/data/__init__.py',
        'src/data/dataset.py',
        'src/models/__init__.py',
        'src/models/detector.py',
        'src/utils/__init__.py',
        'src/utils/visualization.py',
        'src/train.py',
        'src/inference.py',
    ]
    
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        assert os.path.exists(full_path), f"Missing required file: {file_path}"
    
    print("✓ All required files exist")


def test_python_syntax():
    """Test that all Python files have valid syntax."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    python_files = [
        'src/data/dataset.py',
        'src/models/detector.py',
        'src/utils/visualization.py',
        'src/train.py',
        'src/inference.py',
        'example_usage.py',
        'setup.py',
    ]
    
    for file_path in python_files:
        full_path = os.path.join(base_dir, file_path)
        with open(full_path, 'r') as f:
            code = f.read()
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise AssertionError(f"Syntax error in {file_path}: {e}")
    
    print("✓ All Python files have valid syntax")


def test_json_validity():
    """Test that JSON annotation files are valid."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    json_files = [
        'data/train/annotations.json',
        'data/val/annotations.json',
    ]
    
    for file_path in json_files:
        full_path = os.path.join(base_dir, file_path)
        with open(full_path, 'r') as f:
            data = json.load(f)
        
        # Verify COCO format structure
        assert 'images' in data, f"{file_path}: Missing 'images' key"
        assert 'annotations' in data, f"{file_path}: Missing 'annotations' key"
        assert 'categories' in data, f"{file_path}: Missing 'categories' key"
        
        # Verify categories
        assert len(data['categories']) == 2, f"{file_path}: Expected 2 categories"
        category_names = [cat['name'] for cat in data['categories']]
        assert 'player' in category_names, f"{file_path}: Missing 'player' category"
        assert 'ball' in category_names, f"{file_path}: Missing 'ball' category"
    
    print("✓ All JSON files are valid and follow COCO format")


def test_yaml_validity():
    """Test that YAML config file is valid."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config_path = os.path.join(base_dir, 'configs/default_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Verify required sections
    assert 'model' in config, "Config: Missing 'model' section"
    assert 'training' in config, "Config: Missing 'training' section"
    assert 'data' in config, "Config: Missing 'data' section"
    
    # Verify key parameters
    assert config['model']['num_classes'] == 3, "Config: Expected 3 classes"
    assert config['training']['epochs'] > 0, "Config: Epochs must be positive"
    
    print("✓ Config file is valid")


def test_annotation_format():
    """Test that annotations have correct format."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    ann_path = os.path.join(base_dir, 'data/train/annotations.json')
    with open(ann_path, 'r') as f:
        data = json.load(f)
    
    # Test annotation structure
    for ann in data['annotations']:
        assert 'id' in ann, "Annotation missing 'id'"
        assert 'image_id' in ann, "Annotation missing 'image_id'"
        assert 'category_id' in ann, "Annotation missing 'category_id'"
        assert 'bbox' in ann, "Annotation missing 'bbox'"
        
        # Test bbox format [x, y, width, height]
        bbox = ann['bbox']
        assert len(bbox) == 4, "Bbox must have 4 values"
        assert all(isinstance(x, (int, float)) for x in bbox), "Bbox values must be numeric"
        assert bbox[2] > 0 and bbox[3] > 0, "Bbox width and height must be positive"
    
    print("✓ Annotations have correct format")


if __name__ == '__main__':
    print("\nRunning project structure tests...\n")
    
    tests = [
        test_project_structure,
        test_python_syntax,
        test_json_validity,
        test_yaml_validity,
        test_annotation_format,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print(f"{'='*60}\n")
    
    if failed > 0:
        exit(1)
