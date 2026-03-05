#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk analisis lengkap struktur NPZ file

Justifikasi:
- Cek semua level nested structure
- Validate compatibility dengan VideoPose3D
- Identify potential issues
"""

import numpy as np
import sys
import os
from pprint import pprint

def print_dict_structure(d, indent=0, max_items=5):
    """
    Print nested dictionary structure dengan indentation
    
    Args:
        d: dictionary to print
        indent: current indentation level
        max_items: maksimum items untuk di-print per level
    """
    prefix = "  " * indent
    
    if isinstance(d, dict):
        items = list(d.items())[:max_items]
        for key, value in items:
            print(f"{prefix}'{key}': {type(value).__name__}", end="")
            
            if isinstance(value, np.ndarray):
                print(f" shape={value.shape} dtype={value.dtype}")
            elif isinstance(value, dict):
                print(f" (dict with {len(value)} keys)")
                print_dict_structure(value, indent + 1, max_items)
            elif isinstance(value, list):
                print(f" (list with {len(value)} items)")
                if len(value) > 0:
                    print(f"{prefix}  [0]: {type(value[0]).__name__}", end="")
                    if isinstance(value[0], np.ndarray):
                        print(f" shape={value[0].shape}")
                    else:
                        print()
            else:
                print(f" = {value}")
        
        if len(d) > max_items:
            print(f"{prefix}... and {len(d) - max_items} more items")
    else:
        print(f"{prefix}{type(d).__name__}: {d}")


def check_npz_structure(npz_path, verbose=True):
    """
    Comprehensive analysis of NPZ structure
    
    Args:
        npz_path: path ke file .npz
        verbose: print detailed information
    """
    print("="*80)
    print(f"ANALYZING NPZ STRUCTURE: {os.path.basename(npz_path)}")
    print("="*80)
    
    # Check file exists
    if not os.path.exists(npz_path):
        print(f"ERROR: File not found: {npz_path}")
        return False
    
    # Check file size
    file_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    print(f"\nFile size: {file_size_mb:.2f} MB")
    
    # Load NPZ
    print("\n" + "-"*80)
    print("STEP 1: Loading NPZ file")
    print("-"*80)
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        print(f"✓ Successfully loaded")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return False
    
    # Check top-level keys
    print("\n" + "-"*80)
    print("STEP 2: Top-level keys")
    print("-"*80)
    
    top_keys = list(data.keys())
    print(f"Keys found: {top_keys}")
    
    # Expected keys for VideoPose3D
    expected_keys = ['positions_2d', 'metadata']
    missing_keys = set(expected_keys) - set(top_keys)
    
    if missing_keys:
        print(f"✗ MISSING expected keys: {missing_keys}")
        return False
    else:
        print(f"✓ All expected keys present")
    
    # Check positions_2d
    print("\n" + "-"*80)
    print("STEP 3: Analyzing 'positions_2d'")
    print("-"*80)
    
    positions_raw = data['positions_2d']
    print(f"Type (raw): {type(positions_raw)}")
    print(f"Shape (raw): {positions_raw.shape if hasattr(positions_raw, 'shape') else 'N/A'}")
    print(f"Dtype (raw): {positions_raw.dtype if hasattr(positions_raw, 'dtype') else 'N/A'}")
    
    # Convert to dict (CRITICAL step)
    try:
        if isinstance(positions_raw, np.ndarray):
            print("\nConverting to dict using .item()...")
            positions = positions_raw.item()
        else:
            positions = positions_raw
        
        print(f"Type (after .item()): {type(positions)}")
        
        if not isinstance(positions, dict):
            print(f"✗ ERROR: positions_2d is not a dict after .item()")
            return False
        
        print(f"✓ Successfully converted to dict")
        print(f"Number of videos: {len(positions)}")
        
    except Exception as e:
        print(f"✗ Failed to convert: {e}")
        return False
    
    # Analyze structure of positions_2d
    print("\n" + "-"*80)
    print("STEP 4: Structure of positions_2d (first 3 videos)")
    print("-"*80)
    
    for i, video_name in enumerate(list(positions.keys())[:3], 1):
        print(f"\nVideo {i}: '{video_name}'")
        print(f"  Type: {type(positions[video_name])}")
        
        if isinstance(positions[video_name], dict):
            print(f"  Keys (actions): {list(positions[video_name].keys())}")
            
            for action_name, action_data in positions[video_name].items():
                print(f"\n  Action: '{action_name}'")
                print(f"    Type: {type(action_data)}")
                
                if isinstance(action_data, list):
                    print(f"    Length: {len(action_data)}")
                    
                    if len(action_data) > 0:
                        print(f"    [0] Type: {type(action_data[0])}")
                        
                        if isinstance(action_data[0], np.ndarray):
                            print(f"    [0] Shape: {action_data[0].shape}")
                            print(f"    [0] Dtype: {action_data[0].dtype}")
                            
                            # Validate shape
                            expected_dims = 3  # [frames, joints, 2]
                            if len(action_data[0].shape) != expected_dims:
                                print(f"    ✗ WARNING: Expected {expected_dims} dimensions, got {len(action_data[0].shape)}")
                            
                            # Check for NaN
                            num_nan = np.isnan(action_data[0]).sum()
                            if num_nan > 0:
                                print(f"    ⚠ Contains {num_nan} NaN values")
                            
                            # Value range
                            valid_data = action_data[0][~np.isnan(action_data[0])]
                            if len(valid_data) > 0:
                                print(f"    Value range: [{valid_data.min():.2f}, {valid_data.max():.2f}]")
                        else:
                            print(f"    [0] Value: {action_data[0]}")
                else:
                    print(f"    Value: {action_data}")
        else:
            print(f"  Value: {positions[video_name]}")
    
    # Check metadata
    print("\n" + "-"*80)
    print("STEP 5: Analyzing 'metadata'")
    print("-"*80)
    
    metadata_raw = data['metadata']
    print(f"Type (raw): {type(metadata_raw)}")
    
    try:
        if isinstance(metadata_raw, np.ndarray):
            metadata = metadata_raw.item()
        else:
            metadata = metadata_raw
        
        print(f"Type (after .item()): {type(metadata)}")
        
        if not isinstance(metadata, dict):
            print(f"✗ ERROR: metadata is not a dict")
            return False
        
        print(f"✓ Successfully converted to dict")
        print(f"\nMetadata keys: {list(metadata.keys())}")
        
    except Exception as e:
        print(f"✗ Failed to convert: {e}")
        return False
    
    # Check required metadata fields
    print("\n" + "-"*80)
    print("STEP 6: Validating metadata fields")
    print("-"*80)
    
    required_fields = ['layout_name', 'num_joints']
    optional_fields = ['keypoints_symmetry', 'video_metadata']
    
    print("\nRequired fields:")
    for field in required_fields:
        if field in metadata:
            print(f"  ✓ '{field}': {metadata[field]}")
        else:
            print(f"  ✗ '{field}': MISSING")
    
    print("\nOptional fields:")
    for field in optional_fields:
        if field in metadata:
            value = metadata[field]
            if isinstance(value, dict):
                print(f"  ✓ '{field}': dict with {len(value)} items")
            elif isinstance(value, list):
                print(f"  ✓ '{field}': list with {len(value)} items")
            else:
                print(f"  ✓ '{field}': {value}")
        else:
            print(f"  - '{field}': not present")
    
    # Check video_metadata structure
    if 'video_metadata' in metadata:
        print("\n" + "-"*80)
        print("STEP 7: Analyzing video_metadata (first 3 videos)")
        print("-"*80)
        
        video_metadata = metadata['video_metadata']
        
        for i, video_name in enumerate(list(video_metadata.keys())[:3], 1):
            print(f"\nVideo {i}: '{video_name}'")
            vm = video_metadata[video_name]
            
            # Check required fields for VideoPose3D
            required_vm_fields = ['w', 'h']
            
            for field in required_vm_fields:
                if field in vm:
                    print(f"  ✓ '{field}': {vm[field]}")
                else:
                    print(f"  ✗ '{field}': MISSING (REQUIRED!)")
            
            # Optional fields
            if 'fps' in vm:
                print(f"  • 'fps': {vm['fps']}")
    
    # Validation summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    checks = {
        'NPZ file exists': os.path.exists(npz_path),
        'positions_2d present': 'positions_2d' in data,
        'positions_2d is dict': isinstance(positions, dict),
        'metadata present': 'metadata' in data,
        'metadata is dict': isinstance(metadata, dict),
        'layout_name in metadata': 'layout_name' in metadata,
        'num_joints in metadata': 'num_joints' in metadata,
        'video_metadata present': 'video_metadata' in metadata,
    }
    
    # Check video structure
    if len(positions) > 0:
        first_video = list(positions.keys())[0]
        first_video_data = positions[first_video]
        
        checks['Video has actions'] = isinstance(first_video_data, dict) and len(first_video_data) > 0
        
        if checks['Video has actions']:
            first_action = list(first_video_data.keys())[0]
            first_action_data = first_video_data[first_action]
            
            checks['Action is list'] = isinstance(first_action_data, list)
            
            if checks['Action is list'] and len(first_action_data) > 0:
                checks['Data is ndarray'] = isinstance(first_action_data[0], np.ndarray)
                
                if checks['Data is ndarray']:
                    checks['Data has 3 dims'] = len(first_action_data[0].shape) == 3
    
    # Check video metadata for first video
    if 'video_metadata' in metadata and len(metadata['video_metadata']) > 0:
        first_vm_name = list(metadata['video_metadata'].keys())[0]
        first_vm = metadata['video_metadata'][first_vm_name]
        
        checks['Video has width (w)'] = 'w' in first_vm
        checks['Video has height (h)'] = 'h' in first_vm
    
    # Print results
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("RESULT: ✓ ALL CHECKS PASSED - Dataset structure is valid!")
    else:
        print("RESULT: ✗ SOME CHECKS FAILED - Please fix the issues above")
    print("="*80)
    
    return all_passed


def main():
    """
    Main function
    """
    # Auto-detect NPZ files
    data_dir = '/workspace/VideoPose3D/data'
    
    if len(sys.argv) > 1:
        # Use provided file path
        npz_path = sys.argv[1]
        check_npz_structure(npz_path)
    else:
        # Check all URFD NPZ files
        import glob
        npz_files = sorted(glob.glob(os.path.join(data_dir, 'data_2d_custom_urfd_*.npz')))
        
        if len(npz_files) == 0:
            print(f"No NPZ files found in {data_dir}")
            print("Usage: python check_npz_structure.py [path/to/file.npz]")
            sys.exit(1)
        
        print(f"Found {len(npz_files)} NPZ files\n")
        
        for npz_file in npz_files:
            check_npz_structure(npz_file)
            print("\n" * 2)


if __name__ == '__main__':
    main()
