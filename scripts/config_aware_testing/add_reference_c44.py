#!/usr/bin/env python3
"""
Script to add DFT and Experimental C44 reference values to the JSON file.
"""
import json
import sys

def add_reference_values(json_path, dft_c44=None, exp_c44=None):
    """Add DFT and experimental C44 reference values to JSON."""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Add DFT reference if provided
    if dft_c44 is not None:
        data["DFT"] = {
            "model": "DFT Reference",
            "backend": "dft",
            "cubic_constants_gpa": {
                "C44": float(dft_c44)
            }
        }
        print(f"Added DFT reference: C44 = {dft_c44} GPa")
    
    # Add experimental reference if provided
    if exp_c44 is not None:
        data["PREDEXP"] = {
            "model": "Experimental Reference",
            "backend": "experiment",
            "cubic_constants_gpa": {
                "C44": float(exp_c44)
            }
        }
        print(f"Added Experimental reference: C44 = {exp_c44} GPa")
    
    # Write back to file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated {json_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_reference_c44.py <json_file> [dft_c44] [exp_c44]")
        print("\nExample for tungsten (BCC):")
        print("  python add_reference_c44.py allegro_elastic_tensors_summary.json 160.0 163.0")
        sys.exit(1)
    
    json_path = sys.argv[1]
    dft_c44 = float(sys.argv[2]) if len(sys.argv) > 2 else None
    exp_c44 = float(sys.argv[3]) if len(sys.argv) > 3 else None
    
    add_reference_values(json_path, dft_c44, exp_c44)
