import os
import sys
import importlib.util

def check_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"[OK] {module_name} imported successfully.")
        return True
    except ImportError as e:
        print(f"[FAIL] Could not import {module_name}: {e}")
        return False

def check_file(path):
    if os.path.exists(path):
        print(f"[OK] File found: {path}")
        return True
    else:
        print(f"[FAIL] File not found: {path}")
        return False

def main():
    print("--- BlockLens Health Check ---")
    
    # 1. Check Dependencies
    dependencies = ["streamlit", "PIL", "transformers", "web3", "cv2", "numpy", "exifread", "dotenv", "datasets", "accelerate"]
    all_deps_ok = True
    for dep in dependencies:
        # Handle mapping for some packages
        if dep == "PIL": module = "PIL"
        elif dep == "cv2": module = "cv2"
        elif dep == "dotenv": module = "dotenv"
        else: module = dep
        
        if not check_import(module):
            all_deps_ok = False
            
    # 2. Check Files
    files = ["app.py", "train_model.py", ".env", "requirements.txt"]
    all_files_ok = True
    for f in files:
        if not check_file(f):
            all_files_ok = False

    # 3. Check Model Loading (Quick Test)
    print("\n--- Testing Model Loading (Quick) ---")
    try:
        from transformers import pipeline
        # Load a small/fast model just to check connectivity and library health
        print("Attempting to load 'umm-maybe/AI-image-detector' pipeline...")
        pipe = pipeline("image-classification", model="umm-maybe/AI-image-detector")
        print("[OK] Model loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        all_deps_ok = False

    print("\n--- Summary ---")
    if all_deps_ok and all_files_ok:
        print("✅ Project seems healthy! You can run 'streamlit run app.py'.")
    else:
        print("❌ Issues detected. Please check the logs above.")

if __name__ == "__main__":
    main()
