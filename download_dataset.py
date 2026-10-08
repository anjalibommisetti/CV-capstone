"""
download_dataset.py
Dataset Acquisition & Splitting for PlantVillage (38 Classes, 14 Crops).

Supported Methods:
1. Direct Zip Download (Standard Python, no external libraries required)
2. Hugging Face Datasets: "mohanty/PlantVillage" (Optional)
3. Kaggle API: "emmarex/plantdisease" (Optional)
4. Local Splitter: 80% Train, 10% Validation, 10% Test
"""

import os
import sys
import shutil
import random
import zipfile
import urllib.request
import importlib
from pathlib import Path


def split_dataset(
    raw_dir: str, 
    output_dir: str = "dataset", 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Splits raw class folders into 80% Train, 10% Validation, and 10% Test sets.
    Preserves class directory structure for tf.keras.preprocessing.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-4, "Ratios must sum to 1.0"
    random.seed(seed)
    
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Source folder does not exist: {raw_dir}")
        
    class_folders = [f for f in raw_path.iterdir() if f.is_dir()]
    if not class_folders:
        raise ValueError(f"No class subfolders found inside {raw_dir}")
        
    print(f"\n[INFO] Found {len(class_folders)} class directories in {raw_dir}")
    print(f"[INFO] Splitting into Train ({int(train_ratio*100)}%), Val ({int(val_ratio*100)}%), Test ({int(test_ratio*100)}%)...")
    
    splits = ["train", "val", "test"]
    for s in splits:
        for cf in class_folders:
            (out_path / s / cf.name).mkdir(parents=True, exist_ok=True)
            
    total_copied = 0
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG", ".JPEG"}
    
    for cf in class_folders:
        images = [img for img in cf.iterdir() if img.suffix.lower() in valid_exts]
        random.shuffle(images)
        n = len(images)
        if n == 0:
            continue
            
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        for img in train_imgs:
            shutil.copy2(img, out_path / "train" / cf.name / img.name)
        for img in val_imgs:
            shutil.copy2(img, out_path / "val" / cf.name / img.name)
        for img in test_imgs:
            shutil.copy2(img, out_path / "test" / cf.name / img.name)
            
        total_copied += n
        print(f"  Organized {cf.name[:32]:<32} -> {len(train_imgs):>4} train | {len(val_imgs):>3} val | {len(test_imgs):>3} test")
        
    print(f"\n[SUCCESS] Split completed: {total_copied} images placed in '{output_dir}'.\n")


def download_direct_zip(url: str, dest_dir: str = "raw_plantvillage"):
    """
    Downloads and extracts a zip archive directly using Python standard libraries.
    """
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "dataset.zip")
    
    print(f"[INFO] Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("[INFO] Extracting archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print(f"[SUCCESS] Dataset downloaded and extracted into '{dest_dir}'.")
        return dest_dir
    except Exception as e:
        print(f"[ERROR] Direct download failed: {e}")
        return None


def download_from_huggingface(output_raw: str = "raw_plantvillage"):
    """
    Downloads the PlantVillage dataset from Hugging Face if datasets is available.
    """
    print("\n[INFO] Checking Hugging Face datasets library...")
    try:
        datasets_mod = importlib.import_module("datasets")
        load_dataset = getattr(datasets_mod, "load_dataset")
        print("[INFO] Loading dataset 'mohanty/PlantVillage' from Hugging Face...")
        ds = load_dataset("mohanty/PlantVillage")
        
        os.makedirs(output_raw, exist_ok=True)
        for split_name in ds.keys():
            for idx, item in enumerate(ds[split_name]):
                image = item["image"]
                label_name = item.get("label", "unknown")
                if isinstance(label_name, int) and hasattr(ds[split_name].features["label"], "int2str"):
                    label_name = ds[split_name].features["label"].int2str(label_name)
                    
                target_dir = os.path.join(output_raw, str(label_name))
                os.makedirs(target_dir, exist_ok=True)
                image.save(os.path.join(target_dir, f"img_{idx}.jpg"))
                if idx % 1000 == 0 and idx > 0:
                    print(f"  Extracted {idx} images...")
                    
        print(f"[SUCCESS] Hugging Face extraction complete: {output_raw}")
        return output_raw
    except ImportError:
        print("[TIP] 'datasets' package is not installed. To use Hugging Face, run: pip install datasets")
        return None
    except Exception as e:
        print(f"[NOTE] Hugging Face download note: {e}")
        return None


def download_from_kaggle(output_raw: str = "raw_plantvillage"):
    """
    Downloads the PlantVillage dataset from Kaggle if kagglehub is installed.
    """
    print("\n[INFO] Checking kagglehub...")
    try:
        kagglehub_mod = importlib.import_module("kagglehub")
        dataset_download = getattr(kagglehub_mod, "dataset_download")
        path = dataset_download("emmarex/plantdisease")
        print(f"[SUCCESS] Kaggle download completed at: {path}")
        return path
    except ImportError:
        print("[TIP] 'kagglehub' is optional. To use Kaggle direct downloader, run: pip install kagglehub")
        return None
    except Exception as e:
        print(f"[NOTE] Kaggle download note: {e}")
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and split PlantVillage dataset (80/10/10).")
    parser.add_argument("--source", choices=["local", "huggingface", "kaggle"], default="local",
                        help="Data source to download from")
    parser.add_argument("--raw_dir", type=str, default="raw_plantvillage",
                        help="Directory containing raw class folders")
    parser.add_argument("--output_dir", type=str, default="dataset",
                        help="Target output directory for train/val/test split")
    
    args = parser.parse_args()
    source_folder = args.raw_dir
    
    if args.source == "huggingface":
        dl = download_from_huggingface(args.raw_dir)
        if dl:
            source_folder = dl
    elif args.source == "kaggle":
        dl = download_from_kaggle(args.raw_dir)
        if dl:
            source_folder = dl
            
    if os.path.exists(source_folder):
        split_dataset(source_folder, output_dir=args.output_dir)
    else:
        print(f"\n=======================================================")
        print(f"  PLANTVILLAGE DATASET SETUP GUIDE")
        print(f"=======================================================")
        print(f"Folder '{source_folder}' was not found yet.")
        print(f"To download the dataset:")
        print(f"  Option 1 (Google Colab / Kaggle CLI):")
        print(f"    kaggle datasets download -d emmarex/plantdisease --unzip -p raw_plantvillage")
        print(f"    python download_dataset.py --source local")
        print(f"")
        print(f"  Option 2 (Hugging Face):")
        print(f"    pip install datasets")
        print(f"    python download_dataset.py --source huggingface")
        print(f"")
        print(f"  Option 3 (Quick Local Test with Sample Leaves):")
        print(f"    python create_sample_data.py")
        print(f"    (Creates sample Tomato & Apple classes in 'dataset/' instantly!)")
        print(f"=======================================================\n")
