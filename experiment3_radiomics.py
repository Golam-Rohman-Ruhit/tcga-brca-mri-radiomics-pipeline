"""
=====================================================================
EXPERIMENT 3: Radiomics Feature Extraction from MRI Data
=====================================================================
Extracts radiomics imaging features of tumor ROI regions
using pyradiomics and saves features as txt and csv files.
=====================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import logging

logging.getLogger('radiomics').setLevel(logging.ERROR)

try:
    from radiomics import featureextractor
    print("PyRadiomics loaded successfully!")
except ImportError:
    print("ERROR: Run: pip install pyradiomics")
    sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"C:\Users\Asus\PycharmProjects\medical"
EXP2_RESULTS = os.path.join(BASE_DIR, "experiment2_results")
EXP3_RESULTS = os.path.join(BASE_DIR, "experiment3_results")

os.makedirs(EXP3_RESULTS, exist_ok=True)


def find_files():
    """Find the image and mask files from Experiment 2."""
    print(f"  Scanning: {EXP2_RESULTS}")

    if not os.path.exists(EXP2_RESULTS):
        print(f"  ERROR: {EXP2_RESULTS} does not exist!")
        return None, None

    files = os.listdir(EXP2_RESULTS)
    print(f"  Files: {files}")

    image_path = None
    mask_path = None

    # Search priority order for IMAGE
    image_names = [
        "original_image.nrrd", "original_image.nii.gz",
        "subtraction_image.nrrd", "subtraction_image.nii.gz",
    ]
    for name in image_names:
        path = os.path.join(EXP2_RESULTS, name)
        if os.path.exists(path):
            image_path = path
            break

    # Search priority order for MASK
    mask_names = [
        "tumor_mask.nrrd", "tumor_mask.nii.gz",
        "dce_subtraction_tumor_mask.nrrd", "dce_subtraction_tumor_mask.nii.gz",
    ]
    for name in mask_names:
        path = os.path.join(EXP2_RESULTS, name)
        if os.path.exists(path):
            mask_path = path
            break

    # Fallback: find any nrrd/nii.gz files
    if image_path is None or mask_path is None:
        volumes = [f for f in files if f.endswith(('.nrrd', '.nii.gz'))]
        for v in volumes:
            vl = v.lower()
            full = os.path.join(EXP2_RESULTS, v)
            if image_path is None and 'mask' not in vl and 'seg' not in vl:
                image_path = full
            if mask_path is None and ('mask' in vl or 'seg' in vl):
                mask_path = full

    return image_path, mask_path


def verify_and_fix_mask(image_path, mask_path):
    """
    Verify mask is valid. If geometry doesn't match image, fix it.
    Returns the (possibly corrected) mask path.
    """
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    mask_arr = sitk.GetArrayFromImage(mask)
    n_tumor = int(mask_arr.sum())
    print(f"  Mask labels: {np.unique(mask_arr)}")
    print(f"  Tumor voxels: {n_tumor}")

    if n_tumor == 0:
        print("  ERROR: Mask is empty!")
        return None

    # Check geometry match
    size_match = (image.GetSize() == mask.GetSize())
    spacing_close = all(
        abs(a - b) < 0.01
        for a, b in zip(image.GetSpacing(), mask.GetSpacing())
    )
    origin_close = all(
        abs(a - b) < 1.0
        for a, b in zip(image.GetOrigin(), mask.GetOrigin())
    )

    if size_match and spacing_close and origin_close:
        print("  Geometry matches: OK")
        return mask_path

    print("  Geometry mismatch detected, correcting mask...")
    print(f"    Image: size={image.GetSize()}, spacing={image.GetSpacing()}")
    print(f"    Mask:  size={mask.GetSize()}, spacing={mask.GetSpacing()}")

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_fixed = resampler.Execute(mask)

    fixed_path = os.path.join(EXP3_RESULTS, "corrected_mask.nrrd")
    sitk.WriteImage(mask_fixed, fixed_path)

    check = sitk.GetArrayFromImage(mask_fixed)
    print(f"  Corrected mask: {int(check.sum())} tumor voxels")
    print(f"  Saved: {fixed_path}")

    return fixed_path


def extract_features(image_path, mask_path):
    """Extract radiomics features."""
    settings = {
        'binWidth': 25,
        'resampledPixelSpacing': None,
        'interpolator': sitk.sitkBSpline,
        'verbose': False,
        'label': 1,
        'geometryTolerance': 1e-2,
        'correctMask': True,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllFeatures()

    print("\n  Enabled feature classes:")
    for cls in extractor.enabledFeatures:
        print(f"    - {cls}")

    print("\n  Extracting features (this may take 1-2 minutes)...")

    try:
        result = extractor.execute(image_path, mask_path)
        print("  Extraction successful!")
        return result
    except Exception as e:
        print(f"  Failed: {e}")

        # Try with resampled data
        print("  Retrying with resampled isotropic data...")
        try:
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)
            mask = sitk.Cast(mask, sitk.sitkUInt8)

            new_spacing = [1.0, 1.0, 1.0]
            orig_size = image.GetSize()
            orig_spacing = image.GetSpacing()
            new_size = [int(round(orig_size[i] * orig_spacing[i] / new_spacing[i]))
                        for i in range(3)]

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetSize(new_size)
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetTransform(sitk.Transform())

            resampler.SetInterpolator(sitk.sitkLinear)
            img_r = resampler.Execute(image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mask_r = resampler.Execute(mask)

            img_path_r = os.path.join(EXP3_RESULTS, "resampled_image.nrrd")
            mask_path_r = os.path.join(EXP3_RESULTS, "resampled_mask.nrrd")
            sitk.WriteImage(img_r, img_path_r)
            sitk.WriteImage(mask_r, mask_path_r)

            result = extractor.execute(img_path_r, mask_path_r)
            print("  Extraction successful with resampled data!")
            return result
        except Exception as e2:
            print(f"  All attempts failed: {e2}")
            return None


def organize_features(result):
    """Sort features into categories."""
    features = {}
    diagnostics = {}

    for key, value in result.items():
        key_str = str(key)
        if 'diagnostics' in key_str:
            diagnostics[key_str] = value
        else:
            try:
                features[key_str] = float(value)
            except (ValueError, TypeError):
                features[key_str] = str(value)

    cats = {
        'shape': {}, 'firstorder': {}, 'glcm': {},
        'glrlm': {}, 'glszm': {}, 'gldm': {}, 'ngtdm': {},
    }

    for key, val in features.items():
        placed = False
        for cat in cats:
            if cat in key.lower():
                cats[cat][key] = val
                placed = True
                break

    return features, cats, diagnostics


def save_features_txt(features, cats, save_dir):
    """Save all features as formatted text."""
    path = os.path.join(save_dir, "radiomics_features.txt")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  RADIOMICS FEATURES - TUMOR ROI REGION\n")
        f.write("  Patient: TCGA-AO-A03M\n")
        f.write("  Extracted using PyRadiomics\n")
        f.write("=" * 80 + "\n")

        total = 0
        for cat, feats in cats.items():
            if not feats:
                continue
            f.write(f"\n{'=' * 60}\n")
            f.write(f"  {cat.upper()} FEATURES ({len(feats)} features)\n")
            f.write(f"{'=' * 60}\n\n")
            for name, val in sorted(feats.items()):
                if isinstance(val, float):
                    f.write(f"  {name}: {val:.6f}\n")
                else:
                    f.write(f"  {name}: {val}\n")
                total += 1

        f.write(f"\n{'=' * 80}\n")
        f.write(f"  TOTAL FEATURES: {total}\n")
        f.write(f"{'=' * 80}\n")

    print(f"  TXT saved: {path}")
    return path


def save_features_csv(features, save_dir):
    """Save features as CSV."""
    csv_path = os.path.join(save_dir, "radiomics_features.csv")
    df = pd.DataFrame([features])
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}")

    summary_path = os.path.join(save_dir, "features_summary.csv")
    rows = []
    for name, val in sorted(features.items()):
        cat = "other"
        for c in ['shape', 'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']:
            if c in name.lower():
                cat = c
                break
        rows.append({'Category': cat, 'Feature_Name': name, 'Value': val})
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"  Summary CSV saved: {summary_path}")

    return csv_path


def visualize_features(cats, save_dir):
    """Create charts of the extracted features."""

    # ---- Chart 1: Feature count per category ----
    counts = {c.upper(): len(f) for c, f in cats.items() if f}
    if counts:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12',
                  '#9b59b6', '#1abc9c', '#e67e22']
        bars = ax.bar(list(counts.keys()), list(counts.values()),
                      color=colors[:len(counts)], edgecolor='black', linewidth=0.5)
        ax.set_title('Radiomics Features by Category', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        for b, c in zip(bars, counts.values()):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                    str(c), ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "features_by_category.png"), dpi=150)
        plt.show()

    # ---- Chart 2: Top first-order features ----
    fo = cats.get('firstorder', {})
    numeric = {k: v for k, v in fo.items() if isinstance(v, float) and np.isfinite(v)}
    if numeric:
        top = sorted(numeric.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        names = [t[0].split('_')[-1] for t in top]
        vals = [t[1] for t in top]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(vals)), vals, color='#3498db', edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_title('Top First-Order Features', fontsize=14, fontweight='bold')
        ax.set_xlabel('Value')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "firstorder_features.png"), dpi=150)
        plt.show()

    # ---- Chart 3: Shape features ----
    sh = cats.get('shape', {})
    numeric_sh = {k: v for k, v in sh.items() if isinstance(v, float) and np.isfinite(v)}
    if numeric_sh:
        fig, ax = plt.subplots(figsize=(14, 6))
        names = [n.split('_')[-1] for n in numeric_sh.keys()]
        vals = list(numeric_sh.values())
        ax.bar(range(len(vals)), vals, color='#2ecc71', edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_title('Shape Features', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shape_features.png"), dpi=150)
        plt.show()

    # ---- Chart 4: Top 30 all features ----
    all_num = {}
    for cat, feats in cats.items():
        for k, v in feats.items():
            if isinstance(v, float) and np.isfinite(v):
                short = k.split('_', 2)[-1] if k.count('_') >= 2 else k
                all_num[short] = v
    if len(all_num) > 5:
        top30 = sorted(all_num.items(), key=lambda x: abs(x[1]), reverse=True)[:30]
        names = [t[0] for t in top30]
        vals = [t[1] for t in top30]
        colors_map = ['#e74c3c' if v < 0 else '#3498db' for v in vals]

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.barh(range(len(vals)), vals, color=colors_map, edgecolor='black', linewidth=0.3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title('Top 30 Radiomics Features', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.axvline(x=0, color='black', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "top30_features.png"), dpi=150)
        plt.show()

    print("  All charts saved.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  EXPERIMENT 3: Radiomics Feature Extraction")
    print("=" * 70)

    # Step 1: Find files
    print("\n[Step 1] Finding image and mask files...")
    image_path, mask_path = find_files()

    if image_path is None:
        print("  ERROR: Image not found! Run Experiment 2 first.")
        sys.exit(1)
    if mask_path is None:
        print("  ERROR: Mask not found! Run Experiment 2 first.")
        sys.exit(1)

    print(f"  Image: {image_path}")
    print(f"  Mask:  {mask_path}")

    # Step 2: Verify mask
    print("\n[Step 2] Verifying mask...")
    valid_mask_path = verify_and_fix_mask(image_path, mask_path)
    if valid_mask_path is None:
        print("  ERROR: Mask is empty! Run Experiment 2 (fixed version) first.")
        sys.exit(1)

    # Step 3: Extract
    print("\n[Step 3] Extracting radiomics features...")
    result = extract_features(image_path, valid_mask_path)
    if result is None:
        print("  ERROR: Extraction failed!")
        sys.exit(1)

    # Step 4: Organize
    print("\n[Step 4] Organizing features...")
    features, cats, diagnostics = organize_features(result)

    total = 0
    print("\n  Feature Summary:")
    print("  " + "-" * 40)
    for cat, feats in cats.items():
        if feats:
            total += len(feats)
            print(f"    {cat.upper():12s}: {len(feats):3d} features")
    print("  " + "-" * 40)
    print(f"    {'TOTAL':12s}: {total:3d} features")

    # Step 5: Save TXT
    print("\n[Step 5] Saving to text file...")
    save_features_txt(features, cats, EXP3_RESULTS)

    # Step 6: Save CSV
    print("\n[Step 6] Saving to CSV...")
    save_features_csv(features, EXP3_RESULTS)

    # Step 7: Visualize
    print("\n[Step 7] Creating charts...")
    visualize_features(cats, EXP3_RESULTS)

    # Print key features
    print("\n" + "=" * 70)
    print("  KEY FEATURES")
    print("=" * 70)
    for k, v in sorted(cats.get('shape', {}).items()):
        if isinstance(v, float):
            print(f"    {k.split('_')[-1]:30s}: {v:.4f}")

    print("\n" + "=" * 70)
    print("  EXPERIMENT 3 COMPLETE!")
    print("=" * 70)
    print(f"  Results in: {EXP3_RESULTS}")
    for f in sorted(os.listdir(EXP3_RESULTS)):
        print(f"    - {f}")
    print("=" * 70)