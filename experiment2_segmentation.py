"""
=====================================================================
EXPERIMENT 2: Breast Tumor Segmentation Based on DCE-MRI
=====================================================================
FIXED VERSION - Produces a valid, non-empty tumor mask.

Your patient: TCGA-AO-A03M
Your DCE series: Bind(11251/7/...) with multiple time points
Image size: 256 x 256 x 37 slices
=====================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"C:\Users\Asus\PycharmProjects\medical"
DATASET_PATH = r"C:\Users\Asus\Downloads\tcga_brca"
EXP1_RESULTS = os.path.join(BASE_DIR, "experiment1_results")
EXP2_RESULTS = os.path.join(BASE_DIR, "experiment2_results")

os.makedirs(EXP2_RESULTS, exist_ok=True)


# ============================================================
# STEP 1: FIND AND LOAD DCE-MRI SERIES
# ============================================================
def find_dce_series(dataset_path):
    """Find all DICOM series and identify pre/post-contrast phases."""
    reader = sitk.ImageSeriesReader()
    all_series = []

    for dirpath, dirnames, filenames in os.walk(dataset_path):
        series_ids = reader.GetGDCMSeriesIDs(dirpath)
        for sid in series_ids:
            file_names = reader.GetGDCMSeriesFileNames(dirpath, sid)
            if not file_names:
                continue

            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(file_names[0])
            file_reader.ReadImageInformation()

            try:
                desc = file_reader.GetMetaData("0008|103e").strip()
            except:
                desc = "Unknown"
            try:
                series_num = file_reader.GetMetaData("0020|0011").strip()
            except:
                series_num = "0"

            all_series.append({
                'id': sid,
                'folder': dirpath,
                'description': desc,
                'series_number': series_num,
                'num_files': len(file_names),
            })

    return all_series


def load_series(folder, series_id):
    """Load a DICOM series as a 3D volume."""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder, series_id)
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    image = reader.Execute()
    image = sitk.Cast(image, sitk.sitkFloat32)
    return image


# ============================================================
# STEP 2: CREATE SUBTRACTION IMAGE
# ============================================================
def create_subtraction(pre_image, post_image):
    """Subtract pre-contrast from post-contrast to highlight enhancement."""
    if pre_image.GetSize() != post_image.GetSize():
        print("  Resampling post to match pre dimensions...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(pre_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        post_image = resampler.Execute(post_image)

    subtracted = post_image - pre_image
    subtracted = sitk.Threshold(subtracted, lower=0, upper=100000, outsideValue=0)
    return subtracted


# ============================================================
# STEP 3: ROBUST TUMOR SEGMENTATION
# ============================================================
def segment_tumor(image):
    """
    Segment tumor from subtraction image using multiple robust methods.
    Returns a binary mask with label=1 for tumor.
    """
    arr = sitk.GetArrayFromImage(image)
    print(f"  Image shape: {arr.shape}")
    print(f"  Value range: [{arr.min():.2f}, {arr.max():.2f}]")
    print(f"  Mean: {arr.mean():.2f}, Std: {arr.std():.2f}")
    print(f"  Non-zero voxels: {(arr > 0).sum()}")

    if arr.max() <= 0:
        print("  WARNING: No positive values. Using absolute values.")
        arr = np.abs(arr)

    positive = arr[arr > 0]
    if len(positive) == 0:
        print("  ERROR: Image is completely zero!")
        return None

    best_mask = None
    best_score = 0

    # ---- Method 1: Percentile thresholding ----
    print("\n  Method 1: Percentile thresholding...")
    for pct in [80, 85, 88, 90, 92, 95]:
        thresh = np.percentile(positive, pct)
        binary = (arr > thresh).astype(np.uint8)

        mask_sitk = sitk.GetImageFromArray(binary)
        mask_sitk.CopyInformation(image)
        mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)

        # Clean up with 2D-oriented kernels (z-spacing is huge: 17.8mm)
        mask_sitk = sitk.BinaryMorphologicalClosing(mask_sitk, [3, 3, 1])
        mask_sitk = sitk.BinaryMorphologicalOpening(mask_sitk, [2, 2, 1])

        # Connected components - keep largest
        cc = sitk.ConnectedComponent(mask_sitk)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)

        if stats.GetNumberOfLabels() == 0:
            continue

        largest = max(stats.GetLabels(), key=lambda l: stats.GetNumberOfPixels(l))
        size = stats.GetNumberOfPixels(largest)

        total = arr.size
        if 100 < size < total * 0.15:
            print(f"    P{pct} (thresh={thresh:.1f}): {size} voxels [GOOD]")
            if size > best_score:
                best_score = size
                best_mask = sitk.BinaryThreshold(cc, lowerThreshold=largest,
                                                  upperThreshold=largest)
        elif size > 0:
            print(f"    P{pct} (thresh={thresh:.1f}): {size} voxels")
            if best_mask is None and size > 20:
                best_mask = sitk.BinaryThreshold(cc, lowerThreshold=largest,
                                                  upperThreshold=largest)
                best_score = size

    # ---- Method 2: Otsu ----
    print("\n  Method 2: Otsu thresholding...")
    try:
        otsu = sitk.OtsuThresholdImageFilter()
        otsu.SetInsideValue(0)
        otsu.SetOutsideValue(1)
        mask_otsu = otsu.Execute(image)
        thresh_val = otsu.GetThreshold()
        print(f"    Otsu threshold: {thresh_val:.2f}")

        mask_otsu = sitk.BinaryMorphologicalClosing(mask_otsu, [3, 3, 1])
        mask_otsu = sitk.BinaryMorphologicalOpening(mask_otsu, [2, 2, 1])

        cc = sitk.ConnectedComponent(mask_otsu)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)

        if stats.GetNumberOfLabels() > 0:
            largest = max(stats.GetLabels(), key=lambda l: stats.GetNumberOfPixels(l))
            size = stats.GetNumberOfPixels(largest)
            print(f"    Otsu result: {size} voxels")

            if size > best_score and 50 < size < arr.size * 0.3:
                best_score = size
                best_mask = sitk.BinaryThreshold(cc, lowerThreshold=largest,
                                                  upperThreshold=largest)
    except Exception as e:
        print(f"    Otsu failed: {e}")

    # ---- Method 3: Region growing from brightest point ----
    print("\n  Method 3: Region growing...")
    try:
        smoothed = sitk.CurvatureAnisotropicDiffusion(
            image, timeStep=0.0625, conductanceParameter=3.0,
            conductanceScalingUpdateInterval=1, numberOfIterations=5
        )

        max_idx = np.unravel_index(np.argmax(arr), arr.shape)
        seed = [int(max_idx[2]), int(max_idx[1]), int(max_idx[0])]
        print(f"    Seed at brightest voxel: {seed}, value={arr[max_idx]:.2f}")

        mask_rg = sitk.ConfidenceConnected(
            smoothed, seedList=[seed],
            numberOfIterations=3, multiplier=2.5,
            initialNeighborhoodRadius=2, replaceValue=1
        )

        count = int(sitk.GetArrayFromImage(mask_rg).sum())
        print(f"    Region growing: {count} voxels")

        if count > best_score and 30 < count < arr.size * 0.3:
            best_score = count
            best_mask = mask_rg
    except Exception as e:
        print(f"    Region growing failed: {e}")

    # ---- Method 4: Mean + std threshold (fallback) ----
    if best_mask is None or best_score < 50:
        print("\n  Method 4: Mean+std threshold (fallback)...")
        for mult in [1.0, 1.5, 2.0]:
            thresh = float(positive.mean() + mult * positive.std())
            binary = (arr > thresh).astype(np.uint8)
            mask_fb = sitk.GetImageFromArray(binary)
            mask_fb.CopyInformation(image)
            mask_fb = sitk.Cast(mask_fb, sitk.sitkUInt8)

            mask_fb = sitk.BinaryMorphologicalClosing(mask_fb, [3, 3, 1])

            count = int(sitk.GetArrayFromImage(mask_fb).sum())
            print(f"    mean + {mult}*std = {thresh:.1f}: {count} voxels")

            if count > 30:
                cc = sitk.ConnectedComponent(mask_fb)
                stats = sitk.LabelShapeStatisticsImageFilter()
                stats.Execute(cc)
                if stats.GetNumberOfLabels() > 0:
                    largest = max(stats.GetLabels(),
                                  key=lambda l: stats.GetNumberOfPixels(l))
                    best_mask = sitk.BinaryThreshold(cc, lowerThreshold=largest,
                                                      upperThreshold=largest)
                    best_score = stats.GetNumberOfPixels(largest)
                    break

    # ---- Method 5: Top 5% brightest voxels (last resort) ----
    if best_mask is None or best_score < 10:
        print("\n  Method 5: Top 5% brightest voxels (last resort)...")
        thresh = np.percentile(positive, 95)
        binary = (arr > thresh).astype(np.uint8)
        best_mask = sitk.GetImageFromArray(binary)
        best_mask.CopyInformation(image)
        best_mask = sitk.Cast(best_mask, sitk.sitkUInt8)
        best_score = int(binary.sum())
        print(f"    Result: {best_score} voxels")

    print(f"\n  Final mask: {best_score} tumor voxels")
    return best_mask


# ============================================================
# STEP 4: VISUALIZATION
# ============================================================
def visualize_subtraction(pre_arr, post_arr, sub_arr, save_path):
    """Show pre, post, and subtracted images side by side."""
    mid = pre_arr.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(pre_arr[mid], cmap='gray')
    axes[0].set_title('Pre-contrast', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(post_arr[mid], cmap='gray')
    axes[1].set_title('Post-contrast', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(sub_arr[mid], cmap='hot')
    axes[2].set_title('Subtracted (Post - Pre)', fontsize=14)
    axes[2].axis('off')

    plt.suptitle('DCE-MRI Subtraction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


def visualize_segmentation(image, mask, save_path, title="Segmentation"):
    """Overlay mask contours on the image."""
    from scipy import ndimage

    img_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)

    tumor_slices = np.where(mask_arr.sum(axis=(1, 2)) > 0)[0]
    if len(tumor_slices) == 0:
        print("  WARNING: No tumor slices to show.")
        return

    n_show = min(8, len(tumor_slices))
    indices = np.linspace(0, len(tumor_slices) - 1, n_show, dtype=int)
    show_slices = tumor_slices[indices]

    cols = 4
    rows = max(1, int(np.ceil(n_show / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if n_show == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(title, fontsize=16, fontweight='bold')

    for idx, ax in enumerate(axes.flat):
        if idx < n_show:
            z = show_slices[idx]
            sl_img = img_arr[z]
            sl_mask = mask_arr[z]

            if sl_img.max() > sl_img.min():
                disp = (sl_img - sl_img.min()) / (sl_img.max() - sl_img.min())
            else:
                disp = np.zeros_like(sl_img, dtype=float)

            rgb = np.stack([disp, disp, disp], axis=-1)

            if sl_mask.any():
                contour = ndimage.binary_dilation(sl_mask, iterations=1) ^ sl_mask.astype(bool)
                rgb[contour, :] = [1, 0, 0]
                rgb[sl_mask > 0, 0] = np.minimum(rgb[sl_mask > 0, 0] + 0.3, 1.0)

            ax.imshow(rgb)
            ax.set_title(f'Slice {z} ({int(sl_mask.sum())} px)', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


def visualize_3views(image, mask, save_path):
    """Show axial, sagittal, coronal views with mask overlay."""
    from scipy import ndimage

    img_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)

    tumor_coords = np.argwhere(mask_arr > 0)
    if len(tumor_coords) > 0:
        center = tumor_coords.mean(axis=0).astype(int)
        cz, cy, cx = center
    else:
        cz, cy, cx = [s // 2 for s in img_arr.shape]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    views = [
        (img_arr[cz], mask_arr[cz], f'Axial (slice {cz})', None),
        (img_arr[:, cy, :], mask_arr[:, cy, :], f'Sagittal (slice {cy})', 'auto'),
        (img_arr[:, :, cx], mask_arr[:, :, cx], f'Coronal (slice {cx})', 'auto'),
    ]

    for ax, (view_img, view_mask, title_txt, aspect) in zip(axes, views):
        if view_img.max() > view_img.min():
            disp = (view_img - view_img.min()) / (view_img.max() - view_img.min())
        else:
            disp = np.zeros_like(view_img, dtype=float)

        rgb = np.stack([disp, disp, disp], axis=-1)

        if view_mask.any():
            contour = ndimage.binary_dilation(view_mask, iterations=1) ^ view_mask.astype(bool)
            rgb[contour, :] = [1, 0, 0]
            rgb[view_mask > 0, 0] = np.minimum(rgb[view_mask > 0, 0] + 0.3, 1.0)

        ax.imshow(rgb, aspect=aspect)
        ax.set_title(title_txt, fontsize=13)
        ax.axis('off')

    plt.suptitle('Tumor Segmentation - 3 Views', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  EXPERIMENT 2: Breast Tumor Segmentation (DCE-MRI)")
    print("=" * 70)

    # ---- Check if subtraction image already exists ----
    existing_sub = os.path.join(EXP2_RESULTS, "subtraction_image.nrrd")
    existing_sub_gz = os.path.join(EXP2_RESULTS, "subtraction_image.nii.gz")

    sub_image = None

    if os.path.exists(existing_sub):
        print("\n[Step 1] Found existing subtraction image, loading...")
        sub_image = sitk.ReadImage(existing_sub)
        sub_image = sitk.Cast(sub_image, sitk.sitkFloat32)
        print(f"  Size: {sub_image.GetSize()}, Spacing: {sub_image.GetSpacing()}")

        sub_arr = sitk.GetArrayFromImage(sub_image)
        if sub_arr.max() <= 0:
            print("  No positive values, will recreate subtraction.")
            sub_image = None

    elif os.path.exists(existing_sub_gz):
        print("\n[Step 1] Found existing subtraction image (.nii.gz), loading...")
        sub_image = sitk.ReadImage(existing_sub_gz)
        sub_image = sitk.Cast(sub_image, sitk.sitkFloat32)
        print(f"  Size: {sub_image.GetSize()}, Spacing: {sub_image.GetSpacing()}")

        sub_arr = sitk.GetArrayFromImage(sub_image)
        if sub_arr.max() <= 0:
            sub_image = None

    if sub_image is None:
        # ---- Build subtraction from scratch ----
        print("\n[Step 1] Scanning for DCE-MRI series...")
        all_series = find_dce_series(DATASET_PATH)

        if not all_series:
            print("  ERROR: No DICOM series found!")
            sys.exit(1)

        print(f"  Found {len(all_series)} series:")
        for i, s in enumerate(all_series):
            print(f"    [{i:2d}] #{s['series_number']:>3s}: {s['description']}"
                  f" ({s['num_files']} files)")

        sorted_series = sorted(all_series,
                                key=lambda x: int(x['series_number'])
                                if x['series_number'].isdigit() else 999)

        # Find DCE phases: Bind series with "/7/" pattern
        bind7 = [s for s in sorted_series
                 if 'Bind' in s['description'] and '/7/' in s['description']]

        if len(bind7) >= 2:
            pre_series = bind7[0]
            post_series = bind7[1]
        else:
            large = [s for s in sorted_series if s['num_files'] > 20]
            if len(large) >= 2:
                pre_series = large[0]
                post_series = large[1]
            else:
                print("  ERROR: Cannot identify DCE phases.")
                sys.exit(1)

        print(f"\n  Pre-contrast:  {pre_series['description']}")
        print(f"  Post-contrast: {post_series['description']}")

        print("\n[Step 2] Loading DICOM series...")
        print("  Loading pre-contrast...")
        pre_image = load_series(pre_series['folder'], pre_series['id'])
        print(f"    Size: {pre_image.GetSize()}")

        print("  Loading post-contrast...")
        post_image = load_series(post_series['folder'], post_series['id'])
        print(f"    Size: {post_image.GetSize()}")

        print("\n[Step 3] Creating subtraction image...")
        sub_image = create_subtraction(pre_image, post_image)

        sitk.WriteImage(sub_image, os.path.join(EXP2_RESULTS, "subtraction_image.nrrd"))
        sitk.WriteImage(sub_image, os.path.join(EXP2_RESULTS, "subtraction_image.nii.gz"))
        print("  Saved subtraction images.")

        pre_arr = sitk.GetArrayFromImage(pre_image)
        post_arr = sitk.GetArrayFromImage(post_image)
        sub_arr = sitk.GetArrayFromImage(sub_image)
        visualize_subtraction(pre_arr, post_arr, sub_arr,
                              os.path.join(EXP2_RESULTS, "dce_subtraction_segmentation_results.png"))
    else:
        sub_arr = sitk.GetArrayFromImage(sub_image)

    # ---- Segment the tumor ----
    print("\n[Step 4] Segmenting tumor from subtraction image...")
    mask = segment_tumor(sub_image)

    if mask is None:
        print("\n  CRITICAL: Segmentation failed!")
        sys.exit(1)

    mask_arr = sitk.GetArrayFromImage(mask)
    n_voxels = int(mask_arr.sum())

    if n_voxels == 0:
        print("\n  CRITICAL: Mask is empty after all methods!")
        sys.exit(1)

    # ---- Save mask ----
    print(f"\n[Step 5] Saving tumor mask ({n_voxels} voxels)...")

    sitk.WriteImage(mask, os.path.join(EXP2_RESULTS, "tumor_mask.nrrd"))
    sitk.WriteImage(mask, os.path.join(EXP2_RESULTS, "tumor_mask.nii.gz"))
    sitk.WriteImage(mask, os.path.join(EXP2_RESULTS, "dce_subtraction_tumor_mask.nrrd"))
    sitk.WriteImage(mask, os.path.join(EXP2_RESULTS, "dce_subtraction_tumor_mask.nii.gz"))

    sitk.WriteImage(sitk.Cast(sub_image, sitk.sitkInt16),
                    os.path.join(EXP2_RESULTS, "original_image.nrrd"))
    sitk.WriteImage(sitk.Cast(sub_image, sitk.sitkInt16),
                    os.path.join(EXP2_RESULTS, "original_image.nii.gz"))

    print("  All mask files saved.")

    # ---- Statistics ----
    spacing = sub_image.GetSpacing()
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    tumor_mm3 = n_voxels * voxel_vol
    tumor_cm3 = tumor_mm3 / 1000.0
    n_tumor_slices = int((mask_arr.sum(axis=(1, 2)) > 0).sum())
    tumor_slice_range = np.where(mask_arr.sum(axis=(1, 2)) > 0)[0]

    stats_text = (
        f"Tumor Segmentation Statistics\n"
        f"{'=' * 50}\n"
        f"Patient: TCGA-AO-A03M\n"
        f"Method: Intensity Thresholding + Morphological Cleanup\n"
        f"Tumor voxels: {n_voxels}\n"
        f"Tumor volume: {tumor_mm3:.2f} mm3\n"
        f"Tumor volume: {tumor_cm3:.2f} cm3\n"
        f"Voxel spacing: {spacing}\n"
        f"Slices with tumor: {n_tumor_slices}\n"
        f"Slice range: {tumor_slice_range[0]} to {tumor_slice_range[-1]}\n"
        f"Image size: {sub_image.GetSize()}\n"
    )

    with open(os.path.join(EXP2_RESULTS, "dce_subtraction_tumor_statistics.txt"), 'w') as f:
        f.write(stats_text)

    print(f"\n  Tumor: {n_voxels} voxels, {tumor_cm3:.2f} cm3, {n_tumor_slices} slices")

    # ---- Visualize ----
    print("\n[Step 6] Visualizing segmentation...")
    visualize_segmentation(sub_image, mask,
                           os.path.join(EXP2_RESULTS, "dce_subtraction_segmentation_results.png"),
                           title="Breast Tumor Segmentation")
    visualize_3views(sub_image, mask,
                     os.path.join(EXP2_RESULTS, "dce_subtraction_3view_segmentation.png"))

    print("\n" + "=" * 70)
    print("  EXPERIMENT 2 COMPLETE!")
    print("=" * 70)
    print(f"  Results in: {EXP2_RESULTS}")
    for f in sorted(os.listdir(EXP2_RESULTS)):
        print(f"    - {f}")
    print("=" * 70)