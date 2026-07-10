"""
============================================================
EXPERIMENT 1: Visualization and Preprocessing of Medical Images
============================================================
Tasks:
  (1) Load DICOM files, select a patient, output info, save screenshot
  (2) Convert 2D DICOM slices into 3D volume (nrrd and nii.gz)
  (3) Screenshots are saved for your report
============================================================
BEFORE RUNNING: Update DATA_ROOT below to your actual TCGA-BRCA folder path
============================================================
"""

import os
import sys
import glob
import pydicom
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import SimpleITK as sitk

# ============================================================
# >>> CHANGE THIS PATH TO YOUR ACTUAL TCGA-BRCA DATA FOLDER <<<
# ============================================================
DATA_ROOT = r"C:\Users\Asus\Downloads\tcga_brca"

# Output directory for all results
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment1_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_all_patients(data_root):
    """Scan the data root and find all patient folders."""
    patients = []
    if not os.path.exists(data_root):
        print(f"ERROR: Data root not found: {data_root}")
        print("Please update DATA_ROOT in this script.")
        sys.exit(1)

    for item in os.listdir(data_root):
        full_path = os.path.join(data_root, item)
        if os.path.isdir(full_path):
            patients.append(full_path)

    if not patients:
        # Try one level deeper
        for item in os.listdir(data_root):
            sub = os.path.join(data_root, item)
            if os.path.isdir(sub):
                for item2 in os.listdir(sub):
                    full_path = os.path.join(sub, item2)
                    if os.path.isdir(full_path):
                        patients.append(full_path)

    return sorted(patients)


def find_dicom_files(folder):
    """Recursively find all DICOM files in a folder."""
    dicom_files = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            fpath = os.path.join(root, f)
            # Try to read as DICOM
            if f.endswith('.dcm') or not '.' in f:
                try:
                    pydicom.dcmread(fpath, stop_before_pixels=True)
                    dicom_files.append(fpath)
                except:
                    continue
    return sorted(dicom_files)


def find_dicom_series(patient_folder):
    """Find all DICOM series within a patient folder."""
    series_dict = {}
    for root, dirs, files in os.walk(patient_folder):
        for f in files:
            fpath = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                series_uid = str(ds.SeriesInstanceUID)
                if series_uid not in series_dict:
                    series_dict[series_uid] = {
                        'files': [],
                        'description': getattr(ds, 'SeriesDescription', 'Unknown'),
                        'folder': root
                    }
                series_dict[series_uid]['files'].append(fpath)
            except:
                continue
    return series_dict


# ============================================================
# TASK 1: Load DICOM files, display patient info, save screenshot
# ============================================================
def task1_load_and_display_dicom():
    print("=" * 60)
    print("TASK 1: Loading DICOM files and displaying patient info")
    print("=" * 60)

    # Find all patients
    patients = find_all_patients(DATA_ROOT)
    print(f"\nFound {len(patients)} patient folders")

    if len(patients) == 0:
        print("No patient folders found. Checking for DICOM files directly...")
        dicom_files = find_dicom_files(DATA_ROOT)
        if dicom_files:
            print(f"Found {len(dicom_files)} DICOM files directly in root")
            patient_folder = DATA_ROOT
        else:
            print("No DICOM files found. Please check DATA_ROOT path.")
            return None, None
    else:
        # Print all patient folder names
        print("\nPatient folders found:")
        for i, p in enumerate(patients[:20]):  # Show first 20
            print(f"  [{i}] {os.path.basename(p)}")
        if len(patients) > 20:
            print(f"  ... and {len(patients) - 20} more")

        # Select first patient
        patient_folder = patients[0]
        print(f"\n>>> Selected patient: {os.path.basename(patient_folder)}")

    # Find all DICOM series for this patient
    series_dict = find_dicom_series(patient_folder)
    print(f"\nFound {len(series_dict)} DICOM series for this patient:")

    series_list = list(series_dict.items())
    for i, (uid, info) in enumerate(series_list):
        print(f"  Series {i}: {info['description']} ({len(info['files'])} slices)")
        print(f"    Folder: {info['folder']}")

    if not series_dict:
        print("No DICOM series found for this patient.")
        return None, None

    # Select the first series with the most slices (likely the main volume)
    best_series_uid = max(series_dict, key=lambda k: len(series_dict[k]['files']))
    best_series = series_dict[best_series_uid]
    print(f"\n>>> Selected series: {best_series['description']} ({len(best_series['files'])} slices)")

    # Load one DICOM file and display detailed information
    sample_file = best_series['files'][0]
    ds = pydicom.dcmread(sample_file)

    info_text = []
    info_text.append("=" * 60)
    info_text.append("DICOM FILE INFORMATION")
    info_text.append("=" * 60)
    info_text.append(f"File: {os.path.basename(sample_file)}")
    info_text.append(f"Patient Name: {getattr(ds, 'PatientName', 'N/A')}")
    info_text.append(f"Patient ID: {getattr(ds, 'PatientID', 'N/A')}")
    info_text.append(f"Patient Sex: {getattr(ds, 'PatientSex', 'N/A')}")
    info_text.append(f"Patient Age: {getattr(ds, 'PatientAge', 'N/A')}")
    info_text.append(f"Study Date: {getattr(ds, 'StudyDate', 'N/A')}")
    info_text.append(f"Study Description: {getattr(ds, 'StudyDescription', 'N/A')}")
    info_text.append(f"Series Description: {getattr(ds, 'SeriesDescription', 'N/A')}")
    info_text.append(f"Modality: {getattr(ds, 'Modality', 'N/A')}")
    info_text.append(f"Manufacturer: {getattr(ds, 'Manufacturer', 'N/A')}")
    info_text.append(f"Rows: {getattr(ds, 'Rows', 'N/A')}")
    info_text.append(f"Columns: {getattr(ds, 'Columns', 'N/A')}")
    info_text.append(f"Pixel Spacing: {getattr(ds, 'PixelSpacing', 'N/A')}")
    info_text.append(f"Slice Thickness: {getattr(ds, 'SliceThickness', 'N/A')}")
    info_text.append(f"Bits Allocated: {getattr(ds, 'BitsAllocated', 'N/A')}")
    info_text.append(f"Bits Stored: {getattr(ds, 'BitsStored', 'N/A')}")
    info_text.append(f"Window Center: {getattr(ds, 'WindowCenter', 'N/A')}")
    info_text.append(f"Window Width: {getattr(ds, 'WindowWidth', 'N/A')}")
    info_text.append(f"Number of slices in series: {len(best_series['files'])}")
    info_text.append("=" * 60)

    # Print and save info
    for line in info_text:
        print(line)

    info_path = os.path.join(OUTPUT_DIR, "task1_dicom_info.txt")
    with open(info_path, 'w') as f:
        f.write('\n'.join(info_text))
    print(f"\n[SAVED] DICOM info -> {info_path}")

    # Display the DICOM image and save screenshot
    if hasattr(ds, 'pixel_array'):
        pixel_data = ds.pixel_array.astype(float)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(pixel_data, cmap='gray')
        axes[0].set_title(f"Original DICOM Image\n{getattr(ds, 'SeriesDescription', 'N/A')}", fontsize=12)
        axes[0].axis('off')

        # With patient info overlay
        axes[1].imshow(pixel_data, cmap='gray')
        axes[1].set_title("DICOM with Patient Info", fontsize=12)
        text_str = (f"ID: {getattr(ds, 'PatientID', 'N/A')}\n"
                    f"Modality: {getattr(ds, 'Modality', 'N/A')}\n"
                    f"Size: {getattr(ds, 'Rows', '?')}x{getattr(ds, 'Columns', '?')}\n"
                    f"Date: {getattr(ds, 'StudyDate', 'N/A')}")
        axes[1].text(5, 5, text_str, fontsize=9, color='yellow',
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        axes[1].axis('off')

        # Histogram of pixel values
        axes[2].hist(pixel_data.flatten(), bins=100, color='steelblue', alpha=0.7)
        axes[2].set_title("Pixel Value Distribution", fontsize=12)
        axes[2].set_xlabel("Pixel Value")
        axes[2].set_ylabel("Frequency")

        plt.suptitle(f"Experiment 1 - Task 1: DICOM Visualization\nPatient: {getattr(ds, 'PatientID', 'N/A')}",
                      fontsize=14, fontweight='bold')
        plt.tight_layout()

        screenshot_path = os.path.join(OUTPUT_DIR, "task1_dicom_screenshot.png")
        plt.savefig(screenshot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] Screenshot -> {screenshot_path}")

    # Also show multiple slices from the series
    print("\nLoading multiple slices for overview...")
    slices = []
    slice_files = sorted(best_series['files'])

    # Load all slices and sort by position
    for sf in slice_files:
        try:
            s = pydicom.dcmread(sf)
            slices.append(s)
        except:
            continue

    if len(slices) > 1:
        # Sort by ImagePositionPatient or InstanceNumber
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except:
            try:
                slices.sort(key=lambda x: int(x.InstanceNumber))
            except:
                pass

        # Show grid of slices
        n_show = min(16, len(slices))
        indices = np.linspace(0, len(slices) - 1, n_show, dtype=int)

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        for idx, ax in enumerate(axes.flat):
            if idx < n_show:
                pixel = slices[indices[idx]].pixel_array.astype(float)
                ax.imshow(pixel, cmap='gray')
                ax.set_title(f"Slice {indices[idx]+1}/{len(slices)}", fontsize=9)
            ax.axis('off')

        plt.suptitle(f"Experiment 1 - Multiple Slices Overview\nPatient: {getattr(ds, 'PatientID', 'N/A')} | "
                     f"Series: {best_series['description']} | Total: {len(slices)} slices",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        multi_path = os.path.join(OUTPUT_DIR, "task1_multiple_slices.png")
        plt.savefig(multi_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] Multiple slices overview -> {multi_path}")

    return patient_folder, series_dict


# ============================================================
# TASK 2: Convert 2D DICOM slices to 3D volume (nrrd & nii.gz)
# ============================================================
def task2_convert_to_3d(patient_folder, series_dict):
    print("\n" + "=" * 60)
    print("TASK 2: Converting 2D DICOM to 3D volume files")
    print("=" * 60)

    if not series_dict:
        print("No series data available. Skipping.")
        return None

    converted_files = {}

    for series_uid, info in series_dict.items():
        series_desc = info['description'].replace(' ', '_').replace('/', '_')
        series_folder = info['folder']

        print(f"\nConverting series: {info['description']} ({len(info['files'])} slices)")
        print(f"  Source folder: {series_folder}")

        try:
            # Use SimpleITK to read the DICOM series
            reader = sitk.ImageSeriesReader()

            # Get DICOM file names from the folder
            dicom_names = reader.GetGDCMSeriesFileNames(series_folder, series_uid)

            if len(dicom_names) == 0:
                # Try without series UID filter
                dicom_names = reader.GetGDCMSeriesFileNames(series_folder)

            if len(dicom_names) == 0:
                print(f"  WARNING: Could not find DICOM series files. Skipping.")
                continue

            print(f"  Found {len(dicom_names)} DICOM files")
            reader.SetFileNames(dicom_names)

            # Read the series as a 3D image
            image_3d = reader.Execute()

            # Print 3D volume info
            size = image_3d.GetSize()
            spacing = image_3d.GetSpacing()
            origin = image_3d.GetOrigin()
            direction = image_3d.GetDirection()

            print(f"  3D Volume Info:")
            print(f"    Size: {size}")
            print(f"    Spacing: {spacing}")
            print(f"    Origin: {origin}")
            print(f"    Pixel Type: {image_3d.GetPixelIDTypeAsString()}")

            # Save as NRRD
            nrrd_filename = f"patient_volume_{series_desc}.nrrd"
            nrrd_path = os.path.join(OUTPUT_DIR, nrrd_filename)
            sitk.WriteImage(image_3d, nrrd_path)
            print(f"  [SAVED] NRRD -> {nrrd_path}")

            # Save as NIfTI (.nii.gz)
            nifti_filename = f"patient_volume_{series_desc}.nii.gz"
            nifti_path = os.path.join(OUTPUT_DIR, nifti_filename)
            sitk.WriteImage(image_3d, nifti_path)
            print(f"  [SAVED] NIfTI -> {nifti_path}")

            converted_files[series_uid] = {
                'nrrd': nrrd_path,
                'nifti': nifti_path,
                'image': image_3d,
                'description': info['description']
            }

        except Exception as e:
            print(f"  ERROR converting series: {e}")
            continue

    # Save 3D volume visualization
    if converted_files:
        # Take the first converted volume for visualization
        first_key = list(converted_files.keys())[0]
        image_3d = converted_files[first_key]['image']
        desc = converted_files[first_key]['description']

        # Convert to numpy array
        arr = sitk.GetArrayFromImage(image_3d)  # shape: (Z, Y, X)
        print(f"\n  3D numpy array shape: {arr.shape}")

        # Show axial, sagittal, coronal views
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        mid_z = arr.shape[0] // 2
        mid_y = arr.shape[1] // 2
        mid_x = arr.shape[2] // 2

        # Axial view (XY plane)
        axes[0].imshow(arr[mid_z, :, :], cmap='gray')
        axes[0].set_title(f"Axial View (Slice {mid_z})", fontsize=12)
        axes[0].axis('off')

        # Coronal view (XZ plane)
        axes[1].imshow(arr[:, mid_y, :], cmap='gray', aspect='auto')
        axes[1].set_title(f"Coronal View (Slice {mid_y})", fontsize=12)
        axes[1].axis('off')

        # Sagittal view (YZ plane)
        axes[2].imshow(arr[:, :, mid_x], cmap='gray', aspect='auto')
        axes[2].set_title(f"Sagittal View (Slice {mid_x})", fontsize=12)
        axes[2].axis('off')

        plt.suptitle(f"Experiment 1 - Task 2: 3D Volume Views\n{desc} | Shape: {arr.shape}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        views_path = os.path.join(OUTPUT_DIR, "task2_3d_volume_views.png")
        plt.savefig(views_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] 3D volume views -> {views_path}")

    # Save conversion summary
    summary_path = os.path.join(OUTPUT_DIR, "task2_conversion_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("3D Volume Conversion Summary\n")
        f.write("=" * 50 + "\n\n")
        for uid, data in converted_files.items():
            f.write(f"Series: {data['description']}\n")
            f.write(f"  NRRD file: {data['nrrd']}\n")
            f.write(f"  NIfTI file: {data['nifti']}\n\n")
    print(f"[SAVED] Summary -> {summary_path}")

    return converted_files


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# EXPERIMENT 1: Visualization & Preprocessing")
    print("#" * 60)
    print(f"\nData root: {DATA_ROOT}")
    print(f"Output dir: {OUTPUT_DIR}\n")

    # Task 1
    patient_folder, series_dict = task1_load_and_display_dicom()

    # Task 2
    if patient_folder and series_dict:
        converted = task2_convert_to_3d(patient_folder, series_dict)
    else:
        print("\nCould not proceed to Task 2. Please check your data path.")

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 COMPLETE!")
    print(f"All results saved in: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nFiles generated:")
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")
