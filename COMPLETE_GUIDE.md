# Complete Experiment Guide: Medical Image Analysis (TCGA-BRCA)

---

## SETUP: Install Python Packages First

Open PyCharm terminal (bottom panel) and run:

```
pip install pydicom SimpleITK numpy matplotlib nibabel pyradiomics scikit-learn pandas openpyxl xlrd scipy Pillow
```

If any package fails, install them one by one:
```
pip install pydicom
pip install SimpleITK
pip install pyradiomics
pip install scikit-learn
pip install nibabel
```

Then copy all 4 Python scripts into your PyCharm project folder:
`C:\Users\Asus\PycharmProjects\medical\`

---

## DAY 1, EXPERIMENT 1: Visualization and Preprocessing

### Step 1: Update the Data Path

Open `experiment1_visualization.py` in PyCharm. Find this line near the top:

```python
DATA_ROOT = r"C:\Users\Asus\Downloads\tcga_brca"
```

Verify this matches your actual TCGA-BRCA data folder. To check, open File Explorer and navigate to `C:\Users\Asus\Downloads\tcga_brca`. You should see patient folders inside (named like `TCGA-A7-XXXX`).

### Step 2: Run the Script

In PyCharm terminal:
```
python experiment1_visualization.py
```

### Step 3: Check Results

After running, a folder called `experiment1_results` will appear in your project with these files. The ones you need for your report are:

| File | What It Is | Use In Report |
|------|-----------|---------------|
| `task1_dicom_info.txt` | Patient DICOM metadata | Copy into report as evidence of loading DICOM |
| `task1_dicom_screenshot.png` | Single slice with info overlay + histogram | Screenshot for Task 1 |
| `task1_multiple_slices.png` | Grid of 16 slices from the series | Shows 2D slices overview |
| `patient_volume_*.nrrd` | 3D volume in NRRD format | Load into 3D Slicer (see below) |
| `patient_volume_*.nii.gz` | 3D volume in NIfTI format | Load into 3D Slicer / ITK-SNAP |
| `task2_3d_volume_views.png` | Axial/Coronal/Sagittal views | Screenshot for Task 2 |
| `task2_conversion_summary.txt` | List of converted files | Reference for report |

---

## HOW TO USE 3D SLICER (Step-by-Step)

### Installing 3D Slicer
You already have `Slicer-5.10.0-win-amd64.exe` in Downloads. Double-click to install if you haven't already, then open 3D Slicer.

### Loading DICOM Files Directly

1. Open 3D Slicer
2. Click the **DICOM** button (database icon) in the toolbar, OR go to **File > Add DICOM Data**
3. Click **Import DICOM files** in the DICOM Browser
4. Navigate to your TCGA-BRCA data folder: `C:\Users\Asus\Downloads\tcga_brca`
5. Select the patient folder (e.g., `TCGA-A7-XXXX`) and click **Import**
6. Wait for import to complete (may take a few minutes)
7. After import, you'll see the patient listed in the DICOM Browser
8. Select the patient → select the study → select the series
9. Click **Load** to load the images into the viewer

### Loading NRRD / NIfTI Files (from your Experiment 1 results)

1. Open 3D Slicer
2. Go to **File > Add Data** (or drag the file into the Slicer window)
3. Navigate to your `experiment1_results` folder
4. Select the `.nrrd` or `.nii.gz` file
5. Click **OK** to load
6. The volume will appear in all three view panels (Axial, Sagittal, Coronal)

### Navigating in 3D Slicer

Once your image is loaded, you'll see three 2D panels and one 3D panel:

- **Red panel** = Axial view (top-down)
- **Yellow panel** = Sagittal view (side view)
- **Green panel** = Coronal view (front view)
- **3D panel** = 3D rendering

Controls:
- **Scroll mouse wheel** in any panel to move through slices
- **Left-click + drag** to adjust window/level (brightness/contrast)
- **Right-click + drag** to zoom
- **Middle-click + drag** to pan
- **Ctrl + mouse wheel** to zoom in/out

### Taking Screenshots for Your Report

**Method 1: Built-in Screenshot Tool**
1. Go to **File > Save Screenshot** or press the camera icon
2. Choose the view you want to capture
3. Select save location and filename
4. Click **Save**

**Method 2: Windows Screenshot**
1. Arrange 3D Slicer to show the view you want
2. Press **Windows + Shift + S** to open Snipping Tool
3. Select the area to capture
4. Paste into your report or save as image

### Screenshots You Need to Save from 3D Slicer

Save these screenshots for your report:

1. **DICOM loaded view** - All 3 panels showing the MRI volume (axial, sagittal, coronal)
2. **3D volume rendering** - Click the eye icon next to "Volume Rendering" module to enable 3D view
3. **After loading NRRD file** - Shows successful 3D conversion
4. **After loading NIfTI file** - Shows .nii.gz works correctly

### Creating Volume Rendering (3D View)

1. Go to **Modules > Volume Rendering** (dropdown menu at top)
2. Click the eye icon to enable rendering
3. Adjust the **Preset** dropdown (try "CT-Chest-Contrast-Enhanced" or "MR-Default")
4. Use the **Shift** slider to adjust what's visible
5. Rotate the 3D view by clicking and dragging in the 3D panel
6. Take a screenshot of the 3D rendering

---

## DAY 1, EXPERIMENT 2: Breast Tumor Segmentation

### Step 1: Run the Segmentation Script

In PyCharm terminal:
```
python experiment2_segmentation.py
```

The script will:
- Find your DICOM series (pre-contrast and post-contrast)
- Create a subtraction image (highlights the tumor)
- Apply Level Set segmentation
- Save the tumor mask

### Step 2: Check Results

The `experiment2_results` folder will contain:

| File | What It Is | Use In Report |
|------|-----------|---------------|
| `subtraction_image.nrrd` | Subtraction image (post - pre contrast) | Load into 3D Slicer |
| `subtraction_image.nii.gz` | Same in NIfTI format | Alternative format |
| `dce_subtraction_tumor_mask.nrrd` | **TUMOR MASK** (the main result) | Load into 3D Slicer + use in Exp 3 |
| `dce_subtraction_tumor_mask.nii.gz` | Same in NIfTI format | Alternative format |
| `dce_subtraction_segmentation_results.png` | Slices showing tumor overlay | Key screenshot for report |
| `dce_subtraction_3view_segmentation.png` | 3-view segmentation comparison | Key screenshot for report |
| `dce_subtraction_tumor_statistics.txt` | Tumor volume and statistics | Include numbers in report |

### Step 3: Visualize Segmentation in 3D Slicer

1. Open 3D Slicer
2. **File > Add Data** → load the original volume `.nrrd` from experiment1_results
3. **File > Add Data** → load the `tumor_mask.nrrd` from experiment2_results
4. When loading the mask, check **"Show Options"** and set **"Label Map"** = Yes
5. Now you'll see the original image with the tumor mask overlaid in color
6. Scroll through slices to see the tumor segmentation
7. Take screenshots showing the segmentation overlay

### Adjusting Mask Overlay in 3D Slicer

1. In the slice viewer toolbar, look for the layer controls
2. Click the **Overlay** dropdown and select your tumor mask
3. Adjust the **Opacity** slider to see the overlay clearly
4. The tumor region will be highlighted in a different color

---

## DAY 2, EXPERIMENT 3: Radiomics Feature Extraction

### Step 1: Verify Prerequisites

Make sure you have these files from previous experiments:
- A 3D volume file (`.nrrd` or `.nii.gz`) from `experiment1_results`
- A tumor mask file (`.nrrd` or `.nii.gz`) from `experiment2_results`

### Step 2: Run the Script

```
python experiment3_radiomics.py
```

The script will automatically find the image and mask files from previous experiments. If it can't find them, it will ask you to type the full path.

### Step 3: Check Results

The `experiment3_results` folder will contain:

| File | What It Is | Use In Report |
|------|-----------|---------------|
| `radiomics_features.txt` | **All extracted features** (required deliverable) | Main result for Experiment 3 |
| `radiomics_features.csv` | Features in CSV format | Used by Experiment 4 |
| `radiomics_features.json` | Features in JSON format | Backup |
| `feature_summary.txt` | Summary of extraction | Reference |

### What Features Are Extracted

PyRadiomics extracts these categories of features:

- **First Order Statistics** (19 features): mean, median, energy, entropy, skewness, kurtosis, etc.
- **Shape Features** (16 features): volume, surface area, sphericity, elongation, flatness, etc.
- **GLCM** (24 features): Gray Level Co-occurrence Matrix - texture contrast, correlation, energy, etc.
- **GLRLM** (16 features): Gray Level Run Length Matrix - run emphasis, run variance, etc.
- **GLSZM** (16 features): Gray Level Size Zone Matrix - zone emphasis, zone variance, etc.
- **GLDM** (14 features): Gray Level Dependence Matrix - dependence emphasis, etc.
- **NGTDM** (5 features): Neighbouring Gray Tone Difference Matrix - coarseness, busyness, etc.

Total: approximately 100+ features from the original image, plus additional features from LoG and Wavelet filtered images.

---

## DAY 2, EXPERIMENT 4: Prediction Models

### Step 1: Update Clinical Data Path

Open `experiment4_prediction.py` and verify these paths point to your clinical data files:

```python
CLINICAL_DATA_PATH = r"C:\Users\Asus\Downloads\dataset.xlsx"
CLINICAL_DATA_ALT1 = r"C:\Users\Asus\Downloads\Perou-TCGA-BRCA-MRIsPAM50GHI21NKI.xlsx"
CLINICAL_DATA_ALT2 = r"C:\Users\Asus\Downloads\brca-clinicalforwiki.xls"
```

Check the exact filename of the Perou file in your Downloads folder and update if needed.

### Step 2: Run the Script

```
python experiment4_prediction.py
```

### Step 3: Check Results

The `experiment4_results` folder will contain:

| File | What It Is | Use In Report |
|------|-----------|---------------|
| `Mammaprint_type_model_comparison.png` | Bar chart comparing SVM vs LR | Key figure for report |
| `Mammaprint_type_confusion_matrices.png` | Confusion matrices for both models | Key figure for report |
| `Mammaprint_type_cv_scores.png` | 5-fold CV score trends | Shows CV performance |
| `Mammaprint_type_roc_curves.png` | ROC curves (if binary) | Shows discrimination ability |
| `GHI_RS_Score_model_comparison.png` | Same set of plots for GHI | Key figures for report |
| `GHI_RS_Score_confusion_matrices.png` | Confusion matrices for GHI | Key figure for report |
| `GHI_RS_Score_cv_scores.png` | 5-fold CV scores for GHI | Shows CV performance |
| `GHI_RS_Score_roc_curves.png` | ROC curves for GHI | Shows discrimination ability |
| `model_evaluation_report.txt` | **Complete evaluation report** | Copy key numbers into report |

### Understanding the Results

For each target (Mammaprint_type and GHI_RS_Score), the script builds:

1. **SVM model** with RBF kernel, evaluated with 5-fold stratified cross-validation
2. **Logistic Regression model**, same evaluation

Metrics reported:
- **Accuracy**: Overall correct predictions (higher is better)
- **Precision**: Of predicted positives, how many are actually positive
- **Recall**: Of actual positives, how many are correctly predicted
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area Under ROC Curve (discrimination ability, 1.0 = perfect, 0.5 = random)

---

## USING ITK-SNAP (Alternative to 3D Slicer)

You also have `itksnap-3.8.0` installed. Here's how to use it:

### Loading Images
1. Open ITK-SNAP
2. **File > Open Main Image**
3. Navigate to your `.nrrd` or `.nii.gz` file
4. Click **Next** → **Finish**

### Loading Segmentation Overlay
1. After loading the main image
2. **Segmentation > Load from File**
3. Select your tumor mask file
4. The segmentation will appear as a colored overlay

### Taking Screenshots
1. **File > Save Screenshot of Workspace**
2. Or use Windows Snipping Tool (Win + Shift + S)

---

## SUMMARY: Complete Execution Order

```
Step 1:  pip install pydicom SimpleITK numpy matplotlib nibabel pyradiomics scikit-learn pandas openpyxl xlrd scipy Pillow

Step 2:  python experiment1_visualization.py
         → Check experiment1_results/
         → Open .nrrd files in 3D Slicer, take screenshots

Step 3:  python experiment2_segmentation.py
         → Check experiment2_results/
         → Open mask in 3D Slicer overlaid on volume, take screenshots

Step 4:  python experiment3_radiomics.py
         → Check experiment3_results/
         → radiomics_features.txt is your deliverable

Step 5:  python experiment4_prediction.py
         → Check experiment4_results/
         → All plots and model_evaluation_report.txt ready for report
```

---

## TROUBLESHOOTING

### "No DICOM files found"
- Check that your `tcga_brca` folder has patient subfolders with DICOM files
- DICOM files may not have `.dcm` extension - they might have no extension at all
- Try navigating deeper into the folder structure

### "Module not found" errors
- Run `pip install <module_name>` for the missing module
- Make sure you're using the correct Python environment (check bottom-right of PyCharm)

### "Mask has no labeled voxels"
- The segmentation may need parameter tuning
- Try adjusting the threshold multiplier in `experiment2_segmentation.py`
- You can also manually segment in 3D Slicer (see manual segmentation section below)

### Manual Segmentation in 3D Slicer (Backup Plan)
If automatic segmentation doesn't work well:

1. Load your volume in 3D Slicer
2. Go to **Modules > Segment Editor**
3. Click **Add** to create a new segment
4. Use the **Paint** tool to manually paint over the tumor region
5. Use **Threshold** tool to auto-select bright regions
6. Use **Grow from Seeds** for semi-automatic segmentation
7. **File > Save** to export as .nrrd

### "Column not found" in Experiment 4
- Open your `dataset.xlsx` in Excel to check exact column names
- Update the column name search terms in `experiment4_prediction.py`
- The Mammaprint and GHI columns might have different exact names

### PyRadiomics extraction fails
- Ensure the image and mask have the same dimensions
- Check that the mask has non-zero voxels (the tumor region)
- Try running with a simpler mask (just a binary threshold)
