# Assignment 3 - Bundle Adjustment

## Digital Image Processing Course Assignment

This repository contains **Liu Feiyang (SA25001039)**'s implementation of **Assignment 3** for the **Digital Image Processing (DIP)** course.

In this assignment, I completed two tasks:

1. **Bundle Adjustment with PyTorch**
2. **3D Reconstruction with COLMAP**

The first task focuses on recovering the 3D structure, camera extrinsics, and shared focal length from multi-view 2D observations through optimization. The second task uses the COLMAP pipeline to perform full 3D reconstruction from rendered multi-view images.

---

## Repository Structure

```text
Assignment_03/
├─ README.md
├─ task1_bundle_adjustment.py
├─ run_colmap.sh
├─ visualize_data.py
├─ data/
│  ├─ images/
│  ├─ points2d.npz
│  └─ points3d_colors.npy
├─ outputs/
│  ├─ reconstructed.obj
│  └─ camera_params.npz
└─ figures/
   ├─ task1_loss_curve.png
   ├─ task1_point_cloud.png
   ├─ task1_reproj_1.png
   ├─ task1_reproj_2.png
   ├─ task1_reproj_3.png
   ├─ task2_sparse.png
   └─ task2_dense.png
```

---

## Environment Setup

It is recommended to use a **conda environment**.

### Create environment

```bash
conda create -n dip python=3.10
conda activate dip
```

### Install Python dependencies

```bash
pip install numpy matplotlib opencv-python torch torchvision
```

If additional packages are needed, they can be installed manually.

---

## Task 1: Bundle Adjustment with PyTorch

### 1. Task Description

In this task, I implemented **Bundle Adjustment** from scratch using PyTorch. The goal is to recover:

- the shared focal length \(f\),
- the extrinsic parameters \((R, T)\) for all 50 cameras,
- and the 3D coordinates of all sampled points,

from the provided 2D observations stored in `points2d.npz`.

The optimization is based on minimizing the **2D reprojection error** between the predicted image points and the observed image points.

---

### 2. Data Description

The following files are used in Task 1:

- `data/points2d.npz`: multi-view 2D observations. Each view contains an array of shape `(20000, 3)`, where each row is `(x, y, visibility)`.
- `data/points3d_colors.npy`: RGB colors of the 3D points, used to export the final colored point cloud.
- `data/images/`: rendered images for visualization and reprojection comparison.

The 2D observations are used for optimization, while the color file is used only for the final point cloud export.

---

### 3. Method

I parameterized the optimization variables as follows:

- **Focal length**: a single shared scalar parameter for all views.
- **Camera rotations**: represented by Euler angles.
- **Camera translations**: optimized directly as 3D vectors.
- **3D points**: optimized directly as learnable coordinates.

For a 3D point \(P = (X, Y, Z)\), the camera-space coordinates are computed by:

\[
[X_c, Y_c, Z_c]^T = R P + T
\]

According to the coordinate convention given in the assignment, the projection equations are:

\[
u = -f \frac{X_c}{Z_c} + c_x, \qquad
v = f \frac{Y_c}{Z_c} + c_y
\]

where \(c_x = \frac{W}{2}\) and \(c_y = \frac{H}{2}\).

The loss function is mainly the **reprojection error** between the predicted 2D points and the observed 2D points. Visibility masks are used so that only visible points contribute to the loss. To stabilize optimization, mild regularization terms are also added.

The optimization is performed with the **Adam** optimizer in PyTorch.

---

### 4. Running

To run Task 1:

```bash
python task1_bundle_adjustment.py
```

After optimization, the script saves:

- the loss curve,
- the reconstructed colored point cloud in `.obj` format,
- the optimized camera parameters,
- and several reprojection visualizations.

---

### 5. Input and Output

**Input**
- `points2d.npz`
- `points3d_colors.npy`
- rendered images in `data/images/` for visualization

**Output**
- optimized focal length
- optimized camera extrinsics
- optimized 3D point coordinates
- a colored OBJ point cloud
- loss curve
- reprojection visualizations

---

### 6. Results

#### 6.1 Optimization Loss

<img src="figures/task1_loss_curve.png" alt="Task 1 Loss Curve" width="700">

The optimization loss decreases rapidly at the beginning and then gradually converges, which shows that the bundle adjustment process is effective and numerically stable.

#### 6.2 Reprojection Visualization

<img src="figures/task1_reproj_1.png" alt="Task 1 Reprojection 1" width="700">
<img src="figures/task1_reproj_2.png" alt="Task 1 Reprojection 2" width="700">
<img src="figures/task1_reproj_3.png" alt="Task 1 Reprojection 3" width="700">

In the reprojection figures, the **observed points** and the **predicted points** largely overlap with each other, indicating that the optimized 3D structure and camera parameters can explain the 2D observations reasonably well.

#### 6.3 Reconstructed 3D Point Cloud

<img src="figures/task1_point_cloud.png" alt="Task 1 Point Cloud" width="700">

The reconstructed point cloud captures the overall bust structure, including the head, neck, and upper body region. Although the global 3D shape is not perfectly ideal, the result is consistent with the reprojection alignment and demonstrates that the implemented bundle adjustment is able to recover a meaningful 3D structure.

#### 6.4 Final Quantitative Results

You may summarize the final results in a small table like this:

| Metric | Value |
|--------|-------|
| Final focal length | **[fill in your value]** |
| Final reprojection RMSE | **[fill in your value]** |

---

### 7. Discussion

Task 1 shows that bundle adjustment can successfully reduce reprojection error and recover a meaningful 3D structure from multi-view 2D observations. The reprojection results are relatively good, which means that the optimized camera parameters and 3D points fit the observations well.

At the same time, the reconstructed 3D point cloud is not perfectly ideal in global shape. This is understandable because bundle adjustment mainly minimizes the 2D reprojection error, while the recovered 3D structure may still suffer from ambiguity or geometric degeneration without stronger geometric priors or regularization.

Overall, the PyTorch implementation successfully demonstrates the core idea of bundle adjustment.

---
