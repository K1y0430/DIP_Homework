# Assignment 4 Report: Implement Simplified 3D Gaussian Splatting

## 1. Introduction

In this assignment, we implemented a simplified version of 3D Gaussian Splatting (3DGS) in pure PyTorch and completed the pipeline from sparse reconstruction to Gaussian-based differentiable rendering. The whole workflow can be divided into three parts: recovering camera parameters and sparse 3D points with COLMAP, implementing the simplified Gaussian rendering pipeline, and comparing our implementation with the official 3DGS framework.

The `chair` scene was used in the experiments. This report presents the implementation details, training results, and qualitative comparison with the official implementation.

---

## 2. Task 1: Structure-from-Motion with COLMAP

In Task 1, COLMAP was used to recover the camera intrinsics, camera poses, and sparse 3D points from the input multi-view images. The recovered sparse point cloud was then used as the initialization of the Gaussian primitives in the following stage.

After reconstruction, the recovered 3D points were reprojected back to the image planes for verification. The reprojection results showed that the recovered points were generally aligned with the chair structure in the input images, which indicates that the recovered geometry and camera parameters were sufficiently accurate for the subsequent 3DGS training stage.

Although the sparse reconstruction itself is not dense enough for direct photorealistic rendering, it provides a reasonable geometric prior for Gaussian initialization.

---

## 3. Task 2: Simplified 3D Gaussian Splatting

### 3.1 Implementation

The simplified 3DGS pipeline was implemented in pure PyTorch. Compared with the official implementation, this version does not include the highly optimized rasterization pipeline or adaptive densification, but it still covers the core ideas of Gaussian-based scene representation and differentiable rendering.

The implementation mainly includes the following components:

1. **Gaussian initialization from sparse COLMAP points**
   Each sparse 3D point is converted into a Gaussian primitive with learnable position, color, opacity, rotation, and scale.

2. **3D covariance construction**
   The 3D covariance matrix of each Gaussian is constructed from the learnable rotation and scaling parameters.

3. **Projection from 3D to 2D**
   The Gaussian means and covariance matrices are projected from world space to image space through the camera extrinsics and intrinsics.

4. **2D Gaussian evaluation**
   The projected Gaussians contribute pixel values according to the 2D Gaussian density.

5. **Alpha blending rendering**
   After sorting Gaussians by depth, the final image is rendered by alpha compositing.

The model is optimized by minimizing the image reconstruction loss between rendered images and the corresponding ground-truth views.

---

### 3.2 Training Process

To observe the optimization process, debug images were saved during training. The following figures show the rendering quality at different training stages.

#### Early stage

![Early training result](figures/epoch_0000.png)

At the beginning of training, the rendered result is still very rough. The main chair structure is only vaguely visible, and the image is blurry with limited geometric consistency. This is expected because the Gaussian parameters have not yet adapted to the appearance of the scene.

#### Middle stage

![Middle training result](figures/epoch_0100.png)

After further optimization, the chair structure becomes much more recognizable. Major color regions and object boundaries begin to emerge, and the rendered image becomes more stable. However, the edges are still not sufficiently sharp, and some fine details remain unclear.

#### Late stage

![Late training result](figures/epoch_0199.png)

At the late stage of training, the rendered result is noticeably improved. The overall geometry is more coherent, the object boundary is clearer, and the appearance is closer to the target view. Although the simplified implementation is still limited in detail recovery, it can reconstruct the main visual structure of the scene successfully.

---

### 3.3 Final Multi-view Rendering

In addition to the debug images, a multi-view rendering video was generated after training to inspect the reconstructed scene from different viewpoints.

**Supplementary result:** [Final multi-view rendering video](figures/render_mv.mp4)

The video shows that the learned Gaussian representation can produce visually reasonable renderings from multiple viewpoints. The overall shape of the chair remains stable under viewpoint changes, which indicates that the simplified 3DGS model captures the main 3D structure of the scene rather than merely memorizing individual training images.

---

## 4. Task 3: Comparison with the Official 3DGS Implementation

To further evaluate the simplified implementation, we compared it with the official 3DGS framework on the same scene. Since the assignment mainly requires qualitative comparison in terms of rendering quality, the discussion here focuses on the visual results.

### 4.1 Official 3DGS Results

The following figures show representative renderings produced by the official implementation.

![Official 3DGS result 1](figures/00041.png)

![Official 3DGS result 2](figures/00062.png)

![Official 3DGS result 3](figures/00099.png)

From these examples, the official implementation produces cleaner object boundaries, more accurate color reconstruction, and sharper local details. The chair structure is more stable and visually more realistic.

---

### 4.2 Qualitative Comparison

Compared with the official 3DGS, our simplified implementation is able to reconstruct the global structure of the chair and produce meaningful novel-view renderings, which demonstrates that the essential Gaussian rendering mechanism is correctly implemented.

However, the simplified implementation still shows several limitations:

* The rendered images are blurrier than those from the official implementation.
* Fine details and local textures are less sharp.
* Object boundaries are not as clean or stable as those from the official framework.

These differences are reasonable because the simplified implementation only preserves the core rendering logic, while the official 3DGS includes more advanced engineering optimizations and a more complete training pipeline.

Overall, the simplified implementation successfully demonstrates the main idea of 3D Gaussian Splatting, while the official implementation achieves clearly better rendering quality.

---

## 5. Conclusion

In this assignment, we completed a simplified 3D Gaussian Splatting pipeline in PyTorch and verified that it can reconstruct a 3D scene from multi-view images. Starting from sparse COLMAP points, the model was able to learn Gaussian parameters and render reasonable scene appearances from different viewpoints.

The training results show that the simplified implementation gradually improves during optimization and finally captures the main geometry and appearance of the chair scene. The comparison with the official 3DGS further shows that, although our implementation is simpler and less accurate in details, it still reproduces the central idea of Gaussian-based differentiable rendering.

In summary, this assignment helped us understand the full workflow of 3D Gaussian Splatting, including Gaussian parameterization, projection, alpha blending, and the difference between a simplified educational implementation and a fully optimized official system.
