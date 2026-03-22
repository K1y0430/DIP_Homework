# Assignment 1 - Image Warping

## Digital Image Processing Course Assignment

This repository contains **Liu Feiyang (SA25001039)**'s implementation of **Assignment 1** for the **Digital Image Processing (DIP)** course.

The goal of this assignment is to understand and implement two representative image warping tasks:

1. **Basic Image Geometric Transformation**
2. **Point-Guided Image Deformation**

This project uses **Gradio** to provide an interactive interface, so that the transformation and deformation results can be directly observed in the browser.

---

##  Completed Tasks

In this assignment, I completed the following two tasks.

### Task 1: Basic Image Geometric Transformation

<img src="figures/task1.png" alt="Task 1" width="800">

In this task, I implemented global geometric transformation on an input image, including:

- scaling
- rotation
- translation
- horizontal flipping

The transformation is controlled by interactive parameters, and scaling and rotation are implemented around the image center.

### Task 2: Point-Guided Image Deformation

<img src="figures/task2.png" alt="Task 2" width="800">

In this task, I implemented interactive point-guided image warping.

The user first uploads an image, then selects multiple pairs of control points:

- one **source point**
- one corresponding **target point**

The image is then locally deformed according to these point correspondences, producing non-rigid warping effects.

---


## Requirements

To install requirements:

```setup
python -m pip install -r requirements.txt
```


## Running

To run basic transformation, run:

```basic
python run_global_transform.py
```

To run point guided transformation, run:

```point
python run_point_transform.py
```

## Results 
### Basic Transformation
<img src="pics/global_demo.gif" alt="alt text" width="800">

### Point Guided Deformation:
<img src="pics/point_demo.gif" alt="alt text" width="800">

## Acknowledgement

>📋 Thanks for the algorithms proposed by [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf).
