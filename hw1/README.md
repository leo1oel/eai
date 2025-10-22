# Embodied AI Homework 1 — 3DGS

**Release date:** Oct 19, 2025
**Deadline date:** Nov 9, 2025
**Goal:** Familiarize yourself with 3D Gaussian Splatting (3DGS) principles and workflows. All code has been tested on Windows and Linux. If your laptop does not support CUDA, you are encouraged to use Google Colab.

---

## Table of Contents

- [0. Setting up the environment](#0-setting-up-the-environment)
- [1. Preparation](#1-preparation)
- [2. Building Gaussian primitives](#2-building-gaussian-primitives)
- [3. Better rendering](#3-better-rendering)
- [4. About submission](#4-about-submission)
- [References](#references)
- [Contact](#contact)

---

## 0. Setting up the environment

Install dependencies:

```bash
pip install -r requirements.txt
```

> **Windows users:** Manually install **PyTorch + cudatoolkit** to match your CUDA version.

Additional requirements:

- Install **nvcc** (CUDA toolkit).
- Install the **gaussian rasterization kernel**.

Quick check (coding, **5 pts**):

```bash
python render.py
```

If everything is set up correctly, **(incorrectly) rendered** images will appear in the `renders/` folder. Include **one** image in your report to receive 5 points.

---

## 1. Preparation

Build the rasterization submodule:

```bash
cd submodules/diff-gaussian-rasterization
python setup.py build
python setup.py install
```

### Dataset and COLMAP

We provide a simple scene **“The Fruity Ice-cream.”**

- The images are in `assets/fruit.zip`. Extract them and use **COLMAP** to generate a sparse point cloud (via GUI or CLI).

Example CLI:

```bash
colmap automatic_reconstructor   --workspace_path $DATASET_PATH   --image_path $DATASET_PATH/images
```

**Task (writing && coding, 10 pts):** Watch the logs and **identify the stages** involved in the reconstruction pipeline.

### Point cloud initialization

Since **bundle adjustment** can be time-consuming, we provide a precomputed point cloud:

- `assets/points3D.ply`

**Task (coding, 10 pts):** Visualize this point cloud (serves as initialization of the Gaussians) with **MeshLab**, **Open3D**, or another tool you prefer. Include a screenshot in the report.

> We do **not** require you to train a 3DGS model (you’re welcome to try the official repository yourself; training can complete in ~1 hour).  
> Instead, we provide optimized Gaussians in `assets/gs_cloud.ply` and a streamlined inference codebase with core sections left blank for you to implement.

---

## 2. Building Gaussian primitives

In this section, you will construct **elliptic Gaussian** primitives and **splat** them into screen space.

- We denote a 3D Gaussian by its mean and covariance.

**(writing && coding, 15 pts)**

- Derive the **covariance matrix** from a **scaling matrix** and a **rotation matrix**.
- Complete `get_covariance` in `gaussian_model.py`.

**(writing, 5 pts)**

- Suppose a **viewing transformation** \( \mathbf{T}_v \) is applied to a Gaussian with mean \( \mu \) and covariance \( \Sigma \).  
  Write down the **transformed Gaussian** in **camera space**.

**(writing, 10 pts)**

- Suppose a **projection transformation** is applied. Although the mapping is **nonlinear**, approximate it **locally** at the mean by a **linear transformation** (its Jacobian).  
  Derive the expression of the Jacobian and the **new covariance** of the transformed Gaussian.

**(coding, 15 pts)**

- By **sequentially applying** the viewing and projection transformations, you will **splat the 3D Gaussian** into a **2D screen-space Gaussian**.  
  Complete `computeCov2D` in `forward.cu`.

**Rebuild the CUDA extension after changes:**

```bash
python setup.py clean --all
python setup.py build
python setup.py install
```

**(writing, 10 pts)**

- Given a **screen-space Gaussian covariance** \( \Sigma_{2D} \), find a **square bounding-box side length** that covers the **99% confidence** region.

> After implementing the above, running `python render.py` should generate rendered images in `renders/` within a few seconds.

---

## 3. Better rendering

**(coding, 10 pts)** **First-degree spherical harmonics (SH)**

- The current code supports only **0th-order SH**.
- Implement **1st-degree SH** by completing `eval_sh` in `sh_utils.py`.  
  Refer to the table of spherical harmonics (see **References**).  
  *Note:* To improve numerical stability, **alternative signs** for coefficients are used.
- Evaluate rendering performance **before vs. after** and include a comparison in your report.

**(coding, 10 pts)** **Post-hoc pruning**

- Implement **opacity-based pruning** by completing `prune_point` in `gaussian_model.py`.
- Experiment with various **opacity thresholds** and report the **maximum pruning ratio** that **does not** introduce visible artifacts.

**(bonus, 20 pts)**

- Propose and evaluate an **improved post-hoc pruning** strategy. Explain the method and show its performance.

---

## 4. About submission

- **What to submit:** Your **code** and **written report** packed as a `.zip`.
- **Where to submit:** **Web Learning**.
- **Deadline:** **Nov 9, 23:59**.


---

## References

- COLMAP: <https://colmap.github.io/>  
- Spherical harmonics (table for \(\ell=1,2\)):  
  <https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#%E2%84%93_=_1_2>  

---

## Contact

If you have any questions, contact **TA Gu Zhang**:  
**Email:** zg24@mails.tsinghua.edu.cn *(recommended)*  
**WeChat:** Available upon request.

---

*This README.md is adapted from the provided PDF assignment brief.*
