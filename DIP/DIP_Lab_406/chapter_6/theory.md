Here’s a comprehensive list of algorithms and techniques described in the attached PDF (Chapter 6 of a Digital Image Processing text), organized by topic:

---

## 🧮 Filtering and Enhancement Algorithms

### 1. **Geometric Mean Filter**
- Implemented using `colfilt` (MATLAB/Octave) or `generic_filter` (Python).

### 2. **Alpha-Trimmed Mean Filter**
- Also implemented using `colfilt` or `generic_filter`.

### 3. **Root-Mean-Square Filter**
- MATLAB/Octave implementation shown.

### 4. **Unsharp Masking**
- Applied after a 3×3 averaging filter to reverse blurring effects.

### 5. **Kuwahara Filter**
- Suggested to be rewritten as a single function compatible with `colfilt` or `generic_filter`.

---

## 🔍 Interpolation Algorithms

### 6. **Nearest Neighbor Interpolation**
- Assigns value of the nearest original pixel.

### 7. **Linear Interpolation**
- Uses weighted average between two adjacent values.

### 8. **Bilinear Interpolation**
- Applies linear interpolation in both x and y directions.

### 9. **General Interpolation Framework**
- Uses a function \( R(u) \) to interpolate between values.

### 10. **Cubic Interpolation**
- Uses a cubic polynomial \( R_3(u) \) over four points.

### 11. **Bicubic Interpolation**
- Applies cubic interpolation in both directions using 16 surrounding pixels.

---

## 🖼️ Image Scaling Algorithms

### 12. **Image Enlargement via imresize**
- Methods: `'nearest'`, `'bilinear'`, `'bicubic'`.

### 13. **Zero-Interleaving and Spatial Filtering**
- Inserts zeros between pixels and applies filters:
  - Nearest neighbor: `[[1 1 0]; [1 1 0]; [0 0 0]]`
  - Bilinear: `[[1 2 1]; [2 4 2]; [1 2 1]] / 4`
  - Bicubic: 5×5 filter approximating cubic interpolation

### 14. **Image Minimization via Subsampling**
- Removes pixels at regular intervals (e.g., every 4th pixel).

---

## 🔄 Image Rotation Algorithms

### 15. **Rotation via imrotate or transform.rotate**
- Uses interpolation methods: `'nearest'`, `'bilinear'`, `'bicubic'`.

### 16. **Efficient Rotation for Multiples of 90°**
- Uses matrix operations: `flipud`, `fliplr`, `rot90`.

---

## 🧭 Perspective Correction Algorithm

### 17. **Perspective Distortion Correction**
- Warps a trapezoid into a rectangle using linear stretching:
  - Computes stretch factor `str(x)` or squash function `sq(x)`
  - Applies transformation to pixel coordinates

---
