# LiDAR Point Cloud Augmentation & Feature Extraction

This project provides tools to **augment** and **analyze** LiDAR point cloud data (e.g. `.pcd`, `.las`). 

Point Cloud Data (Scan) Augmentation to improve diversity of scan dataset and save labeing cost.



---

## Tools

### 1. `pcd_aug.py` – Data Augmentation

Applies randomized augmentations to LiDAR point clouds:

- **Noising** – Simulate sensor noise in XYZ coordinates
- **Rotating** – Rotate point cloud around Z-axis
- **Scaling** – Apply random scale
- **Translating** – Shift points slightly in X/Y
- **RGB Noising** – Add variation to color values
- **RGB Light Effect** – Simulate lighting variation

```bash
python pcd_aug.py --input_folder ./input --output_folder ./augmented \
--noising 0.5 --scaling 0.5 --translating 0.1 --rotating 0.5 \
--rgb_noising 0.5 --rgb_light_effect 0.5 --aug_num 10
```

### 2. `pcd_feat_extr.py` - Feature extraction on las files

Computes local geometric features per point (e.g., planarity, linearity, omnivariance) using PCA-based techniques via jakteristics.

```
python pcd_feat_extr.py --input sample.las --output output.las
```
