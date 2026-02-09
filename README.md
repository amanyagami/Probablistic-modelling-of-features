# Feature Visualization and Analysis

This repository contains exploratory notebooks for **visualizing and analyzing deep feature embeddings** extracted from trained models, with a particular focus on **class-wise structure** and **adversarial robustness (e.g., PGD attacks)**.

The notebooks are intended for research and analysis workflows where understanding the geometry of learned representations is important.

---

## Contents

- **`visualize only two classes.ipynb`**  
  Focused analysis of feature embeddings for **two selected classes**. This notebook is useful for studying separation, overlap, and robustness properties in a controlled, low-complexity setting.

- **`visualize_features_deeply.ipynb`**  
  A more comprehensive feature analysis notebook that scales the same ideas to **multiple classes and attack settings**, enabling deeper inspection of representation space.

---

## What the Notebooks Do

Both notebooks follow a similar analytical pipeline:

1. **Load model and dataset**  
   Load a pretrained model and the corresponding clean and adversarial datasets.

2. **Compute statistics**  
   - Mean and precision (or covariance-related) statistics of feature embeddings
   - Load and reuse precomputed feature statistics when available

3. **Feature / embedding visualization**  
   - Plot feature representations for individual classes
   - Overlay features from different attack types (e.g., PGD) for comparison
   - Visualize multiple classes in a shared embedding space

4. **Adversarial analysis**  
   - Evaluate embeddings on adversarial datasets (e.g., PGD attacks)
   - Compare clean vs adversarial feature distributions

5. **Geometric analysis**  
   - Compute distances of feature points from a reference (e.g., origin)
   - Estimate class centroids in feature space
   - Analyze distances to centroids and their implications

6. **CSV-based analysis**  
   - Post-process saved feature statistics
   - Perform quantitative analysis of embedding geometry

---

## Typical Use Cases

- Understanding **class separability** in learned representations
- Studying the **effect of adversarial attacks** on feature space geometry
- Comparing **clean vs adversarial embeddings**
- Investigating **centroid-based distances** as robustness or detection signals

---

## Requirements

The notebooks assume a standard deep learning and scientific Python stack, typically including:

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib / Seaborn
- Jupyter Notebook

(Exact versions depend on the training and evaluation setup used.)

---

## How to Use

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```

2. Open the notebooks:
   ```bash
   jupyter notebook
   ```

3. Update model paths, dataset paths, and attack configurations as needed.

4. Run cells sequentially to reproduce visualizations and analyses.

---

## Notes

- These notebooks are **analysis-oriented** and may require minor refactoring for use as a library or pipeline.
- Some results depend on precomputed statistics or CSV files; ensure paths are correctly set.
- The code is intended for experimentation and research rather than production use.

---

## License

Specify your license here (e.g., MIT, Apache 2.0).

