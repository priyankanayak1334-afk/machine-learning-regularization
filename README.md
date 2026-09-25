# machine-learning-regularization
# Day 13: Defeating Overfitting with Regularization (Ridge & Lasso)

## 📌 Project Overview
Production Machine Learning systems frequently fail when models overfit their training data, memorizing noise instead of generalizing to real-world scenarios. This project documents a hands-on technical diagnostic comparing a baseline **Ordinary Least Squares (OLS)** linear regression model against **Ridge (L2)** and **Lasso (L1)** regularized models on a complex, noisy dataset.

---

## 📊 Performance Comparison Matrix

The table below summarizes the trade-offs observed when regularizing the model coefficients:

| Model | Train RMSE | Test RMSE | Train R² | Test R² | Active Features Used | Behavior Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline (OLS)** | **18.21** | **34.50** | **0.942** | **0.781** | 30 / 30 | ❌ **Overfitting.** High variance gap. Explodes coefficient values to fit structural noise. |
| **Ridge (L2)** | 21.45 | 27.10 | 0.910 | 0.865 | 30 / 30 |  **Stable.** Uniformly shrinks coefficients. Drastically improves test metrics. |
| **Lasso (L1)** | 22.02 | 26.85 | 0.901 | 0.868 | 12 / 30 | 🎯 **Sparse.** Drops 18 noisy features to absolute 0. Automates feature selection. |

---

## 🔍 Core Diagnostic Insights

1. **The Variance Gap (Signs of Overfitting):**
   The baseline OLS model exhibits a classic overfitting pattern: it achieves a stellar training R² of `0.942` but degrades heavily to `0.781` on unseen testing data. This gap shows that the baseline model memorized the statistical noise of the 150 samples rather than mapping the core signal.
   
2. **L2 Regularization (Ridge) Behavior:**
   By adding a squared magnitude penalty to the loss function (α=15.0), Ridge restricts features from taking on erratic, massive weights. While training performance drops slightly, its Test RMSE drops significantly, proving excellent generalization.

3. **L1 Regularization (Lasso) Behavior:**
   Lasso applies an absolute magnitude penalty (α=2.5). It acts as an embedded feature selection tool by forcing irrelevant or highly correlated feature coefficients to exactly zero. It successfully filtered out 18 noisy features, matching Ridge's predictive performance with a much simpler model architecture.

---

## 🛠️ Tech Stack & Environment
* **Language:** Python 3.10+
* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Workflow Environment:** Jupyter Notebook

---

## 🚀 How to Run the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com
   cd your-repo-name
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Notebook:**
   ```bash
   jupyter notebook notebooks/regularization_comparison.ipynb
   ```
