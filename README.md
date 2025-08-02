# QRT Financial Data Challenge – Liquid Asset Performance Reconstruction

This repository contains my submission for the QRT Financial Data Challenge 2025. The task involves predicting the sign of future asset returns based on historical price data and asset metadata.

## Project Overview

The objective is to reconstruct the directional return (`RET_TARGET`) of liquid assets using structured financial data. The target is categorized into three classes:
- 1 for positive return
- 0 for near-zero return
- -1 for negative return

The final prediction model uses LightGBM with custom feature engineering, class weighting, and a thresholding strategy for classification.

## Files and Structure

qrt-liquid-asset-challenge/
├── notebooks/
│ └── QRT_Data_Challenge.ipynb # Exploratory notebook and end-to-end development
├── scripts/
│ └── qrt_baseline.py # Clean script for training and submission
├── outputs/
│ └── submission.csv # Final submission predictions
├── data/
│ └── README.md # Instructions for handling large local CSVs
├── requirements.txt # Python dependencies
└── README.md # Project overview (this file)


## Approach

1. **Data Preparation**
   - Merged `X_train.csv`, `y_train.csv`, and `supplementary_data.csv` on `ID_BB_UNIQUE`.
   - Target transformed into sign-based classification: -1, 0, 1.

2. **Feature Engineering**
   - Rolling statistics: mean, standard deviation, max, min of RET columns.
   - Count of positive and negative returns across assets.
   - One-hot encoding for asset `CLASS_LEVEL_1` and `CLASS_LEVEL_2`.

3. **Modeling**
   - LightGBM Classifier.
   - Custom sample weights based on `abs(RET_TARGET)`.

4. **Thresholding Strategy**
   - If predicted probability of class 1 is > 0.6 → classify as 1.
   - If predicted probability of class 1 is < 0.4 → classify as -1.
   - Otherwise → classify as 0.

5. **Evaluation Metric**
   - Weighted accuracy score:
     ```python
     def weighted_accuracy_score(y_true, y_pred, weights):
         return np.sum(weights * (y_pred == y_true)) / np.sum(weights)
     ```

## Dependencies

All dependencies are listed in `requirements.txt`. Key packages include:

- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib
- seaborn

## Large Files

Due to GitHub size limitations, raw datasets (`X_train.csv`, `y_train.csv`, `X_test.csv`, `supplementary_data.csv`) are not included. See `data/README.md` for instructions on placing them locally.

## Author

Nicholas Hong  
B.Eng. Electrical and Electronic Engineering, NTU Singapore  
LinkedIn: [https://www.linkedin.com/in/nicholas-hong001](https://www.linkedin.com/in/nicholas-hong001)  
GitHub: [https://github.com/Nicholashsw](https://github.com/Nicholashsw)

---

Built for the QRT Financial Data Challenge 2025. 
