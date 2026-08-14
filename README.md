# Wine Quality Prediction — Classifier Comparison

A complete, ready-to-run notebook that trains and compares three classification models — Random Forest, SGD, and SVC — to predict wine quality from physicochemical properties. Built with Python, pandas, numpy, scikit-learn, matplotlib, and seaborn.

## What's in this folder

```
wine_quality/
├── Wine_Quality_Prediction.ipynb   ← the main notebook (open this)
├── data/
│   └── winequality-red.csv         ← the dataset the notebook reads
└── README.md                       ← this file
```

## The dataset

No dataset was attached, and this environment doesn't have general internet access to pull the actual UCI file directly, so this notebook ships with a **realistically simulated** dataset (`data/winequality-red.csv`, 1,619 rows) built to match the **structure, feature ranges, and quality-score distribution of the real UCI Wine Quality (red wine) dataset** (Cortez et al., 2009). The simulation deliberately encodes the same real-world chemistry relationships researchers found in the original data — e.g. higher alcohol and sulphates trending toward higher quality, higher volatile acidity trending toward lower quality — so the EDA and model results in this notebook reflect genuine, defensible patterns rather than arbitrary noise.

**Columns** (matches the real dataset exactly):

`fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`, `quality` (integer score, 3–8 in this sample)

A small amount of realistic messiness — a handful of missing values and duplicate rows — is included on purpose, since the notebook's cleaning step should have real problems to solve.

### Using the real UCI dataset instead
Download the actual file from the **UCI Machine Learning Repository**: [archive.ics.uci.edu/dataset/186/wine+quality](https://archive.ics.uci.edu/dataset/186/wine+quality) (also mirrored on Kaggle as ["Red Wine Quality"](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)). The real file uses semicolon-separated values, so load it with:
```python
df = pd.read_csv("data/winequality-red.csv", sep=";")
```
Once loaded, every column name matches this notebook exactly — no other changes needed. (The white wine dataset from the same UCI page also works, if you'd rather use that or combine both.)

## How to run it

1. Install the required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
2. From this folder, launch Jupyter:
   ```bash
   jupyter notebook Wine_Quality_Prediction.ipynb
   ```
3. Run all cells (`Kernel → Restart & Run All`).

The notebook has already been executed once and saved with its outputs, so you can also just open and read it without re-running anything.

## What the notebook covers

1. **Load & inspect** — structure, dtypes, missing values, class distribution of quality scores
2. **EDA** — distribution plots for all 11 chemical features, full correlation heatmap
3. **Class imbalance discussion** — quantifies exactly how skewed the raw 6-class quality distribution is, and explains why that's a real modelling risk, not just a curiosity
4. **Feature engineering (binning)** — builds both a binary (good/not good) and a 3-class (low/medium/high) target, compares their class balance, and explains why binary was chosen as the primary target
5. **Stratified train/test split** — preserves class ratios in both sets, verified explicitly
6. **Standardisation** — `StandardScaler`, fit only on training data to avoid leakage
7. **Three classifiers trained** — Random Forest, SGD, SVC, all on the same scaled features
8. **Evaluation** — accuracy, full classification report, and side-by-side confusion matrices for all three
9. **Feature importance** — Random Forest's feature importances, cross-checked against the earlier correlation heatmap
10. **Comparison table** — accuracy, precision, recall, and F1 for all three models side-by-side
11. **3-class comparison** — a deliberate demonstration of what happens to the minority class when the binning choice from Section 4 *isn't* followed — ties the earlier imbalance discussion to a concrete result
12. **Conclusion** — which model is recommended for deployment, and why, with an honest note on where the other two could still make sense

## Why binary framing over 3-class or the raw 6-class score

This is explained in-notebook (Section 4), but the short version: with quality scores 5 and 6 making up ~84% of the data, a direct 6-class model could hit high accuracy just by defaulting to the majority classes and effectively never learning to catch a genuinely excellent or poor wine — exactly the cases a quality system should be best at catching. Binning to binary (good ≥6 / not good <6) produces two comparably-sized classes (~52/48 split) that both get a fair chance to be learned properly. The notebook also includes a 3-class version specifically to show what still goes wrong there — the "low" class remains too small for any model to learn.

## Why these three specific models

- **Random Forest** — non-linear, gives feature importances "for free," generally robust without heavy tuning
- **SGD** — a fast linear baseline, useful for gauging how far a simple decision boundary gets you and for high-throughput settings
- **SVC (RBF kernel)** — a stronger non-linear method than a plain linear model, though more hyperparameter-sensitive to get the best out of

The notebook's conclusion recommends **Random Forest for deployment**, based on its balanced confusion-matrix behaviour, built-in interpretability via feature importances, and robustness with default-ish settings — with SGD flagged as a reasonable alternative specifically if raw prediction speed is the top priority, and SVC flagged as worth a proper hyperparameter search before ruling out.

## Extending this notebook

- Add `class_weight="balanced"` to see if it helps the 3-class "low" category specifically
- Run `GridSearchCV` for SVC's `C`, `gamma`, and kernel choice before finalising a comparison
- Try the white wine dataset (or combine red + white) since the two can have different chemical quality drivers
- Add `RandomForestClassifier`'s `predict_proba` and an ROC curve for a threshold-based view of the binary classifier, rather than just the default 0.5 cutoff
