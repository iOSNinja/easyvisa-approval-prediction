# EasyVisa — Visa Approval Prediction (Classification)

> Ensemble classifiers that predict US work-visa certification and recommend applicant profiles, handling class imbalance.


## Problem

The OFLC (via EasyVisa) wants to shortlist visa applications likely to be certified and understand the drivers of case status.


## At a glance

- **Problem type:** Supervised Binary Classification (ensembles)
- **Target:** `case_status` — Certified / Denied
- **Primary metric(s):** F1 / Recall (primary), Precision, Accuracy; confusion matrix; cross-validated scorer
- **Tech stack:** Python, pandas, NumPy, scikit-learn (ensembles, GridSearch/RandomizedSearch), XGBoost, imbalanced-learn (over/undersampling), matplotlib, seaborn


## Data

`EasyVisa.csv` (~1.8 MB) — continent, education, job experience, training need, employer size & year, region, prevailing wage, wage unit, full-time flag.


## Approach / Steps

1. Data overview & type checks; fix negative `no_of_employees`
2. EDA — univariate & bivariate (education, continent, experience, wage vs. case status)
3. Preprocessing — outlier check, encoding, train/test split
4. Define criterion (F1 / Recall) and metric helper functions
5. Model building on original, oversampled, and undersampled data
6. Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost
7. Hyperparameter tuning of the top models
8. Model performance summary, final model selection, feature importance
9. Actionable insights & applicant-profile recommendations


## Repo structure

```
easyvisa-approval-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
├── data/
```


## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook   # open the notebook in notebooks/
```


## License

Private project — Not licensed for distribution.
