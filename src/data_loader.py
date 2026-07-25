"""Dataset loaders for the credit-scoring project.

Graduated out of notebook 01 so every notebook and script shares one source of truth
for the German Credit schema instead of copy-pasting it.
"""
from pathlib import Path

import pandas as pd

# Repo root = one level up from this file's folder (src/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]

GERMAN_COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment_since", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age_years", "other_installment",
    "housing", "existing_credits", "job", "num_dependents", "telephone",
    "foreign_worker", "target",
]

GERMAN_CODE_MAP = {
    "checking_status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "no account"},
    "credit_history": {
        "A30": "no credits/all paid", "A31": "all paid this bank", "A32": "paid duly till now",
        "A33": "past delays", "A34": "critical/other credits"},
    "purpose": {
        "A40": "car (new)", "A41": "car (used)", "A42": "furniture/equip", "A43": "radio/tv",
        "A44": "appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
        "A48": "retraining", "A49": "business", "A410": "other"},
    "savings_status": {
        "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM",
        "A65": "unknown/none"},
    "employment_since": {
        "A71": "unemployed", "A72": "< 1 yr", "A73": "1-4 yrs", "A74": "4-7 yrs", "A75": ">= 7 yrs"},
    "personal_status_sex": {
        "A91": "male div/sep", "A92": "female div/sep/mar", "A93": "male single",
        "A94": "male mar/wid", "A95": "female single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {
        "A121": "real estate", "A122": "life insurance", "A123": "car/other", "A124": "unknown/none"},
    "other_installment": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {
        "A171": "unempl/unskilled-nonres", "A172": "unskilled-res", "A173": "skilled",
        "A174": "management/self-emp"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

# Which columns are genuinely numeric (everything else coded is categorical).
GERMAN_NUMERIC = [
    "duration_months", "credit_amount", "installment_rate", "residence_since",
    "age_years", "existing_credits", "num_dependents",
]
GERMAN_CATEGORICAL = list(GERMAN_CODE_MAP.keys())


def load_german_credit(path: str | Path | None = None, decode: bool = True) -> pd.DataFrame:
    """Load German Credit and add a 0/1 `default` column (1 = defaulted).

    If `decode` is True, coded categorical values (A11, A34, ...) are replaced with
    readable labels.
    """
    if path is None:
        path = REPO_ROOT / "data" / "raw" / "german.data"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=GERMAN_COLUMNS)
    df["default"] = (df["target"] == 2).astype(int)  # 1 = bad/default, 0 = good/repaid
    if decode:
        for col, mapping in GERMAN_CODE_MAP.items():
            df[col] = df[col].map(mapping)
    return df
