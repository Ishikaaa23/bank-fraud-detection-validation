"""
Feature engineering for the real bank fraud dataset.
Schema: Gender, Age, State, Account_Type, Transaction_Date, Transaction_Time,
        Transaction_Amount, Transaction_Type, Merchant_Category, Account_Balance,
        Transaction_Device, Transaction_Location, Device_Type,
        Transaction_Description, Is_Fraud
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ── Categorical mappings ───────────────────────────────────────────────────────
GENDER_MAP         = {'Male': 0, 'Female': 1}
ACCOUNT_TYPE_MAP   = {'Savings': 0, 'Checking': 1, 'Business': 2}
DEVICE_TYPE_MAP    = {'ATM': 0, 'POS': 1, 'Desktop': 2, 'Mobile': 3}
TRANSACTION_TYPE_MAP = {
    'Credit': 0, 'Bill Payment': 1, 'Debit': 2, 'Transfer': 3, 'Withdrawal': 4,
}
MERCHANT_CAT_MAP   = {
    'Groceries': 0, 'Health': 1, 'Restaurant': 2,
    'Clothing': 3, 'Entertainment': 4, 'Electronics': 5,
}

# High-risk device strings (partial match)
HIGH_RISK_DEVICES = ['voice assistant', 'payment gateway', 'pos mobile app']


def parse_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Transaction_Date (DD-MM-YYYY) and Transaction_Time (HH:MM:SS)."""
    df = df.copy()

    dt = pd.to_datetime(
        df['Transaction_Date'].astype(str) + ' ' + df['Transaction_Time'].astype(str),
        format='%d-%m-%Y %H:%M:%S', errors='coerce'
    )

    df['txn_hour']          = dt.dt.hour
    df['txn_day_of_week']   = dt.dt.dayofweek        # 0=Mon … 6=Sun
    df['txn_day_of_month']  = dt.dt.day
    df['txn_month']         = dt.dt.month
    df['txn_is_night']      = ((dt.dt.hour >= 22) | (dt.dt.hour <= 5)).astype(int)
    df['txn_is_weekend']    = (dt.dt.dayofweek >= 5).astype(int)
    df['txn_is_month_end']  = (dt.dt.day >= 28).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode all categorical columns to numeric."""
    df = df.copy()

    df['gender_enc']           = df['Gender'].map(GENDER_MAP).fillna(0).astype(int)
    df['account_type_enc']     = df['Account_Type'].map(ACCOUNT_TYPE_MAP).fillna(0).astype(int)
    df['device_type_enc']      = df['Device_Type'].map(DEVICE_TYPE_MAP).fillna(1).astype(int)
    df['transaction_type_enc'] = df['Transaction_Type'].map(TRANSACTION_TYPE_MAP).fillna(2).astype(int)
    df['merchant_cat_enc']     = df['Merchant_Category'].map(MERCHANT_CAT_MAP).fillna(0).astype(int)

    # High-risk device flag from Transaction_Device string
    device_lower = df['Transaction_Device'].str.lower().fillna('')
    df['is_high_risk_device'] = device_lower.apply(
        lambda d: int(any(h in d for h in HIGH_RISK_DEVICES))
    )

    # Location: extract state index as ordinal (frequency encoding would need full data)
    df['state_freq'] = df['State'].map(
        df['State'].value_counts(normalize=True).to_dict()
    ).fillna(0.0)

    return df


def engineer_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Amount and balance derived features."""
    df = df.copy()

    df['log_amount']          = np.log1p(df['Transaction_Amount'])
    df['log_balance']         = np.log1p(df['Account_Balance'])
    df['amount_to_balance']   = df['Transaction_Amount'] / (df['Account_Balance'] + 1)
    df['balance_after']       = df['Account_Balance'] - df['Transaction_Amount']
    df['would_overdraft']     = (df['balance_after'] < 0).astype(int)
    df['amount_gt_half_bal']  = (df['Transaction_Amount'] > df['Account_Balance'] * 0.5).astype(int)
    df['log_balance_after']   = np.log1p(df['balance_after'].clip(lower=0))

    return df


def engineer_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Age and demographic features."""
    df = df.copy()

    df['age_group'] = pd.cut(
        df['Age'],
        bins=[0, 25, 35, 50, 65, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(2.0)
    df['is_senior']  = (df['Age'] >= 60).astype(int)
    df['is_young']   = (df['Age'] <= 25).astype(int)

    return df


def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-feature interactions that are meaningful for fraud."""
    df = df.copy()

    df['night_x_high_amount']   = df['txn_is_night'] * df['log_amount']
    df['night_x_transfer']      = df['txn_is_night'] * (df['transaction_type_enc'] == 3).astype(int)
    df['mobile_x_high_amount']  = (df['device_type_enc'] == 3).astype(int) * df['log_amount']
    df['risk_device_x_amount']  = df['is_high_risk_device'] * df['log_amount']
    df['weekend_x_withdrawal']  = df['txn_is_weekend'] * (df['transaction_type_enc'] == 4).astype(int)
    df['young_x_high_amount']   = df['is_young'] * df['amount_to_balance']
    df['overdraft_x_night']     = df['would_overdraft'] * df['txn_is_night']

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline — call this on train and test sets."""
    df = parse_datetime_features(df)
    df = encode_categoricals(df)
    df = engineer_amount_features(df)
    df = engineer_demographic_features(df)
    df = engineer_interaction_features(df)
    return df


def get_numeric_feature_cols(df: pd.DataFrame, exclude: list = None) -> list:
    """Return all engineered numeric feature columns (excludes raw strings and target)."""
    exclude = exclude or []
    raw_string_cols = [
        'Gender', 'State', 'City', 'Account_Type', 'Transaction_Date',
        'Transaction_Time', 'Transaction_Type', 'Merchant_Category',
        'Transaction_Device', 'Transaction_Location', 'Device_Type',
        'Transaction_Description', 'Transaction_Currency', 'Is_Fraud',
    ]
    skip = set(raw_string_cols + exclude)
    return [c for c in df.columns if c not in skip and df[c].dtype in [np.float64, np.int64, float, int]]
