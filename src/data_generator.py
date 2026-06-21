"""
Data loader and synthetic generator matching the real bank fraud dataset schema.
Real data: 23 columns + Is_Fraud label, Indian banking context (INR).

To use real data, call: load_real_data('path/to/your_dataset.csv')
For demo/testing, call: generate_dataset()
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import uuid

# ── Column groups ──────────────────────────────────────────────────────────────
PII_COLS = ['Customer_ID', 'Customer_Name', 'Customer_Contact',
            'Customer_Email', 'Transaction_ID', 'Merchant_ID', 'Bank_Branch']

TARGET_COL = 'Is_Fraud'

RAW_FEATURE_COLS = [
    'Gender', 'Age', 'State', 'Account_Type',
    'Transaction_Date', 'Transaction_Time', 'Transaction_Amount',
    'Transaction_Type', 'Merchant_Category', 'Account_Balance',
    'Transaction_Device', 'Transaction_Location', 'Device_Type',
    'Transaction_Description',
]

INDIAN_STATES = [
    'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Telangana', 'Gujarat',
    'Delhi', 'Rajasthan', 'Uttar Pradesh', 'West Bengal', 'Kerala',
    'Andhra Pradesh', 'Madhya Pradesh', 'Punjab', 'Haryana', 'Sikkim',
    'Goa', 'Assam', 'Odisha', 'Jharkhand', 'Bihar',
    'Dadra and Nagar Haveli and Daman and Diu',
]

STATE_CITIES = {
    'Maharashtra': ['Mumbai', 'Pune', 'Nagpur'],
    'Karnataka': ['Bengaluru', 'Mysuru', 'Hubli'],
    'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai'],
    'Telangana': ['Hyderabad', 'Warangal', 'Karimnagar'],
    'Gujarat': ['Ahmedabad', 'Surat', 'Vadodara'],
    'Delhi': ['New Delhi', 'Dwarka', 'Rohini'],
    'Rajasthan': ['Jaipur', 'Jodhpur', 'Udaipur'],
    'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Agra'],
    'West Bengal': ['Kolkata', 'Howrah', 'Durgapur'],
    'Kerala': ['Thiruvananthapuram', 'Kochi', 'Kozhikode'],
    'Andhra Pradesh': ['Guntur', 'Visakhapatnam', 'Vijayawada'],
    'Madhya Pradesh': ['Bhopal', 'Indore', 'Jabalpur'],
    'Punjab': ['Amritsar', 'Ludhiana', 'Chandigarh'],
    'Haryana': ['Gurugram', 'Faridabad', 'Ambala'],
    'Sikkim': ['Gangtok', 'Jorethang', 'Namchi'],
    'Goa': ['Panaji', 'Margao', 'Vasco da Gama'],
    'Assam': ['Guwahati', 'Dibrugarh', 'Silchar'],
    'Odisha': ['Bhubaneswar', 'Cuttack', 'Rourkela'],
    'Jharkhand': ['Ranchi', 'Jamshedpur', 'Dhanbad'],
    'Bihar': ['Patna', 'Gaya', 'Bhagalpur'],
    'Dadra and Nagar Haveli and Daman and Diu': ['Daman', 'Silvassa', 'Diu'],
}

TRANSACTION_DEVICES = [
    'Debit/Credit Card', 'Voice Assistant', 'POS Mobile App', 'Bank Branch',
    'Smart Card', 'ATM Booth Kiosk', 'Payment Gateway Device', 'POS Mobile Device',
    'Mobile Banking App', 'Internet Banking', 'UPI App', 'NEFT/RTGS',
]

LEGIT_DESCRIPTIONS = [
    'Grocery shopping', 'Utility bill payment', 'Restaurant meal', 'Taxi fare',
    'Electricity bill', 'Mobile recharge', 'Salary credit', 'Rent payment',
    'Insurance premium', 'School fee', 'Medical consultation', 'Fuel payment',
    'Supermarket purchase', 'Online food order', 'Apparel purchase',
    'Courier charges', 'Car service', 'Student loan repayment', 'POS transaction',
    'ATM withdrawal', 'Tech gadgets purchase', 'Crowdfunding contribution',
    'Import duty payment', 'Electronics purchase', 'Specialty store shopping',
]

FRAUD_DESCRIPTIONS = [
    'Wire transfer', 'International transfer', 'Cryptocurrency purchase',
    'Gift card purchase', 'Casino transaction', 'Unknown merchant',
    'Rapid fund transfer', 'Overseas payment', 'Bulk cash withdrawal',
    'Third party transfer', 'POS transaction', 'ATM withdrawal',
    'Online transaction', 'Card not present', 'Contactless payment',
]


def load_real_data(path: str) -> pd.DataFrame:
    """Load the real dataset. Drops PII columns automatically."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in PII_COLS if c in df.columns], errors='ignore')
    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found. Check CSV."
    return df


def generate_dataset(n_samples: int = 100_000, fraud_rate: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic dataset matching the real schema (same column names + types).
    Replace this with load_real_data('your_dataset.csv') when running on real data.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    def make_block(n, is_fraud):
        states = rng.choice(INDIAN_STATES, size=n)
        cities = [random.choice(STATE_CITIES.get(s, ['Unknown'])) for s in states]

        base_date = datetime(2025, 1, 1)
        dates = [base_date + timedelta(days=int(d)) for d in rng.integers(0, 90, size=n)]
        date_strs = [d.strftime('%d-%m-%Y') for d in dates]

        if is_fraud:
            # Fraud skewed toward late night / early morning
            hours = rng.choice(np.arange(24), size=n,
                               p=_hour_probs([0, 1, 2, 3, 4, 23]))
        else:
            hours = rng.choice(np.arange(24), size=n,
                               p=_hour_probs([9, 10, 12, 14, 17, 18]))
        minutes = rng.integers(0, 60, size=n)
        seconds = rng.integers(0, 60, size=n)
        time_strs = [f'{h:02d}:{m:02d}:{s:02d}' for h, m, s in zip(hours, minutes, seconds)]

        amounts = (rng.lognormal(mean=11.0, sigma=1.0, size=n)
                   if is_fraud else
                   rng.lognormal(mean=10.5, sigma=0.8, size=n)).clip(14, 99_000)

        balances = rng.uniform(5_000, 100_000, size=n)

        descs = [random.choice(FRAUD_DESCRIPTIONS if is_fraud else LEGIT_DESCRIPTIONS)
                 for _ in range(n)]

        devices = (rng.choice(['POS Mobile App', 'Internet Banking', 'Payment Gateway Device',
                               'Voice Assistant', 'Debit/Credit Card'], size=n)
                   if is_fraud else
                   rng.choice(TRANSACTION_DEVICES, size=n))

        device_types = rng.choice(
            ['Mobile', 'Desktop'] if is_fraud else ['POS', 'Mobile', 'Desktop', 'ATM'],
            size=n
        )

        return {
            'Gender':                rng.choice(['Male', 'Female'], size=n),
            'Age':                   rng.integers(18, 71, size=n),
            'State':                 states,
            'Account_Type':          rng.choice(['Savings', 'Checking', 'Business'], size=n,
                                                 p=[0.6, 0.25, 0.15]),
            'Transaction_Date':      date_strs,
            'Transaction_Time':      time_strs,
            'Transaction_Amount':    amounts,
            'Transaction_Type':      rng.choice(
                                         ['Transfer', 'Withdrawal'] if is_fraud else
                                         ['Bill Payment', 'Debit', 'Transfer', 'Credit', 'Withdrawal'],
                                         size=n),
            'Merchant_Category':     rng.choice(
                                         ['Electronics', 'Entertainment'] if is_fraud else
                                         ['Groceries', 'Restaurant', 'Electronics',
                                          'Clothing', 'Health', 'Entertainment'],
                                         size=n),
            'Account_Balance':       balances,
            'Transaction_Device':    devices,
            'Transaction_Location':  [f'{c}, {s}' for c, s in zip(cities, states)],
            'Device_Type':           device_types,
            'Transaction_Description': descs,
            TARGET_COL:              int(is_fraud),
        }

    legit = make_block(n_legit, is_fraud=False)
    fraud = make_block(n_fraud, is_fraud=True)

    df_legit = pd.DataFrame(legit)
    df_fraud = pd.DataFrame(fraud)
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def _hour_probs(peak_hours, size=24):
    base = np.ones(size)
    for h in peak_hours:
        base[h] += 5
    return (base / base.sum()).tolist()
