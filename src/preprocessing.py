import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """Load dataset."""
    return pd.read_csv(path)

def preprocess_data(df):
    """Scale numeric features and return scaled data + scaler."""
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled, scaler
def save_preprocessed_data(df, path):
    """Save preprocessed data to a CSV file."""
    df.to_csv(path, index=False)
def load_preprocessed_data(path):
    """Load preprocessed data from a CSV file."""
    return pd.read_csv