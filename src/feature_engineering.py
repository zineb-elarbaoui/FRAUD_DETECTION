import numpy as np
def create_features(df):
    """Create or transform features."""
    df["TransactionAmountLog"] = df["Amount"].apply(lambda x: np.log1p(x))
    # Add any domain features here
    return df
def select_features(df, feature_list):
    """Select a subset of features."""
    return df[feature_list]

def engineer_features(df):
    """Perform feature engineering: create and select features."""
    df = create_features(df)
    selected_features = [
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
        "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "TransactionAmountLog"
    ]
    df = select_features(df, selected_features)
    return df
