from src.preprocessing import load_data, preprocess_data, save_preprocessed_data, load_preprocessed_data
from src.feature_engineering import create_features, engineer_features
from src.model_supervised import train_logistic_regression, train_random_forest, train_svm, train_xgboost
from src.evaluation import evaluate_model
from sklearn.model_selection import train_test_split

def main():
    df = load_data("data/raw/creditcard.csv")
    df = create_features(df)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_scaled, scaler = preprocess_data(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_lr = train_logistic_regression(X_train_scaled, y_train)

    results_lr = evaluate_model(model_lr, X_test_scaled, y_test)

    model_svm = train_svm(X_train_scaled, y_train)
    results_svm = evaluate_model(model_svm, X_test_scaled, y_test)

    model_xgb = train_xgboost(X_train_scaled, y_train)
    results_xgb = evaluate_model(model_xgb, X_test_scaled, y_test)  
    print("Logistic Regression Results:", results_lr)
    print("SVM Results:", results_svm)
    print("XGBoost Results:", results_xgb)


    

if __name__ == "__main__":
    main()
