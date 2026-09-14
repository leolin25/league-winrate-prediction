from numpy import test
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import engineer_features, select_model_features
from src.model import train_model, evaluate_model, save_model

def main():
    # read the raw data
    df_raw = pd.read_csv('data/raw/high_diamond_ranked_10min.csv')

    # engineer the needed features for the model
    processed_df = engineer_features(df_raw)

    optimal_features = [
        'gold_diff', 
        'xp_diff', 
        'cs_diff', 
        'objective_diff',
        'first_blood',
        'herald_diff'
    ]

    X, y = select_model_features(processed_df, optimal_features)

    # split the data into 80% training data and 20% testing data 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # train and evaluate the model
    pipeline = train_model(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    for metric_name, value in metrics.items():
        print(f'{metric_name}: {value}')

    # save model
    os.makedirs('models', exist_ok=True)
    save_model(pipeline, 'models/baseline.joblib')

if __name__ == '__main__':
    main()