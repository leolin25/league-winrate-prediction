from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# function to create a pipeline for the data and train the model
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline([
        ('scalar', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

# evaluate the model and return a dict with performance metrics
def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob), # area under the ROC curve 
        'Log-Loss': log_loss(y_test, y_prob)
    }

    return metrics

# save model in disc
def save_model(model: Pipeline, filepath: str='models/baseline.joblib') -> None:
    joblib.dump(model, filepath)
    print(f'Model saved to {filepath}')

# load the model from disc
def load_model(filepath: str='models/baseline.joblib') -> Pipeline:
    return joblib.load(filepath)