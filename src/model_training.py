# model_training.py

import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from src.data_preprocessing import get_preprocessor


# Load data
DATA_PATH = os.path.join("data", "loan_data.csv")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["loan_status"] = df["loan_status"].str.strip().str.capitalize()

X = df.drop(columns=["loan_id", "loan_status"])
y = df["loan_status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=0.5
    ))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
os.makedirs("models", exist_ok=True)

# SAVE MODEL
with open(os.path.join("models", "loan_model.pkl"), "wb") as f:
    pickle.dump(pipeline, f)

print("Model trained and saved successfully")
