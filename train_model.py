"""Train the Titanic survival model and save it to model.joblib.

Usage: place Kaggle's train.csv next to this file, then run
    python train_model.py
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
SEX_MAP = {"female": 0, "male": 1}
PORT_MAP = {"C": 0, "Q": 1, "S": 2}  # same order LabelEncoder used in the notebook


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda s: s.fillna(s.median()))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Sex"] = df["Sex"].map(SEX_MAP)
    df["Embarked"] = df["Embarked"].map(PORT_MAP)
    return df[FEATURES]


if __name__ == "__main__":
    train = pd.read_csv("train.csv")
    X, y = prepare(train), train["Survived"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("scaler", MinMaxScaler()),
        ("classifier", XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
    ])
    model.fit(X_tr, y_tr)
    print(f"Test accuracy: {model.score(X_te, y_te):.3f}")
    joblib.dump(model, "model.joblib")
    print("Saved model.joblib")
