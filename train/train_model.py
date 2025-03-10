import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib  # Pour sauvegarder le modèle
from src.preprocessing import preprocess_data


def split_dataset(df):

    df["date"] = pd.to_datetime(df["date"])

    select_features = ['fuel_type',
                        'température',
                        'humidity',
                        'wind_speed',
                        'holiday',
                        'promotions',
                        'day_of_week',
                        'prix_total'
                        ]
    
    last_year = df["date"].max().year
    train_df = df[df["date"].dt.year < last_year]
    test_df = df[df["date"].dt.year == last_year]
    
    print("Données divisées : train (4 ans) et test (1 an).")

    return train_df[select_features], test_df[select_features]

def train_model():
    # Charger les données générées
    df = pd.read_csv("data/dataset.csv")
    
    df_cleaned = preprocess_data(df)
    
    train_df, test_df = split_dataset(df_cleaned)
    
    # Séparer en train/test
    X_train, X_test, y_train, y_test = train_df.drop(columns=["prix_total"]), test_df.drop(columns=["prix_total"]) ,train_df["prix_total"], test_df["prix_total"]

    # Entraîner le modèle Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prédictions sur le test set
    y_pred = model.predict(X_test)

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Affichage des résultats
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Sauvegarder le modèle
    joblib.dump(model, "models/random_forest_model.pkl")

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    train_model()