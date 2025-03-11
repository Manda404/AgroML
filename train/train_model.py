import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
from preprocessing import preprocess_data
from generate_dataset import generate_dataset


def split_dataset(df):
    df["date"] = pd.to_datetime(df["date"])

    select_features = [
        'fuel_type', 'température', 'humidity', 'wind_speed',
        'holiday', 'promotions', 'day_of_week', 'prix_total'
    ]
    
    last_year = df["date"].max().year
    train_df = df[df["date"].dt.year < last_year]
    test_df = df[df["date"].dt.year == last_year]
    
    print("Données divisées : train (4 ans) et test (1 an).")

    return train_df[select_features], test_df[select_features]


def train_model():
    # Charger les données générées
    df = generate_dataset()   # pd.read_csv("./data/dataset.csv")

    df_cleaned = preprocess_data(df)

    train_df, test_df = split_dataset(df_cleaned)

    # Séparer en features (X) et target (y)
    X_train, X_test = train_df.drop(columns=["prix_total"]), test_df.drop(columns=["prix_total"])
    y_train, y_test = train_df["prix_total"], test_df["prix_total"]

    # Entraîner le modèle Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prédictions sur le test set
    y_pred = model.predict(X_test)

    # Calcul des métriques de régression
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Affichage des résultats
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Sauvegarder le modèle
    joblib.dump(model, "models/random_forest_model.pkl")

    return mae, rmse, r2


if __name__ == "__main__":
    train_model()
