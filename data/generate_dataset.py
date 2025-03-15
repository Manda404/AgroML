import pandas as pd
import numpy as np
import random
import os

def generate_dataset(output_path="data/dataset.csv", n_samples=1000):
    np.random.seed(42)
    random.seed(42)
    
    start_date = pd.to_datetime("today") - pd.DateOffset(years=5)
    dates = pd.date_range(start=start_date, periods=n_samples, freq='D')
    station_ids = [f"Station_{i}" for i in range(1, 6)]
    fuel_types = ["Diesel", "Essence", "GPL"]
    quantities = np.random.randint(100, 5000, size=n_samples)
    prix_moyen = np.random.uniform(1.2, 2.5, size=n_samples)  # Prix par litre
    prix_total = quantities * prix_moyen
    humidity = np.random.uniform(10, 90, size=n_samples)  # Humidité en %
    wind_speed = np.random.uniform(0, 20, size=n_samples)  # Vitesse du vent en km/h
    holiday = np.random.choice([0, 1], size=n_samples)  # Jour férié ou non
    promotions = np.random.choice([0, 1], size=n_samples)  # Promotion active ou non
    day_of_week = [date.weekday() for date in dates]  # Jour de la semaine (0=Monday, 6=Sunday)
    
    df = pd.DataFrame({
        "date": dates,
        "station": np.random.choice(station_ids, size=n_samples),
        "fuel_type": np.random.choice(fuel_types, size=n_samples),
        "température": np.random.uniform(-10, 40, size=n_samples),
        "humidity": humidity,
        "wind_speed": wind_speed,
        "holiday": holiday,
        "promotions": promotions,
        "day_of_week": day_of_week,
        "quantité_vendue": quantities,
        "prix_moyen": prix_moyen,
        "prix_total": prix_total  # Target
    })
    
    # Introduire des valeurs manquantes
    df.loc[np.random.choice(df.index, size=50, replace=False), "température"] = np.nan
    df.loc[np.random.choice(df.index, size=30, replace=False), "prix_moyen"] = np.nan
    df.loc[np.random.choice(df.index, size=40, replace=False), "humidity"] = np.nan
    
    # Ajouter des outliers
    df.loc[np.random.choice(df.index, size=10, replace=False), "quantité_vendue"] *= 10
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset généré avec succès et sauvegardé dans {output_path} !")
    return df