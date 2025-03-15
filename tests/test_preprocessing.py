import unittest
import pandas as pd
import numpy as np
from preprocessing import handle_missing_values, handle_outliers, convert_categorical_to_numeric, preprocess_data

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """Crée un dataset avec des valeurs manquantes, des outliers et des colonnes catégorielles"""
        self.df = pd.DataFrame({
            "température": [20, np.nan, 25, np.nan, 30],
            "humidity": [50, np.nan, 70, 80, np.nan],
            "wind_speed": [5, 10, 15, 20, 25],
            "holiday": [1, 0, 1, 0, 1],
            "promotions": [0, 1, 0, 1, 0],
            "day_of_week": [0, 1, 2, 3, 4],
            "quantité_vendue": [200, 500, 300, 4000, 10000],  # Dernière valeur est un outlier
            "prix_moyen": [1.5, 2.0, np.nan, 1.8, 2.2],
            "prix_total": [300, 1000, np.nan, 7200, 22000],  # Doit être quantité * prix_moyen
            "station": ["Station_1", "Station_2", np.nan, "Station_1", "Station_3"],
            "fuel_type": ["Diesel", "Essence", "GPL", np.nan, "Diesel"]
        })

    def test_handle_missing_values(self):
        """Vérifie que les valeurs manquantes sont bien remplacées"""
        df_cleaned = handle_missing_values(self.df.copy())
        self.assertFalse(df_cleaned.isna().any().any(), "Il ne doit plus y avoir de valeurs manquantes après traitement")

    def test_handle_outliers(self):
        """Vérifie que les outliers ont bien été remplacés par le 95e percentile"""
        df_cleaned = handle_outliers(self.df.copy())
        upper_bound = np.percentile(self.df["quantité_vendue"], 95)
        self.assertLessEqual(df_cleaned["quantité_vendue"].max(), upper_bound, "Les outliers doivent être remplacés")

    def test_convert_categorical_to_numeric(self):
        """Vérifie que les colonnes catégorielles sont bien converties en numérique"""
        df_encoded = convert_categorical_to_numeric(self.df.copy())
        self.assertTrue(pd.api.types.is_integer_dtype(df_encoded["station"]), "La colonne 'station' doit être numérique")
        self.assertTrue(pd.api.types.is_integer_dtype(df_encoded["fuel_type"]), "La colonne 'fuel_type' doit être numérique")

    def test_preprocess_data(self):
        """Vérifie que l'ensemble du prétraitement fonctionne correctement"""
        df_processed = preprocess_data(self.df.copy())

        # Vérifier qu'il n'y a plus de valeurs manquantes
        self.assertFalse(df_processed.isna().any().any(), "Aucune valeur ne doit être manquante après le prétraitement")

        # Vérifier que les outliers ont bien été traités
        upper_bound = np.percentile(self.df["quantité_vendue"], 95)
        self.assertLessEqual(df_processed["quantité_vendue"].max(), upper_bound, "Les outliers doivent être traités")

        # Vérifier que les colonnes catégorielles sont converties
        self.assertTrue(pd.api.types.is_integer_dtype(df_processed["station"]), "La colonne 'station' doit être numérique après encodage")
        self.assertTrue(pd.api.types.is_integer_dtype(df_processed["fuel_type"]), "La colonne 'fuel_type' doit être numérique après encodage")

if __name__ == "__main__":
    unittest.main()