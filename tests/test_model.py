import unittest
import joblib
import pandas as pd
from train_model import train_model

class TestModelTraining(unittest.TestCase):

    def test_model_performance(self):
        """Vérifie que le modèle atteint des performances acceptables"""
        accuracy, precision, recall, f1 = train_model()
        
        self.assertGreater(accuracy, 0.65, "L'accuracy doit être > 0.65")
        self.assertGreater(f1, 0.65, "Le F1-score doit être > 0.65")

    def test_model_saving(self):
        """Vérifie que le modèle est bien sauvegardé"""
        train_model()
        model_path = "models/random_forest_model.pkl"
        try:
            model = joblib.load(model_path)
            self.assertIsNotNone(model, "Le modèle n'a pas été sauvegardé correctement")
        except FileNotFoundError:
            self.fail(f"Le fichier {model_path} n'a pas été créé")

if __name__ == "__main__":
    unittest.main()
