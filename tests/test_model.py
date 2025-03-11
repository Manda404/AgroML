import unittest
import joblib
import os
from train_model import train_model

class TestModelTraining(unittest.TestCase):

    def test_model_performance(self):
        """Vérifie que le modèle atteint des performances acceptables en régression"""
        mae, rmse, r2 = train_model()

        #self.assertLess(mae, 10, "Le MAE doit être < 10 (erreur moyenne raisonnable)")
        self.assertGreater(r2, 0.65, "Le R² doit être > 0.65 (bonne capacité de prédiction)")

    def test_model_saving(self):
        """Vérifie que le modèle est bien sauvegardé"""
        train_model()
        model_path = "models/random_forest_model.pkl"

        self.assertTrue(os.path.exists(model_path), f"Le fichier {model_path} n'a pas été créé")
        try:
            model = joblib.load(model_path)
            self.assertIsNotNone(model, "Le modèle n'a pas été sauvegardé correctement")
        except Exception as e:
            self.fail(f"Erreur lors du chargement du modèle : {str(e)}")

if __name__ == "__main__":
    unittest.main()
