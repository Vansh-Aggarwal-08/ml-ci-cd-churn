import unittest
import joblib
from sklearn.ensemble import RandomForestClassifier

class TestChurnModelTraining(unittest.TestCase):
    def test_model_training(self):
        # Load saved churn model
        model = joblib.load("model/churn_model.pkl")

        # Check that model is a RandomForest
        self.assertIsInstance(model, RandomForestClassifier)

        # Check it has learned feature importances
        self.assertGreaterEqual(len(model.feature_importances_), 10)  # Telco has many features

if __name__ == "__main__":
    unittest.main()
