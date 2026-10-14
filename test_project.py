import unittest

import numpy as np
import pandas as pd

from generate_data import generate_dataset
from main import optimize_fertilizer


class FixedResponseModel:
    def predict(self, candidates: pd.DataFrame) -> np.ndarray:
        return 8 - (
            ((candidates["Fertilizer_N"] - 80) / 40) ** 2
            + ((candidates["Fertilizer_P"] - 50) / 30) ** 2
            + ((candidates["Fertilizer_K"] - 70) / 35) ** 2
        )


class ProjectTests(unittest.TestCase):
    def test_synthetic_data_is_reproducible_and_complete(self) -> None:
        first = generate_dataset(rows=90, seed=12)
        second = generate_dataset(rows=90, seed=12)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 90)
        self.assertEqual(set(first["Crop"]), {"Maize", "Rice", "Wheat"})
        self.assertTrue((first["Yield"] > 0).all())
        self.assertTrue({"Fertilizer_N", "Fertilizer_P", "Fertilizer_K"}.issubset(first.columns))

    def test_optimizer_selects_best_grid_candidate(self) -> None:
        context = {
            "Soil_N": 50,
            "Soil_P": 40,
            "Soil_K": 50,
            "Temperature": 24,
            "Humidity": 70,
            "Soil_pH": 6.5,
            "Rainfall": 180,
        }
        result = optimize_fertilizer(FixedResponseModel(), context, "Maize", input_penalty=0)
        self.assertEqual(
            (result["Recommended_N"], result["Recommended_P"], result["Recommended_K"]),
            (80, 50, 70),
        )
        self.assertGreater(result["Predicted_Yield"], 0)


if __name__ == "__main__":
    unittest.main()