"""Generate a reproducible illustrative dataset; values are not farm measurements."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CROP_PROFILES = {
    "Maize": {"yield": 7.6, "n": 145, "p": 85, "k": 105, "temperature": 25, "ph": 6.3, "rainfall": 180},
    "Rice": {"yield": 6.5, "n": 135, "p": 82, "k": 110, "temperature": 27, "ph": 6.2, "rainfall": 220},
    "Wheat": {"yield": 5.4, "n": 115, "p": 75, "k": 95, "temperature": 19, "ph": 6.6, "rainfall": 150},
}


def generate_dataset(rows: int = 3000, seed: int = 42, output_path: Path | None = None) -> pd.DataFrame:
    if rows < 30:
        raise ValueError("Generate at least 30 rows for a train/test split.")
    rng = np.random.default_rng(seed)
    crops = np.array(list(CROP_PROFILES))
    records = []
    for _ in range(rows):
        crop = str(rng.choice(crops))
        profile = CROP_PROFILES[crop]
        soil_n = rng.uniform(15, 100)
        soil_p = rng.uniform(10, 80)
        soil_k = rng.uniform(15, 95)
        temperature = rng.uniform(12, 34)
        humidity = rng.uniform(45, 95)
        soil_ph = rng.uniform(5.0, 8.0)
        rainfall = rng.uniform(80, 300)
        fertilizer_n = rng.uniform(0, 160)
        fertilizer_p = rng.uniform(0, 100)
        fertilizer_k = rng.uniform(0, 140)

        nutrient_fit = np.exp(
            -0.5
            * (
                ((soil_n + 0.65 * fertilizer_n - profile["n"]) / 70) ** 2
                + ((soil_p + 0.65 * fertilizer_p - profile["p"]) / 42) ** 2
                + ((soil_k + 0.65 * fertilizer_k - profile["k"]) / 55) ** 2
            )
            / 3
        )
        climate_fit = np.exp(-0.5 * ((temperature - profile["temperature"]) / 8) ** 2)
        soil_fit = np.exp(-0.5 * ((soil_ph - profile["ph"]) / 0.9) ** 2)
        water_fit = np.exp(-0.5 * ((rainfall - profile["rainfall"]) / 100) ** 2)
        humidity_fit = np.exp(-0.5 * ((humidity - 75) / 25) ** 2)
        yield_value = (
            profile["yield"]
            * nutrient_fit
            * climate_fit
            * soil_fit
            * water_fit
            * humidity_fit
            + rng.normal(0, 0.22)
        )
        records.append(
            {
                "Soil_N": round(soil_n, 2),
                "Soil_P": round(soil_p, 2),
                "Soil_K": round(soil_k, 2),
                "Temperature": round(temperature, 2),
                "Humidity": round(humidity, 2),
                "Soil_pH": round(soil_ph, 2),
                "Rainfall": round(rainfall, 2),
                "Crop": crop,
                "Fertilizer_N": round(fertilizer_n, 2),
                "Fertilizer_P": round(fertilizer_p, 2),
                "Fertilizer_K": round(fertilizer_k, 2),
                "Yield": round(max(0.1, yield_value), 3),
            }
        )

    data = pd.DataFrame.from_records(records)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate illustrative synthetic fertilizer and yield data.")
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data" / "synthetic_fertilizer_dataset.csv")
    args = parser.parse_args()
    data = generate_dataset(args.rows, args.seed, args.output)
    print(f"Wrote {len(data)} synthetic rows to {args.output}")
    print("These illustrative values are not measured farm data.")


if __name__ == "__main__":
    main()