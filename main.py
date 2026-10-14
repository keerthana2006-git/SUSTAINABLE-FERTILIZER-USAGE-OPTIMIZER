import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from generate_data import generate_dataset


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_fertilizer_dataset.csv"
NUMERIC_FEATURES = [
	"Soil_N",
	"Soil_P",
	"Soil_K",
	"Temperature",
	"Humidity",
	"Soil_pH",
	"Rainfall",
	"Fertilizer_N",
	"Fertilizer_P",
	"Fertilizer_K",
]
CATEGORICAL_FEATURES = ["Crop"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Yield"


def build_model() -> Pipeline:
	preprocessing = ColumnTransformer(
		[
			("numeric", "passthrough", NUMERIC_FEATURES),
			("crop", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
		]
	)
	return Pipeline(
		[
			("preprocessing", preprocessing),
			(
				"regressor",
				RandomForestRegressor(
					n_estimators=250,
					min_samples_leaf=2,
					max_features=0.9,
					random_state=42,
					n_jobs=-1,
				),
			),
		]
	)


def optimize_fertilizer(
	model: Pipeline, context: dict[str, float], crop: str, input_penalty: float
) -> dict[str, float | str]:
	nitrogen = np.arange(0, 161, 10)
	phosphorus = np.arange(0, 101, 10)
	potassium = np.arange(0, 141, 10)
	candidates = pd.DataFrame(
		[
			{**context, "Crop": crop, "Fertilizer_N": n, "Fertilizer_P": p, "Fertilizer_K": k}
			for n in nitrogen
			for p in phosphorus
			for k in potassium
		]
	)
	predicted_yields = model.predict(candidates[FEATURES])
	total_input = candidates[["Fertilizer_N", "Fertilizer_P", "Fertilizer_K"]].sum(axis=1)
	utility = predicted_yields - input_penalty * total_input
	best_index = int(np.argmax(utility.to_numpy()))
	best = candidates.iloc[best_index]
	total = float(total_input.iloc[best_index])
	return {
		"Crop": crop,
		"Recommended_N": int(best["Fertilizer_N"]),
		"Recommended_P": int(best["Fertilizer_P"]),
		"Recommended_K": int(best["Fertilizer_K"]),
		"Predicted_Yield": round(float(predicted_yields[best_index]), 3),
		"Total_Fertilizer_Input": int(total),
		"Yield_per_Fertilizer_Unit": round(float(predicted_yields[best_index]) / max(total, 1), 4),
		"Objective_Score": round(float(utility.iloc[best_index]), 3),
	}


def run_analysis(data_path: Path, rows: int, seed: int, input_penalty: float) -> None:
	if not data_path.exists():
		generate_dataset(rows=rows, seed=seed, output_path=data_path)

	data = pd.read_csv(data_path)
	required = set(FEATURES + [TARGET])
	missing = sorted(required - set(data.columns))
	if missing:
		raise ValueError(
			f"Dataset is missing columns required for modeling: {', '.join(missing)}. "
			"Use data/synthetic_fertilizer_dataset.csv or generate it with generate_data.py."
		)
	if data.empty or data[FEATURES + [TARGET]].isna().any().any():
		raise ValueError("Dataset must contain rows with no missing model inputs or target values.")

	print("SUSTAINABLE FERTILIZER USAGE OPTIMIZER")
	print("Synthetic-data research prototype; not agronomic advice.\n")
	print(f"Dataset: {data_path.relative_to(ROOT) if data_path.is_relative_to(ROOT) else data_path}")
	print(f"Rows: {len(data)} | Crops: {', '.join(sorted(data['Crop'].unique()))}")
	print("\nMean synthetic yield by crop:")
	print(data.groupby("Crop")[TARGET].mean().round(2).to_string())

	train_x, test_x, train_y, test_y = train_test_split(
		data[FEATURES], data[TARGET], test_size=0.2, random_state=seed
	)
	model = build_model()
	model.fit(train_x, train_y)
	predictions = model.predict(test_x)
	baseline = DummyRegressor(strategy="mean").fit(train_x, train_y)
	baseline_predictions = baseline.predict(test_x)
	metrics = {
		"rows": int(len(data)),
		"test_rows": int(len(test_y)),
		"model_mae": round(float(mean_absolute_error(test_y, predictions)), 4),
		"model_rmse": round(float(np.sqrt(mean_squared_error(test_y, predictions))), 4),
		"model_r2": round(float(r2_score(test_y, predictions)), 4),
		"baseline_mae": round(float(mean_absolute_error(test_y, baseline_predictions)), 4),
		"baseline_rmse": round(float(np.sqrt(mean_squared_error(test_y, baseline_predictions))), 4),
		"baseline_r2": round(float(r2_score(test_y, baseline_predictions)), 4),
		"input_penalty_per_unit": input_penalty,
		"data_type": "synthetic; illustrative only",
	}

	model_dir = ROOT / "models"
	graph_dir = ROOT / "graph"
	model_dir.mkdir(exist_ok=True)
	graph_dir.mkdir(exist_ok=True)
	import joblib

	joblib.dump(model, model_dir / "fertilizer_yield_model.joblib")
	(model_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

	context = data[NUMERIC_FEATURES[:7]].median().to_dict()
	recommendations = pd.DataFrame(
		[optimize_fertilizer(model, context, crop, input_penalty) for crop in sorted(data["Crop"].unique())]
	)
	recommendations.to_csv(model_dir / "recommendations.csv", index=False)

	preprocessing = model.named_steps["preprocessing"]
	feature_names = preprocessing.get_feature_names_out()
	importances = model.named_steps["regressor"].feature_importances_
	importance = pd.Series(importances, index=feature_names).sort_values()
	figure, axes = plt.subplots(1, 2, figsize=(12, 5))
	for crop, group in data.groupby("Crop"):
		axes[0].scatter(
			test_y.loc[test_x.index].loc[test_x["Crop"] == crop],
			pd.Series(predictions, index=test_x.index).loc[test_x["Crop"] == crop],
			alpha=0.6,
			label=crop,
		)
	axes[0].plot([data[TARGET].min(), data[TARGET].max()], [data[TARGET].min(), data[TARGET].max()], "k--")
	axes[0].set(title="Held-out yield: actual vs predicted", xlabel="Actual yield", ylabel="Predicted yield")
	axes[0].legend()
	importance.tail(10).plot.barh(ax=axes[1], color="#3f8068")
	axes[1].set(title="Top model feature importances", xlabel="Importance", ylabel="")
	figure.tight_layout()
	figure.savefig(graph_dir / "model_diagnostics.png", dpi=160)
	plt.close(figure)

	print("\nHeld-out model metrics:")
	print(json.dumps(metrics, indent=2))
	print("\nInput-efficient recommendations for median soil/climate conditions:")
	print(recommendations.to_string(index=False))
	print("\nSaved model, metrics, and recommendations under models/; diagnostics under graph/.")
	print("Recommendations and metrics are based on synthetic data and are not field-validated.")


def main() -> None:
	parser = argparse.ArgumentParser(description="Train and evaluate the synthetic fertilizer optimizer.")
	parser.add_argument("--data", type=Path, default=DATA_PATH, help="CSV dataset path")
	parser.add_argument("--rows", type=int, default=3000, help="Rows generated if the dataset does not exist")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for generation and train/test split")
	parser.add_argument(
		"--input-penalty",
		type=float,
		default=0.002,
		help="Yield trade-off per fertilizer input unit in the optimizer objective",
	)
	args = parser.parse_args()
	if args.rows < 30:
		parser.error("--rows must be at least 30")
	if args.input_penalty < 0:
		parser.error("--input-penalty must be non-negative")
	run_analysis(args.data.resolve(), args.rows, args.seed, args.input_penalty)


if __name__ == "__main__":
	main()
