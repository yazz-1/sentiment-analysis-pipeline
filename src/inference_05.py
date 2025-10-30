import pandas as pd
import joblib
import os

# ----------------------
# Configuración
# ----------------------
NEW_DATA = "data/preprocessed_new_imdb.csv"
PIPELINE = "models/pipeline.joblib"
OUTPUT_DIR = "output/inference_results.csv"
COLUMN = "clean_text"

if __name__ == '__main__':
	# ----------------------
	# 1. Cargar datos nuevos
	# ----------------------
	df_clean = pd.read_csv(NEW_DATA)

	# ----------------------
	# 2. Cargar pipeline
	# ----------------------
	if not os.path.exists(PIPELINE):
		raise FileNotFoundError(f"Pipeline no encontrada: {PIPELINE}")

	pipeline = joblib.load(PIPELINE)

	# ----------------------
	# 3. Inferencia
	# ----------------------
	predictions = pipeline.predict(df_clean['clean_text'])
	label_mapping = {0: "neg", 1: "pos"}
	df_clean["prediction"] = predictions
	df_clean["prob_positive"] = pipeline.predict_proba(df_clean['clean_text'])[:, 1]

	df_clean.to_csv(OUTPUT_DIR, index=False)
	print(f"✅ Inferencia completada. Resultados guardados en: {OUTPUT_DIR}")
