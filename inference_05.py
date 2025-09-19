import pandas as pd
import joblib
import os

# ----------------------
# Configuración
# ----------------------
NEW_DATA = "data/preprocessed_new_imdb.csv"
MODEL = "models/logistic_model.joblib"
VECTORIZER = "models/tfidf_vectorizer.joblib"
OUTPUT_DIR = "output/inference_results.csv"
COLUMN = "clean_text"  # o el nombre real de tu columna de texto

# ----------------------
# 1. Cargar datos nuevos
# ----------------------
df_clean = pd.read_csv(NEW_DATA)

# ----------------------
# 3. Cargar modelo y vectorizador
# ----------------------
if not os.path.exists(MODEL):
	raise FileNotFoundError(f"Modelo no encontrado: {MODEL}")
if not os.path.exists(VECTORIZER):
	raise FileNotFoundError(f"Vectorizador no encontrado: {VECTORIZER}")

model = joblib.load(MODEL)
vectorizer = joblib.load(VECTORIZER)

# ----------------------
# 4. Vectorizar textos
# ----------------------
X_new = vectorizer.transform(df_clean[COLUMN])

# ----------------------
# 5. Inferencia
# ----------------------
predictions = model.predict(X_new)
label_mapping = {0: "neg", 1: "pos"}
df_clean["prediction"] = [label_mapping[p] for p in predictions]
df_clean["prob_positive"] = model.predict_proba(X_new)[:, 1]

df_clean.to_csv(OUTPUT_DIR, index=False)
print(f"✅ Inferencia completada. Resultados guardados en: {OUTPUT_DIR}")

