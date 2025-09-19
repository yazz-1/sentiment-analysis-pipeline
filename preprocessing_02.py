import pandas as pd
import re
import logging
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stop_words

# Configurar logging
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s"
)

# Cargar el modelo de spaCy
try:
	nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  
except OSError:
	logging.error("Modelo de spaCy no encontrado. Ejecuta: python -m spacy download en_core_web_sm")
	raise


def clean_doc(doc):
	"""
	Procesa un documento spaCy y devuelve tokens lematizados sin stopwords ni puntuación.
	"""
	return [
		token.lemma_
		for token in doc
		if token.is_alpha and token.lemma_.lower() not in stop_words
	]


def preprocess_dataframe(df, text_column="text", output="clean_text", batch_size=2000, n_process=2):
	"""
	Aplica el preprocesamiento al DataFrame usando spaCy con procesamiento en paralelo.

	Parámetros:
	- df: DataFrame
	- text_column: columna de texto original
	- output: 'clean_text' o 'tokens'
	- batch_size: tamaño de lote para spaCy (default=2000)
	- n_process: número de núcleos a usar (default=2)

	Retorna:
	- DataFrame con columna adicional
	"""
	logging.info(f"Preprocesando '{text_column}' con salida '{output}' (batch_size={batch_size}, n_process={n_process})")
	df = df.dropna(subset=[text_column])
	df[text_column] = df[text_column].fillna("")

	texts = df[text_column].astype(str).tolist()
	results = []

	for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
		tokens = clean_doc(doc)
		if output == "clean_text":
			results.append(" ".join(tokens))
		elif output == "tokens":
			results.append(tokens)
		else:
			raise ValueError("El parámetro 'output' debe ser 'clean_text' o 'tokens'")

	df[output] = results
	logging.info("✅ Preprocesamiento completado.")
	return df


# Ejemplo de uso
if __name__ == "__main__":
	import os

	DATA_PATH = "data/imdb_reviews.csv"
	COLUMN = "text"
	OUTPUT = "clean_text"
	NEW_DATA_PATH = "data/new_reviews.csv"

	# Ajustar los siguientes parametros segun la capacidad del ordenador
	BATCH = 2000
	PROCESS = 1

	df = pd.read_csv(DATA_PATH)
	df = preprocess_dataframe(df, text_column=COLUMN, output=OUTPUT, batch_size=BATCH, n_process=PROCESS)

	# Crear carpeta si no existe
	os.makedirs("data", exist_ok=True)

	# Guardar CSVs preprocesados
	OUTPUT_PATH = "data/preprocessed_imdb.csv"
	df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
	logging.info(f"✅ Dataset preprocesado guardado en {OUTPUT_PATH}")

	df_new = pd.read_csv(NEW_DATA_PATH)
	df_clean = preprocess_dataframe(df_new, text_column=COLUMN, output=OUTPUT, batch_size=BATCH, n_process=PROCESS)

	OUTPUT_PATH = "data/preprocessed_new_imdb.csv"
	df_clean.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
	logging.info(f"✅ Dataset preprocesado guardado en {OUTPUT_PATH}")

	# Mostrar preview
	print(df[[COLUMN, OUTPUT]].head())
	print(df_clean[[COLUMN, OUTPUT]].head())

