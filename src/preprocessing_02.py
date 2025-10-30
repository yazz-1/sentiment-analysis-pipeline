import pandas as pd
import logging
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stop_words

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def clean_doc(doc):
    """
    Procesa un documento spaCy y devuelve tokens lematizados sin stopwords ni puntuación.
    """
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and token.lemma_.lower() not in stop_words
    ]

def preprocess_dataframe(df, text_column="text", output="clean_text", batch_size=500, n_process=1):
    """
    Versión optimizada para usar menos memoria.
    """
    logging.info(f"Preprocesando '{text_column}' (batch_size={batch_size}, n_process={n_process})")

    # Cargar el modelo de spaCy con optimizaciones
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
    except OSError:
        logging.error("Modelo de spaCy no encontrado. Ejecuta: python -m spacy download en_core_web_sm")
        raise

    # Limpiar memoria antes de empezar
    import gc
    gc.collect()

    df = df.dropna(subset=[text_column]).copy()
    df[text_column] = df[text_column].fillna("")

    texts = df[text_column].astype(str).tolist()
    results = []

    # Procesar en chunks más pequeños
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        logging.info(f"Procesando lote {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        for doc in nlp.pipe(batch_texts, batch_size=100, n_process=n_process):
            tokens = clean_doc(doc)
            if output == "clean_text":
                results.append(" ".join(tokens))
            elif output == "tokens":
                results.append(tokens)

        # Limpiar memoria después de cada lote
        gc.collect()

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

    # Parámetros optimizados para evitar OOM
    BATCH = 100
    PROCESS = 1   # Usar solo 1 proceso

    # Procesar primer archivo
    df = pd.read_csv(DATA_PATH)
    df = preprocess_dataframe(df, text_column=COLUMN, output=OUTPUT, batch_size=BATCH, n_process=PROCESS)

    # Guardar CSVs preprocesados
    OUTPUT_PATH = "data/preprocessed_imdb.csv"
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    logging.info(f"✅ Dataset preprocesado guardado en {OUTPUT_PATH}")

    # Procesar el segundo archivo
    if os.path.exists(NEW_DATA_PATH):
        df_new = pd.read_csv(NEW_DATA_PATH)
        df_clean = preprocess_dataframe(df_new, text_column=COLUMN, output=OUTPUT, batch_size=BATCH, n_process=PROCESS)

        OUTPUT_PATH_NEW = "data/preprocessed_new_imdb.csv"
        df_clean.to_csv(OUTPUT_PATH_NEW, index=False, encoding="utf-8")
        logging.info(f"✅ Dataset preprocesado guardado en {OUTPUT_PATH_NEW}")

    # Mostrar preview
    print(df[[COLUMN, OUTPUT]].head())
    if 'df_clean' in locals():
        print(df_clean[[COLUMN, OUTPUT]].head())
