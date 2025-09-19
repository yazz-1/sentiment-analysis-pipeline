import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import os
import ast

# Configuración
INPUT_FILE = "data/preprocessed_imdb.csv"
OUTPUT_DIR = "output"
TEXT_COL = "clean_text"  # o "tokens"
LABEL_COL = "label"

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Leer el archivo
df = pd.read_csv(INPUT_FILE)

# Si la columna es tokens en string (listas), convertirlas
if TEXT_COL == "tokens" and isinstance(df["tokens"].iloc[0], str):
    df["tokens"] = df["tokens"].apply(ast.literal_eval)

print("\n📄 Dataset cargado:")
print(df.info())
print(df.head())

# =====================
# 1. Distribución de clases
# =====================
if LABEL_COL in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=LABEL_COL)
    plt.title("Distribución de clases")
    plt.savefig(f"{OUTPUT_DIR}/class_distribution.png")
    plt.close()
else:
    print(f"⚠️ Columna {LABEL_COL} no encontrada en dataset")

# =====================
# 2. Histograma de longitud de reseñas
# =====================
if TEXT_COL == "clean_text":
    lengths = df[TEXT_COL].str.split().str.len()
elif TEXT_COL == "tokens":
    lengths = df[TEXT_COL].apply(len)

plt.figure(figsize=(8, 4))
sns.histplot(lengths, bins=30, kde=True)
plt.title("Distribución de longitud de reseñas")
plt.xlabel("Número de palabras")
plt.ylabel("Frecuencia")
plt.savefig(f"{OUTPUT_DIR}/length_distribution.png")
plt.close()

# =====================
# 3. Nube de palabras (solo con clean_text)
# =====================
if TEXT_COL == "clean_text":
    full_text = " ".join(df[TEXT_COL].dropna())
    wordcloud = WordCloud(width=1000, height=400, background_color="white", random_state=42).generate(full_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Nube de Palabras")
    plt.savefig(f"{OUTPUT_DIR}/wordcloud.png")
    plt.close()

# =====================
# 4. Palabras más comunes (solo con tokens)
# =====================
if TEXT_COL == "tokens" and isinstance(df[TEXT_COL].iloc[0], list):
    all_tokens = df[TEXT_COL].explode()
    top_words = all_tokens.value_counts().head(20)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_words.values, y=top_words.index, palette="viridis")
    plt.title("Top 20 palabras más frecuentes")
    plt.xlabel("Frecuencia")
    plt.savefig(f"{OUTPUT_DIR}/top_words.png")
    plt.close()

print("\n✅ Análisis EDA completado. Visualizaciones guardadas en /output/")
