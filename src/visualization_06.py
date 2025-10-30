import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import os


def make_plots(df, output_dir):

	# Gráfico de predicciones
	plt.figure(figsize=(6, 5))
	sns.countplot(x="prediction", hue="prediction", data=df, palette="pastel", legend=False)
	plt.title("Distribución de predicciones")
	plt.xlabel("Etiqueta predicha")
	plt.ylabel("Número de textos")
	plt.savefig(os.path.join(output_dir, "prediction_distribution.png"))
	plt.close()

	# Histograma de probabilidades
	plt.figure(figsize=(6, 5))
	sns.histplot(df["prob_positive"], bins=10, kde=True, color="skyblue")
	plt.title("Confianza de predicciones positivas")
	plt.xlabel("Probabilidad de clase positiva")
	plt.ylabel("Frecuencia")
	plt.savefig(os.path.join(output_dir, "probability_histogram.png"))
	plt.close()

	# Nubes de palabras
def generate_wordcloud(texts, output_path, title=""):
	text_combined = " ".join(texts.astype(str))
	wordcloud = WordCloud(width=800, height=400, background_color="white", random_state=42).generate(text_combined)
	plt.figure(figsize=(10, 5))
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.title(title)
	plt.savefig(output_path)
	plt.close()

def export_tableau(df, output_dir):
	# Exportación para Tableau

	# A) Exportar datos individuales para Tableau
	cols = [c for c in ["text", "clean_text", "prediction", "prob_positive"] if c in df.columns]
	df[cols].to_csv(os.path.join(output_dir, "for_tableau.csv"), index=False)

	# B) Exportar frecuencias de palabras para Tableau
	if "clean_text" in df.columns:
		word_freq = []
		from spacy.lang.en.stop_words import STOP_WORDS as stop_words

		# Positivas
		positive_words = [w for w in " ".join(df[df["prediction"] == "pos"]["clean_text"]).split() if w not in stop_words]
		pos_counts = Counter(positive_words).most_common(50)
		for word, freq in pos_counts:
			word_freq.append({"word": word, "frequency": freq, "sentiment": "positive"})

		# Negativas
		negative_words = [w for w in " ".join(df[df["prediction"] == "neg"]["clean_text"]).split() if w not in stop_words]
		neg_counts = Counter(negative_words).most_common(50)
		for word, freq in neg_counts:
			word_freq.append({"word": word, "frequency": freq, "sentiment": "negative"})

		word_freq_df = pd.DataFrame(word_freq)
		word_freq_df.to_csv(os.path.join(output_dir, "word_freq_for_tableau.csv"), index=False)

	print("✅ Visualizaciones completadas y CSVs para Tableau guardados en 'output/'.")

if __name__ == '__main__':
	sns.set(style="whitegrid")
	RESULTS = "output/inference_results.csv"
	OUTPUT_DIR = "output/"
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	df = pd.read_csv(RESULTS)

	make_plots(df, OUTPUT_DIR)

	if "clean_text" in df.columns:
		# Textos positivos
		generate_wordcloud(
			df[df["prediction"] == "pos"]["clean_text"],
			os.path.join(OUTPUT_DIR, "wordcloud_positive.png"),
			title="Nube de palabras - Positivas"
		)

		# Textos negativos
		generate_wordcloud(
			df[df["prediction"] == "neg"]["clean_text"],
			os.path.join(OUTPUT_DIR, "wordcloud_negative.png"),
			title="Nube de palabras - Negativas"
		)

	export_tableau(df, OUTPUT_DIR)
