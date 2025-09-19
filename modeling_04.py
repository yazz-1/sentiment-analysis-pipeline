import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)
import matplotlib.pyplot as plt
import joblib

# -----------------------
# Configuración
# -----------------------

INPUT_FILE = "data/preprocessed_imdb.csv"
TEXT_COL = "clean_text"      # Cambia a "tokens" si estás usando ese campo
LABEL_COL = "label"
OUTPUT_DIR = "output"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# 1. Cargar datos
# -----------------------

df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=[TEXT_COL, LABEL_COL])

# -----------------------
# 2. División train/test
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COL], df[LABEL_COL], test_size=0.2, random_state=42, stratify=df[LABEL_COL]
)

# -----------------------
# 3. Convertir etiquetas a numéricas
# -----------------------

label_mapping = {"neg": 0, "pos": 1}
y_train = y_train.map(label_mapping)
y_test = y_test.map(label_mapping)

# -----------------------
# 4. Vectorización TF-IDF
# -----------------------

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------
# 5. Entrenamiento del modelo
# -----------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -----------------------
# 6. Evaluación estándar
# -----------------------

y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1)
rec = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print(f"\n📊 Evaluación del modelo:\n-------------------------")
print(f"Accuracy:  {acc:.4f}")
print(f"Precisión: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\n📄 Reporte detallado:")
print(classification_report(y_test, y_pred))

# -----------------------
# 7. Matriz de confusión
# -----------------------

cm = confusion_matrix(y_test, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["neg","pos"])
disp.plot(cmap='Blues', values_format=".2f")
plt.title("Matriz de Confusión Normalizada")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_normalized.png")
plt.close()

print(f"\n✅ Matriz de confusión guardada en {OUTPUT_DIR}/confusion_matrix_normalized.png")

# -----------------------
# 8. Curva ROC y AUC
# -----------------------

y_probs = model.predict_proba(X_test_tfidf)[:, 1]  # Probabilidad de clase positiva
positive_class = label_mapping["pos"]
fpr, tpr, thresholds = roc_curve(y_test, y_probs, pos_label=positive_class)
auc_score = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Curva ROC")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")
plt.close()

print(f"✅ Curva ROC y AUC guardadas en {OUTPUT_DIR}/roc_curve.png")
print(f"🎯 AUC Score: {auc_score:.4f}")

# -----------------------
# 9. Guardado del modelo, vectorizador y métricas
# -----------------------

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, f"{MODEL_DIR}/logistic_model.joblib")
joblib.dump(vectorizer, f"{MODEL_DIR}/tfidf_vectorizer.joblib")

metrics = {
	"Accuracy": acc,
	"Precisión": prec,
	"Recall": rec,
	"F1-score": f1,
	"AUC": auc_score
}

metrics_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_for_tableau.csv"), index=False)

print(f"\n✅ Modelo y vectorizador guardados en {MODEL_DIR}/")
print(f"\n✅ Métricas y curva ROC guardadas en {OUTPUT_DIR}/")
