import os
import pandas as pd
from sklearn.pipeline import Pipeline
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


def modeling(df, text_col, label_col):

    # Cargar datos

    df = df.dropna(subset=[text_col, label_col])

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col], test_size=0.2, random_state=42, stratify=df[LABEL_COL]
    )

    # Pipeline de vectorizador TF-IDF y modelo de Regresión Logística
    pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
            ("clf", LogisticRegression(max_iter=1000))
        ])

    # Entrenamiento del Pipeline
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test

def evaluate(pipeline, X_test, y_test, output_dir):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Evaluación estándar
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='pos')
    rec = recall_score(y_test, y_pred, pos_label='pos')
    f1 = f1_score(y_test, y_pred, pos_label='pos')

    print(f"\n📊 Evaluación del modelo:\n-------------------------")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precisión: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\n📄 Reporte detallado:")
    print(classification_report(y_test, y_pred))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["neg","pos"])
    disp.plot(cmap='Blues', values_format=".2f")
    plt.title("Matriz de Confusión Normalizada")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_normalized.png")
    plt.close()

    print(f"\n✅ Matriz de confusión guardada en {output_dir}/confusion_matrix_normalized.png")

    # Curva ROC y AUC
    y_probs = pipeline.predict_proba(X_test)[:, 1]  # Probabilidad de clase positiva
    #positive_class = label_mapping["pos"]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs, pos_label='pos')
    auc_score = roc_auc_score(y_test, y_probs)

    metrics = {
        "Accuracy": acc,
        "Precisión": prec,
        "Recall": rec,
        "F1-score": f1,
        "AUC": auc_score
    }

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

    print(f"✅ Curva ROC y AUC guardadas en {output_dir}/roc_curve.png")
    print(f"🎯 AUC Score: {auc_score:.4f}")
    return metrics

def save_results(pipeline, metrics, output_dir, model_dir):
    # Guardado del modelo, vectorizador y métricas
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(pipeline, f"{model_dir}/pipeline.joblib")


    metrics_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
    metrics_df.to_csv(os.path.join(output_dir, "metrics_for_tableau.csv"), index=False)

    print(f"\n✅ Modelo y vectorizador guardados en {model_dir}/")
    print(f"\n✅ Métricas y curva ROC guardadas en {output_dir}/")

if __name__ == '__main__':
    INPUT_FILE = "data/preprocessed_imdb.csv"
    TEXT_COL = "clean_text"
    LABEL_COL = "label"
    OUTPUT_DIR = "output"
    MODEL_DIR = "models"

    df = pd.read_csv(INPUT_FILE)

    pipeline, X_test, y_test = modeling(df, TEXT_COL, LABEL_COL)
    metrics = evaluate(pipeline, X_test, y_test, OUTPUT_DIR)
    save_results(pipeline, metrics, OUTPUT_DIR, MODEL_DIR)
