import os
import pandas as pd

DATA_DIR = "aclImdb"
REVIEWS_FILE = "data/imdb_reviews.csv"
NEW_REVIEWS_FILE = "data/new_reviews.csv"


def load_imdb_data(data_dir):
    """
    Carga las reseñas etiquetadas (pos/neg) de train y test.
    """
    data = []
    for split in ["train", "test"]:
        for label in ["pos", "neg"]:
            folder = os.path.join(data_dir, split, label)
            for fname in os.listdir(folder):
                if fname.endswith(".txt"):
                    with open(os.path.join(folder, fname), encoding="utf-8", errors="ignore") as f:
                        text = f.read().strip()
                    data.append({"text": text, "label": label, "split": split})
    return pd.DataFrame(data)


def load_unsup_reviews(data_dir):
    """
    Carga las reseñas no etiquetadas (unsup) del set de entrenamiento.
    """
    data = []
    folder = os.path.join(data_dir, "train", "unsup")
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            with open(os.path.join(folder, fname), encoding="utf-8") as f:
                text = f.read().strip()
            data.append({"text": text})
    return pd.DataFrame(data)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df_labeled = load_imdb_data(DATA_DIR)
    df_labeled["label_num"] = df_labeled["label"].map({"neg": 0, "pos": 1})
    df_labeled.to_csv(REVIEWS_FILE, index=False, encoding="utf-8")
    print(f"✅ Reseñas etiquetadas guardadas en {REVIEWS_FILE}")

    df_unsup = load_unsup_reviews(DATA_DIR)
    df_unsup.to_csv(NEW_REVIEWS_FILE, index=False, encoding="utf-8")
    print(f"✅ Reseñas no etiquetadas guardadas en {NEW_REVIEWS_FILE}")
