# ============================================================
# Fichier des modèles classiques 
# =============================================================

from pathlib import Path
import logging
import numpy as np, pandas as pd, joblib, datetime, time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# === Dossiers ===
PROJ_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJ_ROOT / "data/features"
SAVED_DIR = PROJ_ROOT / "data/saved"
MODELS_DIR = PROJ_ROOT / "models/classiques_models"
RESULTS_DIR = PROJ_ROOT / "resultats/metrics"
for d in [FEATURES_DIR, SAVED_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True, parents=True)

# === Logging ===
LOGS_DIR = PROJ_ROOT / "resultats/logs"
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(filename=LOGS_DIR / "train_classiques.log",
                    level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

# ------------------------------------------------------------------
# Configuration grid-search par algorithme
# ------------------------------------------------------------------
GRIDS = {
    "SVM": {
        "clf": SVC(probability=True, random_state=42),  # ✅ probability=True pour ROC
        "param_grid": {"C": [0.01, 0.1, 1, 10, 100],
                       "kernel": ["linear", "rbf"],
                       "gamma": ["scale", 0.01, 0.001]},
    },
    "KNN": {
        "clf": KNeighborsClassifier(),
        "param_grid": {"n_neighbors": [3, 5, 7, 9, 11],
                       "weights": ["uniform", "distance"],
                       "metric": ["euclidean", "manhattan"]},
    },
    "DecisionTree": {
        "clf": DecisionTreeClassifier(random_state=42),
        "param_grid": {"criterion": ["gini", "entropy"],
                       "max_depth": [None, 5, 10, 20],
                       "min_samples_split": [2, 5, 10]},
    },
    "GaussianNB": {
        "clf": GaussianNB(),
        "param_grid": {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
    },
}

# === Fonction d’entraînement ===
def train_and_save(dataset, feature, algo):
    X = np.load(FEATURES_DIR / f"{dataset}_{feature}.npy")
    y = np.load(SAVED_DIR / f"{dataset}_labels.npy")
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    scaler = None
    if algo in ("SVM", "KNN", "GaussianNB"):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    start = time.time()
    grid = GridSearchCV(GRIDS[algo]["clf"], GRIDS[algo]["param_grid"], cv=3, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    model_name = f"{dataset}_{feature}_{algo}.pkl"
    joblib.dump(grid.best_estimator_, MODELS_DIR / model_name)
    if scaler is not None:
        joblib.dump(scaler, MODELS_DIR / f"{dataset}_{feature}_{algo}_scaler.pkl")

    logging.info(f"{dataset}-{feature}-{algo} | acc={acc:.3f} | f1={f1:.3f} | recall={recall:.3f}")
    return {...}  # idem ton dict

# === Main ===
if __name__ == "__main__":
    ...
