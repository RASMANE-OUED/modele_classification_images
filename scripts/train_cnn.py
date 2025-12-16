# ============================================================
#  Fichier : train_cnn_models.py
#  Objectif : Entraîner des CNN légers sur les jeux d'images sauvegardés
# ============================================================

import numpy as np, pandas as pd, joblib, datetime, time, logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ------------------------------------------------------------------
# 1.  Définition des répertoires
# ------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJ_ROOT / "data/saved"
MODELS_DIR = PROJ_ROOT / "models/cnn_models"
RESULTS_DIR = PROJ_ROOT / "resultats/metrics"
LOGS_DIR = PROJ_ROOT / "logs"

for d in [SAVE_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------
# 2.  Configuration logging
# ------------------------------------------------------------------
logging.basicConfig(
    filename=LOGS_DIR / "cnn_training.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ------------------------------------------------------------------
# 3.  Hyper-paramètres
# ------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 15
INIT_LR = 1e-3
PATIENCE = 8
INPUT_SHAPE = (224, 224, 3)

# ------------------------------------------------------------------
# 4.  Architecture du CNN
# ------------------------------------------------------------------
def build_light_cnn(num_classes: int):
    """CNN léger et régularisé"""
    model = models.Sequential([
        layers.Conv2D(32, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal', input_shape=INPUT_SHAPE),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ------------------------------------------------------------------
# 5.  Entraînement d’un dataset
# ------------------------------------------------------------------
def train_dataset(ds_name: str):
    try:
        # ---------- Chargement ----------
        X = np.load(SAVE_DIR / f"{ds_name}_images.npy")
        y = np.load(SAVE_DIR / f"{ds_name}_labels.npy")

        le = LabelEncoder()
        y_int = le.fit_transform(y)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_int, test_size=0.15, stratify=y_int, random_state=42)

        X_train = X_train.astype("float32") / 255.0
        X_val = X_val.astype("float32") / 255.0

        # ---------- Modèle ----------
        model = build_light_cnn(num_classes=len(le.classes_))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        cb = [
            callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
        ]

        # ---------- Entraînement ----------
        logging.info(f"🚀 Training CNN for {ds_name} ...")
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=cb,
            verbose=2
        )

        # ---------- Évaluation ----------
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')

        logging.info(f"{ds_name} | acc={acc:.3f} | f1={f1:.3f} | classes={len(le.classes_)}")

        print(f"\n===== {ds_name} =====")
        print(classification_report(y_val, y_pred, target_names=le.classes_))

        # ---------- Sauvegarde ----------
        model_path = MODELS_DIR / f"{ds_name}_CNN.h5"
        encoder_path = MODELS_DIR / f"{ds_name}_CNN_labelenc.pkl"
        model.save(model_path)
        joblib.dump(le, encoder_path)

        print(f"💾  Model + encoder saved  ->  {model_path}")
        logging.info(f"Saved model -> {model_path}")

        # ---------- Résumé ----------
        return {
            "dataset": ds_name,
            "algorithm": "CNN_light",
            "accuracy": acc,
            "f1_weighted": f1,
            "final_epochs": len(history.history['loss']),
            "model_path": str(model_path),
            "timestamp": datetime.datetime.now(),
        }

    except Exception as e:
        logging.error(f"❌ Error training {ds_name}: {e}")
        print(f"❌ Error training {ds_name}: {e}")
        return None

# ------------------------------------------------------------------
# 6.  Boucle principale
# ------------------------------------------------------------------
if __name__ == "__main__":
    datasets = ["covid19_xrays", "iris", "wildfire", "DTD"]
    rows = []

    for ds in datasets:
        if (SAVE_DIR / f"{ds}_images.npy").exists():
            result = train_dataset(ds)
            if result:
                rows.append(result)
        else:
            print(f"⏭️  Skip {ds} – images not found")

    if rows:
        df = pd.DataFrame(rows)
        csv_path = RESULTS_DIR / "cnn_benchmark.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ CNN training done – summary : {csv_path}")
        logging.info(f"✅ CNN training completed – summary saved : {csv_path}")
    else:
        print("⚠️ Aucun dataset trouvé pour l’entraînement.")
