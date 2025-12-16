# evaluate_models.py
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split  # ✅ Import ajouté
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, roc_auc_score, classification_report
)
from sklearn.preprocessing import label_binarize

PROJ_ROOT=Path(__file__).resolve().parent.parent 
FEATURES_DIR = Path("data/features")  # ✅ Bon chemin
SAVED_DIR = Path("data/saved")        # ✅ Pour les labels
MODELS_DIR = PROJ_ROOT / "resultats/plots"
SUMMARY = PROJ_ROOT / "resultats/metrics/benchmark_summary.csv"

def evaluate_model(model_path, dataset, feature, algo):
    print(f"\n🔍 Évaluation : {model_path.name}")

    # ✅ Charger modèle + scaler si existe
    model = joblib.load(model_path)
    scaler = None
    scaler_path = model_path.parent / f"{model_path.stem}_scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print(f"📐 Scaler chargé depuis {scaler_path.name}")

    # ✅ Charger features et labels
    X = np.load(FEATURES_DIR / f"{dataset}_{feature}.npy")
    y = np.load(SAVED_DIR / f"{dataset}_labels.npy")

    # ✅ Même split que l'entraînement
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

    # ✅ Appliquer le scaler SEULEMENT sur test (fit déjà fait sur train)
    if scaler is not None:
        X_test = scaler.transform(X_test)

    # --- Prédictions ---
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # --- Matrice de confusion ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(y_test))
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix\n{dataset} – {feature} – {algo}")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / f"{dataset}_{feature}_{algo}_cm.png", dpi=100)
    plt.close()

    # --- ROC / AUC (si binaire ou multi-class) ---
    try:
        classes = np.unique(y_test)
        y_bin = label_binarize(y_test, classes=classes)
        
        if len(classes) == 2:  # ✅ Binaire
            if hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
            elif hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                raise AttributeError("Pas de decision_function ni predict_proba")
            
            fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=classes[1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}", linewidth=2)
            plt.plot([0,1],[0,1],'--',color='gray', label='Chance')
            plt.xlabel("Taux de faux positifs")
            plt.ylabel("Taux de vrais positifs")
            plt.title(f"Courbe ROC\n{dataset} – {feature} – {algo}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(MODELS_DIR / f"{dataset}_{feature}_{algo}_roc.png", dpi=100)
            plt.close()
            
        elif len(classes) > 2:  # ✅ Multi-class
            if not hasattr(model, "predict_proba"):
                print("⚠️  Modèle sans predict_proba, skip ROC multi-class")
                return
            
            y_score = model.predict_proba(X_test)
            
            plt.figure(figsize=(8, 6))
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
            
            plt.plot([0,1],[0,1],'--',color='gray', label='Chance')
            plt.xlabel("Taux de faux positifs")
            plt.ylabel("Taux de vrais positifs")
            plt.title(f"Courbe ROC (One-vs-Rest)\n{dataset} – {feature} – {algo}")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(MODELS_DIR / f"{dataset}_{feature}_{algo}_roc.png", dpi=100)
            plt.close()
            
        print(f"📊 Graphiques sauvegardés dans {MODELS_DIR}/")
        
    except Exception as e:
        print(f"⚠️  Impossible de tracer ROC : {e}")

# ------------------------------------------------------------------
# Boucle globale
# ------------------------------------------------------------------
if __name__ == "__main__":
    if not SUMMARY.exists():
        print(f"❌ {SUMMARY} n'existe pas. Lance d'abord train_models.py")
        exit(1)
    
    df = pd.read_csv(SUMMARY)
    print(f"📋 {len(df)} modèles à évaluer\n")
    
    for _, row in df.iterrows():
        model_path = Path(row["saved_model"])
        if not model_path.exists():
            print(f"⚠️  Modèle introuvable : {model_path}")
            continue
        
        dataset = row["dataset"]
        feature = row["feature"]
        algo = row["algorithm"]
        
        try:
            evaluate_model(model_path, dataset, feature, algo)
        except Exception as e:
            print(f"❌ Erreur lors de l'évaluation de {model_path.name}: {e}")
    
    print("\n✅ Évaluation terminée !")