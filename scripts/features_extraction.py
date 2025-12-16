import os, pathlib, numpy as np
from tensorflow.keras.applications import VGG16, VGG19, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import torch, torchvision.models as models, torchvision.transforms as T
from PIL import Image

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SAVED_DIR = ROOT_DIR / "data/saved"        # ✅ Répertoire des images prétraitées
FEATURES_DIR = ROOT_DIR / "data/features"  # ✅ Répertoire de sortie des features
FEATURES_DIR.mkdir(exist_ok=True, parents=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] PyTorch device : {device}")

# ------------------------------------------------------------------
# 1.  Modèles TensorFlow
# ------------------------------------------------------------------
TF_MODELS = {
    "VGG16":       VGG16(weights="imagenet", include_top=False, pooling="avg"),
    "InceptionV3": InceptionV3(weights="imagenet", include_top=False, pooling="avg"),
}

# ------------------------------------------------------------------
# 2.  AlexNet PyTorch 
# ------------------------------------------------------------------
alexnet = models.alexnet(weights="IMAGENET1K_V1").to(device).eval()
# on enlève le classifier → (batch, 4096)
alexnet_features = torch.nn.Sequential(*list(alexnet.children())[:-1])  
alexnet_preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# ------------------------------------------------------------------
# 3.  Extracteur UNIQUE
# ------------------------------------------------------------------
def extract(model_name: str, X: np.ndarray):
    """
    X : (N,H,W,3)  RGB  0-255  (uint8 ou float32)
    return : (N, feat_dim)  np.ndarray
    """
    if model_name == "AlexNet":
        feats = []
        for img in X:                       # img 0-255
            pil = Image.fromarray(img.astype('uint8'), 'RGB')
            tens = alexnet_preprocess(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                out = alexnet_features(tens)        # (1, 4096)
            feats.append(out.cpu().numpy().squeeze())
        return np.array(feats)

    # ---------- TensorFlow models ----------
    model = TF_MODELS[model_name]
    # normalisation spécifique au modèle
    if hasattr(model, "preprocess_input"):
        from tensorflow.keras.applications import preprocess_input
        X = preprocess_input(X.copy())
    else:
        X = X / 255.
    return model.predict(X, batch_size=32, verbose=1)

# ------------------------------------------------------------------
# 4.  Boucle sur datasets
# ------------------------------------------------------------------
def process_dataset(ds_name: str):
    # ✅ Charger depuis data/saved
    input_file = SAVED_DIR / f"{ds_name}_images.npy"
    if not input_file.exists():
        print(f"❌ {input_file} n'existe pas, skip")
        return
    
    X = np.load(input_file)
    print(f"📂 Chargé {input_file}  –  shape {X.shape}")
    
    for m_name in list(TF_MODELS.keys()) + ["AlexNet"]:
        # ✅ Sauvegarder dans data/features
        out_file = FEATURES_DIR / f"{ds_name}_{m_name}.npy"
        if out_file.exists():
            print(f"⏩  {out_file} already exists")
            continue
        feat = extract(m_name, X)
        np.save(out_file, feat)
        print(f"🔸  {out_file}  –  {feat.shape}")

if __name__ == "__main__":
    for ds in ["covid19_xrays", "DTD", "iris", "wildfire"]:
        process_dataset(ds)