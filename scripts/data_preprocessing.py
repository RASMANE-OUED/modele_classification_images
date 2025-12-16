import os, pathlib, numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

TARGET_SIZE = (224, 224)          # keep 224 for every CNN
SAVE_DIR    = pathlib.Path("data/saved")
SAVE_DIR.mkdir(exist_ok=True)

# =========================================================================
# Fonction pour le chargement, le redimensionnement et la normalisation 
# =========================================================================
def build_dataset(data_root: pathlib.Path):
    
    images, labels = [], []
    for cls in sorted(d.name for d in data_root.iterdir() if d.is_dir()):
        for file in (data_root/cls).glob("*"):
            try:
                img = load_img(file, target_size=TARGET_SIZE)
                images.append(img_to_array(img))   # 0-255
                labels.append(cls)
            except Exception as e:
                print(f"[WARN] skipped {file}  –  {e}")
    return np.array(images), np.array(labels)

def save_dataset(name: str, X, y):
    np.save(SAVE_DIR/f"{name}_images.npy", X)
    np.save(SAVE_DIR/f"{name}_labels.npy", y)
    print(f"✅ {name}  –  images {X.shape}  labels {y.shape}")

if __name__ == "__main__":
    raw = pathlib.Path("data/images_brutes")
    for ds in raw.iterdir():
        if ds.is_dir():
            X, y = build_dataset(ds)
            save_dataset(ds.name, X, y)