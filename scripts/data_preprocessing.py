import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# =============================================
# Fonction de chargement des images 
# =============================================
def charger_images(base_path, target_size=(224, 224)):
    images = []
    labels = []
    class_names = []

    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            class_names.append(category)
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                try:
                    img = load_img(file_path, target_size=target_size)
                    img_array = img_to_array(img) / 255.0  # Normalisation
                    images.append(img_array)
                    labels.append(category)
                except Exception as e:
                    print(f"Erreur avec {file_path}: {e}")
    
    return np.array(images), np.array(labels), class_names

# Exemple pour un dossier
images_animals, labels_animals, classes_animals = charger_images("data/DTD")
images_flowers, labels_flowers, classes_flowers = charger_images("dataset/iris")
images_fire, labels_fire, classes_fire = charger_images("dataset/wildfire")
images_ct, labels_ct, classes_ct = charger_images("dataset/covid19_xrays")

# Fusionner tous les ensembles
X = np.concatenate([images_animals, images_flowers, images_fire, images_ct])
y = np.concatenate([labels_animals, labels_flowers, labels_fire, labels_ct])

# Encodage des labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Partitionnement
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
