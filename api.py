"""
FastAPI Application pour la Classification d'Images
Modèles Classiques + CNN
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.models import load_model
import torch
import torchvision.models as models
import torchvision.transforms as T

# ==============================================================================
# Configuration
# ==============================================================================
app = FastAPI(
    title="Image Classification API",
    description="API de classification d'images avec modèles classiques et CNN",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Répertoires
PROJ_ROOT = Path(__file__).resolve().parent
MODELS_CLASSIC = PROJ_ROOT / "models/classiques_models"
MODELS_CNN = PROJ_ROOT / "models/cnn_models"
SAVED_DIR = PROJ_ROOT / "data/saved"

# Device PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Cache des modèles
# ==============================================================================
FEATURE_EXTRACTORS = {}
CLASSIFIERS_CACHE = {}
CNN_MODELS_CACHE = {}

# ==============================================================================
# Modèles Pydantic
# ==============================================================================
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    model_type: str

class HealthResponse(BaseModel):
    status: str
    models_available: Dict[str, int]

class ModelsListResponse(BaseModel):
    classic_models: List[str]
    cnn_models: List[str]
    feature_extractors: List[str]

# ==============================================================================
# Utilitaires - Extraction de Features
# ==============================================================================
def load_vgg16():
    """Charge VGG16 pour extraction"""
    if "VGG16" not in FEATURE_EXTRACTORS:
        FEATURE_EXTRACTORS["VGG16"] = VGG16(
            weights="imagenet", include_top=False, pooling="avg"
        )
    return FEATURE_EXTRACTORS["VGG16"]

def load_inception():
    """Charge InceptionV3 pour extraction"""
    if "InceptionV3" not in FEATURE_EXTRACTORS:
        FEATURE_EXTRACTORS["InceptionV3"] = InceptionV3(
            weights="imagenet", include_top=False, pooling="avg"
        )
    return FEATURE_EXTRACTORS["InceptionV3"]

def load_alexnet():
    """Charge AlexNet PyTorch pour extraction"""
    if "AlexNet" not in FEATURE_EXTRACTORS:
        alexnet = models.alexnet(weights="IMAGENET1K_V1").to(device).eval()
        FEATURE_EXTRACTORS["AlexNet"] = torch.nn.Sequential(*list(alexnet.children())[:-1])
        
        FEATURE_EXTRACTORS["AlexNet_preprocess"] = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return FEATURE_EXTRACTORS["AlexNet"], FEATURE_EXTRACTORS["AlexNet_preprocess"]

def extract_features(image: Image.Image, extractor_name: str) -> np.ndarray:
    """
    Extrait les features d'une image
    """
    # Redimensionner
    img = image.resize((224, 224)).convert('RGB')
    img_array = np.array(img)
    
    if extractor_name == "AlexNet":
        alexnet_model, preprocess = load_alexnet()
        pil_img = Image.fromarray(img_array.astype('uint8'), 'RGB')
        tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = alexnet_model(tensor)
        return features.cpu().numpy().squeeze()
    
    # TensorFlow models
    img_array = np.expand_dims(img_array, axis=0)
    
    if extractor_name == "VGG16":
        model = load_vgg16()
        from tensorflow.keras.applications.vgg16 import preprocess_input
        img_array = preprocess_input(img_array)
    elif extractor_name == "InceptionV3":
        model = load_inception()
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        img_array = preprocess_input(img_array)
    else:
        raise ValueError(f"Extracteur non supporté: {extractor_name}")
    
    features = model.predict(img_array, verbose=0)
    return features.squeeze()

# ==============================================================================
# Utilitaires - Chargement Modèles
# ==============================================================================
def load_classic_model(dataset: str, feature: str, algo: str):
    """Charge un modèle classique + scaler"""
    model_key = f"{dataset}_{feature}_{algo}"
    
    if model_key not in CLASSIFIERS_CACHE:
        model_path = MODELS_CLASSIC / f"{model_key}.pkl"
        scaler_path = MODELS_CLASSIC / f"{model_key}_scaler.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        classifier = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        
        CLASSIFIERS_CACHE[model_key] = {"model": classifier, "scaler": scaler}
    
    return CLASSIFIERS_CACHE[model_key]

def load_cnn_model(dataset: str):
    """Charge un modèle CNN + label encoder"""
    if dataset not in CNN_MODELS_CACHE:
        model_path = MODELS_CNN / f"{dataset}_CNN.h5"
        encoder_path = MODELS_CNN / f"{dataset}_CNN_labelenc.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle CNN non trouvé: {model_path}")
        
        cnn_model = load_model(model_path)
        label_encoder = joblib.load(encoder_path)
        
        CNN_MODELS_CACHE[dataset] = {
            "model": cnn_model,
            "encoder": label_encoder
        }
    
    return CNN_MODELS_CACHE[dataset]

def get_label_encoder(dataset: str):
    """Récupère le label encoder d'un dataset"""
    labels_path = SAVED_DIR / f"{dataset}_labels.npy"
    if labels_path.exists():
        y = np.load(labels_path)
        return np.unique(y)
    return None

# ==============================================================================
# Routes
# ==============================================================================
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check de l'API"""
    # Compter seulement les modèles (pas les scalers)
    classic_files = [f for f in MODELS_CLASSIC.glob("*.pkl") if not f.name.endswith("_scaler.pkl")]
    classic_count = len(classic_files)
    
    cnn_files = list(MODELS_CNN.glob("*_CNN.h5"))
    cnn_count = len(cnn_files)
    
    return {
        "status": "OK",
        "models_available": {
            "classic": classic_count // 2 if classic_count > 0 else 0,  # Diviser par 2 car modèle + scaler
            "cnn": cnn_count
        }
    }

@app.get("/models/list", response_model=ModelsListResponse)
async def list_models():
    """Liste tous les modèles disponibles"""
    # Modèles classiques (exclure les scalers)
    classic_files = [f for f in MODELS_CLASSIC.glob("*.pkl") if not f.name.endswith("_scaler.pkl")]
    classic_models = []
    
    for f in classic_files:
        # Format: dataset_feature_algo.pkl
        model_name = f.stem
        classic_models.append(model_name)
    
    # CNN models
    cnn_files = list(MODELS_CNN.glob("*_CNN.h5"))
    cnn_models = [f.stem.replace("_CNN", "") for f in cnn_files]
    
    return {
        "classic_models": sorted(classic_models),
        "cnn_models": sorted(cnn_models),
        "feature_extractors": ["VGG16", "InceptionV3", "AlexNet"]
    }

@app.post("/predict/classic", response_model=PredictionResponse)
async def predict_classic(
    file: UploadFile = File(...),
    dataset: str = "iris",
    feature: str = "VGG16",
    algorithm: str = "SVM"
):
    """
    Prédiction avec modèle classique
    
    Args:
        file: Image à classifier
        dataset: Dataset (covid19_xrays, DTD, iris, wildfire)
        feature: Extracteur (VGG16, InceptionV3, AlexNet)
        algorithm: Algorithme (SVM, KNN, DecisionTree, GaussianNB)
    """
    try:
        # Lire l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Extraire features
        features = extract_features(image, feature)
        features = features.reshape(1, -1)
        
        # Charger modèle + scaler
        model_data = load_classic_model(dataset, feature, algorithm)
        classifier = model_data["model"]
        scaler = model_data["scaler"]
        
        # Appliquer scaler si existe
        if scaler is not None:
            features = scaler.transform(features)
        
        # Prédiction
        prediction = classifier.predict(features)[0]
        probabilities = classifier.predict_proba(features)[0]
        
        # Récupérer les noms de classes
        classes = get_label_encoder(dataset)
        if classes is None:
            classes = classifier.classes_
        
        # Construire réponse
        proba_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
        predicted_class = str(classes[int(prediction)] if isinstance(prediction, (int, np.integer)) else prediction)
        confidence = float(probabilities.max())
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": proba_dict,
            "model_type": f"Classic_{algorithm}_{feature}"
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/predict/cnn", response_model=PredictionResponse)
async def predict_cnn(
    file: UploadFile = File(...),
    dataset: str = "iris"
):
    """
    Prédiction avec CNN
    
    Args:
        file: Image à classifier
        dataset: Dataset (covid19_xrays, DTD, iris, wildfire)
    """
    try:
        # Lire l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Prétraiter
        img = image.resize((224, 224)).convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Charger modèle CNN
        model_data = load_cnn_model(dataset)
        cnn_model = model_data["model"]
        label_encoder = model_data["encoder"]
        
        # Prédiction
        predictions = cnn_model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        probabilities = predictions[0]
        
        # Classes
        classes = label_encoder.classes_
        predicted_class = str(classes[predicted_idx])
        confidence = float(probabilities[predicted_idx])
        
        proba_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": proba_dict,
            "model_type": "CNN_light"
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur CNN: {str(e)}")

@app.get("/datasets/available")
async def get_datasets():
    """Liste les datasets disponibles"""
    datasets = []
    
    for label_file in SAVED_DIR.glob("*_labels.npy"):
        ds_name = label_file.stem.replace("_labels", "")
        img_file = SAVED_DIR / f"{ds_name}_images.npy"
        
        if img_file.exists():
            y = np.load(label_file)
            X = np.load(img_file)
            
            datasets.append({
                "name": ds_name,
                "n_samples": len(X),
                "n_classes": len(np.unique(y)),
                "classes": np.unique(y).tolist()
            })
    
    return {"datasets": datasets}

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)