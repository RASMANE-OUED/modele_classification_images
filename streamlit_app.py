"""
Streamlit Application pour la Classification d'Images
Compatible avec la structure : scripts/ et models/
"""

import streamlit as st
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torchvision.models as models
import torchvision.transforms as T

# ==============================================================================
# Configuration
# ==============================================================================
st.set_page_config(
    page_title="Classification d'Images",
    page_icon="logos/logoi.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Répertoires
PROJ_ROOT = Path(__file__).resolve().parent
MODELS_CLASSIC = PROJ_ROOT / "models/classiques_models"
MODELS_CNN = PROJ_ROOT / "models/cnn_models"
SAVED_DIR = PROJ_ROOT / "data/saved"
RESULTS_DIR = PROJ_ROOT / "resultats/metrics"
PLOTS_DIR = PROJ_ROOT / "resultats/plots"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.image("logos/logoi.jpg")
# ==============================================================================
# Cache des modèles
# ==============================================================================
@st.cache_resource
def load_feature_extractor(extractor_name: str):
    """Charge un extracteur de features"""
    if extractor_name == "VGG16":
        return VGG16(weights="imagenet", include_top=False, pooling="avg")
    elif extractor_name == "InceptionV3":
        return InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    elif extractor_name == "AlexNet":
        alexnet = models.alexnet(weights="IMAGENET1K_V1").to(device).eval()
        alexnet_features = torch.nn.Sequential(*list(alexnet.children())[:-1])
        
        preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return alexnet_features, preprocess
    else:
        raise ValueError(f"Extracteur non supporté: {extractor_name}")

@st.cache_resource
def load_cnn_model(dataset: str):
    """Charge un modèle CNN + label encoder"""
    model_path = MODELS_CNN / f"{dataset}_CNN.h5"
    encoder_path = MODELS_CNN / f"{dataset}_CNN_labelenc.pkl"
    
    if not model_path.exists():
        return None, None
    
    cnn_model = load_model(model_path)
    label_encoder = joblib.load(encoder_path)
    
    return cnn_model, label_encoder

@st.cache_data
def load_benchmark_results():
    """Charge les résultats de benchmark"""
    classic_path = RESULTS_DIR / "benchmark_summary.csv"
    cnn_path = RESULTS_DIR / "cnn_benchmark.csv"
    
    df_classic = pd.read_csv(classic_path) if classic_path.exists() else None
    df_cnn = pd.read_csv(cnn_path) if cnn_path.exists() else None
    
    return df_classic, df_cnn

# ==============================================================================
# Fonctions utilitaires
# ==============================================================================
def extract_features(image: Image.Image, extractor_name: str) -> np.ndarray:
    """Extrait les features d'une image"""
    img = image.resize((224, 224)).convert('RGB')
    img_array = np.array(img)
    
    if extractor_name == "AlexNet":
        alexnet_model, preprocess = load_feature_extractor("AlexNet")
        pil_img = Image.fromarray(img_array.astype('uint8'), 'RGB')
        tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = alexnet_model(tensor)
        return features.cpu().numpy().squeeze()
    
    # TensorFlow models
    img_array = np.expand_dims(img_array, axis=0)
    
    if extractor_name == "VGG16":
        model = load_feature_extractor("VGG16")
        from tensorflow.keras.applications.vgg16 import preprocess_input
        img_array = preprocess_input(img_array)
    elif extractor_name == "InceptionV3":
        model = load_feature_extractor("InceptionV3")
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        img_array = preprocess_input(img_array)
    
    features = model.predict(img_array, verbose=0)
    return features.squeeze()

def get_available_models(dataset: str, feature: str):
    """Récupère les modèles classiques disponibles pour un dataset + feature"""
    available = []
    pattern = f"{dataset}_{feature}_*.pkl"
    
    for model_file in MODELS_CLASSIC.glob(pattern):
        if not model_file.name.endswith("_scaler.pkl"):
            # Extraire l'algorithme
            algo = model_file.stem.split("_")[-1]
            available.append(algo)
    
    return sorted(available)

def get_class_names(dataset: str):
    """Récupère les noms de classes d'un dataset"""
    labels_path = SAVED_DIR / f"{dataset}_labels.npy"
    if labels_path.exists():
        y = np.load(labels_path)
        return np.unique(y)
    return None

# ==============================================================================
# Interface utilisateur
# ==============================================================================
st.title("🎯 Classification d'Images - Modèles Classiques & CNN")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Sélection dataset
DATASETS = ["covid19_xrays", "DTD", "iris", "wildfire"]
dataset = st.sidebar.selectbox("Dataset", DATASETS, index=2)

# Sélection du type de modèle
model_type = st.sidebar.radio("Type de modèle", ["Classique", "CNN"])

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prédiction", " Performances", " Comparaison", "ℹ Info"])

# ==============================================================================
# TAB 1 : Prédiction
# ==============================================================================
with tab1:
    st.header("Classification d'image")
    
    uploaded_file = st.file_uploader(
        "📁 Téléversez une image",
        type=["jpg", "jpeg", "png"],
        help="Formats supportés: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Afficher l'image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Image téléversée", use_container_width=True)
        
        with col2:
            if model_type == "Classique":
                # Configuration modèle classique
                st.subheader("⚙️ Configuration")
                
                feature = st.selectbox(
                    "Extracteur de features",
                    ["VGG16", "InceptionV3", "AlexNet"]
                )
                
                available_algos = get_available_models(dataset, feature)
                
                if not available_algos:
                    st.warning(f"❌ Aucun modèle trouvé pour {dataset} + {feature}")
                    st.stop()
                
                algorithm = st.selectbox("Algorithme", available_algos)
                
                if st.button(" Classifier", type="primary"):
                    with st.spinner("Classification en cours..."):
                        try:
                            # Extraction features
                            features = extract_features(image, feature)
                            features = features.reshape(1, -1)
                            
                            # Charger modèle + scaler
                            model_path = MODELS_CLASSIC / f"{dataset}_{feature}_{algorithm}.pkl"
                            scaler_path = MODELS_CLASSIC / f"{dataset}_{feature}_{algorithm}_scaler.pkl"
                            
                            classifier = joblib.load(model_path)
                            scaler = joblib.load(scaler_path) if scaler_path.exists() else None
                            
                            if scaler:
                                features = scaler.transform(features)
                            
                            # Prédiction
                            prediction = classifier.predict(features)[0]
                            probabilities = classifier.predict_proba(features)[0]
                            
                            # Classes
                            classes = get_class_names(dataset)
                            if classes is None:
                                classes = classifier.classes_
                            
                            # Affichage résultats
                            st.success(f" Prédiction terminée !")
                            
                            st.metric(
                                "Classe prédite",
                                str(classes[int(prediction)] if isinstance(prediction, (int, np.integer)) else prediction),
                                f"Confiance: {probabilities.max():.1%}"
                            )
                            
                            # Probabilités
                            st.subheader(" Probabilités par classe")
                            
                            for i, cls in enumerate(classes):
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.progress(float(probabilities[i]))
                                with col_b:
                                    st.write(f"**{cls}**: {probabilities[i]:.1%}")
                            
                        except Exception as e:
                            st.error(f"❌ Erreur: {e}")
            
            else:  # CNN
                st.subheader(" Prédiction CNN")
                
                # Vérifier si le modèle CNN existe
                cnn_model, label_encoder = load_cnn_model(dataset)
                
                if cnn_model is None:
                    st.warning(f"❌ Modèle CNN non disponible pour {dataset}")
                    st.stop()
                
                if st.button(" Classifier avec CNN", type="primary"):
                    with st.spinner("Prédiction CNN en cours..."):
                        try:
                            # Prétraiter
                            img = image.resize((224, 224))
                            img_array = np.array(img) / 255.0
                            img_array = np.expand_dims(img_array, axis=0)
                            
                            # Prédiction
                            predictions = cnn_model.predict(img_array, verbose=0)
                            predicted_idx = np.argmax(predictions[0])
                            probabilities = predictions[0]
                            
                            classes = label_encoder.classes_
                            predicted_class = classes[predicted_idx]
                            
                            # Affichage
                            st.success(f" Prédiction CNN terminée !")
                            
                            st.metric(
                                "Classe prédite",
                                str(predicted_class),
                                f"Confiance: {probabilities[predicted_idx]:.1%}"
                            )
                            
                            # Probabilités
                            st.subheader(" Probabilités par classe")
                            
                            for i, cls in enumerate(classes):
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.progress(float(probabilities[i]))
                                with col_b:
                                    st.write(f"**{cls}**: {probabilities[i]:.1%}")
                        
                        except Exception as e:
                            st.error(f"❌ Erreur CNN: {e}")

# ==============================================================================
# TAB 2 : Performances
# ==============================================================================
with tab2:
    st.header(" Performances des Modèles")
    
    df_classic, df_cnn = load_benchmark_results()
    
    if model_type == "Classique" and df_classic is not None:
        # Filtrer par dataset
        df_filtered = df_classic[df_classic['dataset'] == dataset]
        
        if not df_filtered.empty:
            st.subheader(f"Résultats pour {dataset}")
            
            # Tableau
            display_df = df_filtered[['feature', 'algorithm', 'accuracy', 'f1_weighted', 'train_time_sec']].copy()
            display_df = display_df.sort_values('f1_weighted', ascending=False)
            
            st.dataframe(
                display_df.style.highlight_max(axis=0, subset=['accuracy', 'f1_weighted']),
                use_container_width=True
            )
            
            # Graphique
            st.subheader(" F1-score par configuration")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            pivot = df_filtered.pivot_table(
                index='algorithm',
                columns='feature',
                values='f1_weighted'
            )
            
            pivot.plot(kind='bar', ax=ax, colormap='viridis')
            ax.set_ylabel('F1-score')
            ax.set_xlabel('Algorithme')
            ax.set_title(f'Performance par algorithme - {dataset}')
            ax.legend(title='Feature Extractor')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
        else:
            st.info(f"Aucun résultat disponible pour {dataset}")
    
    elif model_type == "CNN" and df_cnn is not None:
        df_filtered = df_cnn[df_cnn['dataset'] == dataset]
        
        if not df_filtered.empty:
            st.subheader(f"Résultats CNN - {dataset}")
            
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("Accuracy", f"{df_filtered['accuracy'].values[0]:.3f}")
            with metrics_cols[1]:
                st.metric("F1-score", f"{df_filtered['f1_weighted'].values[0]:.3f}")
            with metrics_cols[2]:
                st.metric("Epochs", int(df_filtered['final_epochs'].values[0]))
        else:
            st.info(f"Aucun résultat CNN pour {dataset}")

# ==============================================================================
# TAB 3 : Comparaison
# ==============================================================================
with tab3:
    st.header(" Comparaison Globale")
    
    df_classic, df_cnn = load_benchmark_results()
    
    if df_classic is not None and df_cnn is not None:
        # Meilleur classique par dataset
        best_classic = df_classic.loc[df_classic.groupby('dataset')['f1_weighted'].idxmax()]
        
        # Comparaison
        comparison = pd.merge(
            best_classic[['dataset', 'algorithm', 'feature', 'f1_weighted', 'accuracy']],
            df_cnn[['dataset', 'f1_weighted', 'accuracy']],
            on='dataset',
            suffixes=('_classic', '_cnn')
        )
        
        st.subheader("🏆 Meilleur modèle par dataset")
        st.dataframe(comparison, use_container_width=True)
        
        # Graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(comparison))
        width = 0.35
        
        ax.bar(x - width/2, comparison['f1_weighted_classic'], width,
               label='Meilleur Classique', color='#3498db')
        ax.bar(x + width/2, comparison['f1_weighted_cnn'], width,
               label='CNN', color='#2ecc71')
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('F1-score')
        ax.set_title('Comparaison : Classiques vs CNN')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison['dataset'], rotation=20)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Données de benchmark non disponibles")

# ==============================================================================
# TAB 4 : Info
# ==============================================================================
with tab4:
    st.header("ℹ️ À propos")
    
    st.markdown("""
    ## 🎯 Application de Classification d'Images
    
    Cette application utilise des modèles de **Deep Learning** et de **Machine Learning classique**
    pour classifier des images sur 4 datasets différents.
    
    ### 📂 Datasets disponibles
    
    - **covid19_xrays** : Classification de radiographies COVID/Non-COVID
    - **DTD** : Classification de textures
    - **iris** : Classification de fleurs iris
    - **wildfire** : Détection d'incendies
    
    ### 🧠 Modèles
    
    #### Modèles Classiques :
    - **Extracteurs** : VGG16, InceptionV3, AlexNet
    - **Classificateurs** : SVM, KNN, Decision Tree, Gaussian Naive Bayes
    
    #### CNN :
    - Architecture légère custom avec BatchNorm et Dropout
    - Entraîné de bout en bout sur les images
    
    ### 📊 Pipeline
    
    1. **Prétraitement** : Redimensionnement 224×224
    2. **Extraction** : Features via CNN pré-entraîné (ou direct pour CNN)
    3. **Classification** : Prédiction avec modèle entraîné
    4. **Résultats** : Classe + probabilités
    
    ### 🛠️ Technologies
    
    - **Backend** : TensorFlow, PyTorch, scikit-learn
    - **Frontend** : Streamlit
    - **Visualisation** : Matplotlib, Seaborn
    """)
    
    # Statistiques
    st.subheader("📊 Statistiques du projet")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        classic_models = len(list(MODELS_CLASSIC.glob("*.pkl")))
        st.metric("Modèles Classiques", classic_models)
    
    with col2:
        cnn_models = len(list(MODELS_CNN.glob("*_CNN.h5")))
        st.metric("Modèles CNN", cnn_models)
    
    with col3:
        datasets_count = len(list(SAVED_DIR.glob("*_images.npy")))
        st.metric("Datasets", datasets_count)