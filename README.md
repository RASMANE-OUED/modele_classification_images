<<<<<<< HEAD
#  Guide de  d'xécution du projet de classification d'images avec
#les modèles traditionnels vs le réseau de neurones convolutionnel

##  Structure du Projet
=======
#   GUIDE D'EXECUTION DU PROJET DE CLASSIFICATION D'IMAGES AVEC LES MODELES TRADITIONNELS VS CNN

##   STRUCTURE DU PROJET
>>>>>>> 27542958576a3ec2e123c159b0bc7bc0c8166102

```
modele_classification_images/
│
├── scripts/
│   ├── data_preprocessing.py
│   ├── features_extraction.py
│   ├── train_classiques.py
│   └── train_cnn.py
│
├── models/
│   ├── classiques_models/
│   │   ├── iris_VGG16_SVM.pkl
│   │   ├── iris_VGG16_SVM_scaler.pkl
│   │   └── ...
│   └── cnn_models/
│       ├── iris_CNN.h5
│       ├── iris_CNN_labelenc.pkl
│       └── ...
│
├── data/
│   ├── images_brutes/
│   ├── saved/
│   └── features/
│
├── resultats/
│   ├── metrics/
│   └── plots/
│
├── api.py              # FastAPI
├── streamlit_app.py    # Streamlit
└── requirements.txt
└── readme.md

---

##  Installation

### 1. Créer un environnement virtuel

```bash
python -m venv venv

# Activer l'environnement
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 3. Exécution des scripts
## 3.1 fichier de prétraitement
python scritps/data_preprocessing.py

## 3.2 fichier d'extraction des caractéristique
python scripts/features_extraction.py

## 3.3 fichier d'entrainement des modèles classiques ou traditionnel
python scripts/train_classiques.py

## 3.4 fichier d'entrainement du réseaux de neurones convolutionnel
python scripts/train_cnn.py 


## 4. Lancement de l'api FastApi (optionnel)
uvicorn api:app --reload

## 5. Lancement de l'interface graphique streamlit
<<<<<<< HEAD
streamlit run streamlit_app.py 
=======
streamlit run streamlit_app.py 
>>>>>>> 27542958576a3ec2e123c159b0bc7bc0c8166102
