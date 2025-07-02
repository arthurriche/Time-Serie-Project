# Time-Serie-Project - Feature Selection et Analyse de Séries Temporelles

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

Ce projet implémente et compare diverses méthodes de sélection de features pour l'analyse de séries temporelles, avec un focus particulier sur la reconnaissance d'activités humaines à partir de données de smartphones. Le projet démontre l'expertise en feature engineering, sélection de variables et analyse de données temporelles.

## 🚀 Key Features

### Méthodes de Sélection de Features
- **Laplacian Score** - Sélection basée sur la similarité locale et la structure du graphe
- **Fisher Score** - Sélection basée sur la séparabilité inter/intra-classe
- **Mutual Information** - Sélection basée sur l'information mutuelle
- **Chi-Square Test** - Sélection basée sur les tests statistiques
- **Lasso Regression** - Sélection basée sur la régularisation L1
- **Low Variance Filter** - Filtrage basé sur la variance des features

### Analyse de Séries Temporelles
- **Human Activity Recognition** - Reconnaissance d'activités humaines
- **Smartphone Sensor Data** - Données de capteurs de smartphones
- **Time Series Clustering** - Clustering de séries temporelles
- **Feature Engineering** - Ingénierie de features temporelles

### Visualisation et Analyse
- **Comparative Analysis** - Comparaison des différentes méthodes
- **Performance Metrics** - Métriques de performance des modèles
- **Interactive Plots** - Visualisations interactives des résultats
- **Statistical Analysis** - Analyse statistique approfondie

## 📁 Project Structure

```
Time-Serie-Project/
├── clusters.py                   # Implémentation des méthodes de clustering
├── test.ipynb                    # Notebook de test et comparaison
├── dataset1.ipynb                # Analyse du dataset principal
├── Methods.ipynb                 # Implémentation des méthodes de sélection
├── graphs/                       # Visualisations et graphiques
│   ├── laplacian_graph.png       # Graphique Laplacian Score
│   ├── fisher_graph.png          # Graphique Fisher Score
│   ├── cmim_graph.png            # Graphique CMIM
│   └── lv_graph.png              # Graphique Low Variance
└── README.md                     # Documentation du projet
```

## 🛠️ Installation

### Prérequis
- Python 3.8+
- Scikit-learn 1.0+
- Pandas 1.3+
- NumPy
- Matplotlib
- SciPy

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/arthurriche/Time-Serie-Project.git
cd Time-Serie-Project

# Installer les dépendances
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install scipy
pip install jupyter
```

## 📈 Quick Start

### 1. Exécution des Tests

```bash
# Lancer le notebook de test
jupyter notebook test.ipynb

# Ou exécuter le script Python
python -c "import test; test.main()"
```

### 2. Analyse du Dataset

```python
# Charger et analyser le dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Charger les données
data_train = pd.read_csv('path/to/train.csv')
data_test = pd.read_csv('path/to/test.csv')

# Préprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

### 3. Application des Méthodes de Sélection

```python
from clusters import ranking_features_lap, fisher_score

# Laplacian Score
ranked_features = ranking_features_lap(X_scaled)

# Fisher Score
fisher_scores = fisher_score(X_scaled, y_train)
```

## 🧮 Technical Implementation

### Laplacian Score

```python
def laplacian_score(X):
    from sklearn.metrics.pairwise import rbf_kernel
    
    # Construction de la matrice d'affinité
    W = rbf_kernel(X)
    D = np.diag(W.sum(axis=1))
    L = D - W  # Matrice de Laplacien
    
    scores = []
    for i in range(X.shape[1]):
        f = X[:, i]
        scores.append((f.T @ L @ f) / (f.T @ D @ f))
    
    return np.array(scores)
```

### Fisher Score

```python
def fisher_score(X, y):
    classes = np.unique(y)
    scores = []
    
    for feature in X.T:
        inter_class = np.sum([np.mean(feature[y == c]) ** 2 for c in classes])
        intra_class = np.sum([np.var(feature[y == c]) for c in classes])
        scores.append(inter_class / intra_class)
    
    return np.array(scores)
```

### Mutual Information

```python
from sklearn.feature_selection import mutual_info_classif

def mutual_information(X, y):
    return mutual_info_classif(X, y)
```

## 📊 Performance Analysis

### Comparaison des Méthodes

| Méthode | Avantages | Inconvénients | Cas d'Usage |
|---------|-----------|---------------|-------------|
| **Laplacian Score** | Capture la structure locale | Sensible aux paramètres | Données avec structure géométrique |
| **Fisher Score** | Séparabilité des classes | Hypothèse gaussienne | Classification supervisée |
| **Mutual Information** | Non-linéaire, robuste | Calcul coûteux | Relations complexes |
| **Chi-Square** | Rapide, simple | Variables catégorielles | Pré-sélection |
| **Lasso** | Régularisation intégrée | Linéaire | Régression |
| **Low Variance** | Très rapide | Perte d'information | Pré-filtrage |

### Métriques d'Évaluation

- **Accuracy** - Précision de classification
- **Feature Importance** - Importance relative des features
- **Computational Time** - Temps de calcul
- **Stability** - Stabilité des sélections

## 🔬 Advanced Features

### Human Activity Recognition

Le projet utilise le dataset UCI HAR (Human Activity Recognition) avec smartphones :

- **6 Activités** : WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
- **561 Features** : Extraites des signaux temporels et fréquentiels
- **30 Sujets** : Données collectées sur 30 volontaires

### Feature Engineering

```python
# Extraction de features temporelles
def extract_temporal_features(signal):
    features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'max': np.max(signal),
        'min': np.min(signal),
        'range': np.max(signal) - np.min(signal),
        'energy': np.sum(signal**2),
        'entropy': -np.sum(signal * np.log2(signal + 1e-10))
    }
    return features
```

### Clustering Temporel

```python
def temporal_clustering(X, n_clusters=6):
    from sklearn.cluster import KMeans
    
    # Clustering K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    return clusters, kmeans
```

## 🚀 Applications

### Reconnaissance d'Activités
- **Fitness Tracking** - Suivi d'activités physiques
- **Healthcare Monitoring** - Surveillance médicale
- **Smart Home Systems** - Systèmes domotiques intelligents

### Analyse de Données Temporelles
- **Financial Time Series** - Séries temporelles financières
- **IoT Sensor Data** - Données de capteurs IoT
- **Biomedical Signals** - Signaux biomédicaux

### Feature Selection
- **Dimensionality Reduction** - Réduction de dimensionnalité
- **Model Interpretability** - Interprétabilité des modèles
- **Computational Efficiency** - Efficacité computationnelle

## 📚 Documentation Technique

### Méthodes de Sélection

#### 1. Laplacian Score
- **Principe** : Mesure la cohérence locale des features
- **Avantage** : Capture la structure géométrique des données
- **Formule** : `L(f) = (f^T L f) / (f^T D f)`

#### 2. Fisher Score
- **Principe** : Maximise la séparabilité inter-classe
- **Avantage** : Optimal pour la classification
- **Formule** : `F(f) = Σ(μ_i - μ)^2 / Σ σ_i^2`

#### 3. Mutual Information
- **Principe** : Mesure la dépendance non-linéaire
- **Avantage** : Capture les relations complexes
- **Formule** : `MI(X,Y) = Σ p(x,y) log(p(x,y)/(p(x)p(y)))`

### Hyperparamètres

- **K-Neighbors** - 5 (pour Laplacian Score)
- **RBF Kernel** - gamma=1.0 (pour la similarité)
- **Lasso Alpha** - 0.1 (pour la régularisation)
- **Variance Threshold** - 0.01 (pour le filtrage)

## 🤝 Contributing

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/amazing-feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## 📝 License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👨‍💻 Author

**Arthur Riche**
- LinkedIn: [Arthur Riche](https://www.linkedin.com/in/arthurriche/)
- Email: arthur.riche@example.com

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** pour le dataset HAR
- **Scikit-learn** pour les outils de machine learning
- **Communauté Open Source** pour les bibliothèques utilisées
- **Pairs de Recherche** pour les discussions et feedback

---

⭐ **Star ce repository si vous le trouvez utile !**

*Ce projet démontre l'expertise en feature selection et analyse de séries temporelles, avec des applications pratiques en reconnaissance d'activités humaines et traitement de données temporelles.* 