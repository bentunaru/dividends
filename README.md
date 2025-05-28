# 📊 Analyse des Actions à Dividendes avec Machine Learning

## 🎯 Description

Ce projet utilise le machine learning (XGBoost et Random Forest) pour analyser et prédire les rendements d'actions à dividendes du S&P 500. Il combine l'analyse technique, fondamentale et historique pour générer des recommandations d'investissement basées sur des données objectives.

## 🚀 Fonctionnalités Principales

- **Collecte de données** : Récupération automatique des données historiques via Yahoo Finance
- **Analyse des dividendes** : Calcul des métriques clés (rendement, croissance, stabilité)
- **Feature Engineering** : Création de 50+ indicateurs techniques et fondamentaux
- **Machine Learning** : Modèles XGBoost et Random Forest pour prédire les rendements futurs
- **Système de recommandation** : Score composite basé sur plusieurs critères
- **Visualisations** : 20+ graphiques interactifs pour l'analyse
- **Gestion des checkpoints** : Sauvegarde automatique des résultats

## 📋 Prérequis

- Python 3.8+
- Conda (recommandé) ou pip

## 🛠️ Installation

### 1. Cloner le repository
```bash
git clone git@github.com:bentunaru/dividends.git
cd dividends
```

### 2. Créer l'environnement conda
```bash
conda create -n dividends python=3.13
conda activate dividends
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

Si le fichier `requirements.txt` n'existe pas, installez manuellement :
```bash
pip install pandas numpy matplotlib seaborn yfinance scikit-learn xgboost optuna jupyter plotly
```

## 📈 Utilisation

### 1. Lancer Jupyter Notebook
```bash
jupyter notebook stocks.ipynb
```

### 2. Exécuter les cellules dans l'ordre

Le notebook est structuré en sections :
1. **Configuration et imports**
2. **Collecte des données** (15 actions du S&P 500)
3. **Analyse exploratoire** (EDA)
4. **Feature Engineering**
5. **Modélisation ML** (XGBoost + Random Forest)
6. **Système de recommandation**
7. **Visualisations finales**

### 3. Résultats

Le système génère :
- **Top 3 actions recommandées** avec allocations suggérées
- **Prédictions de rendement à 12 mois**
- **Score composite** (0-100) pour chaque action
- **Visualisations détaillées** (8 graphiques principaux)

## 📊 Métriques Analysées

### Indicateurs Techniques
- RSI (14 jours)
- MACD et Signal
- Bandes de Bollinger
- Moyennes mobiles (20, 50, 200 jours)
- Volume moyen
- Volatilité

### Métriques de Dividendes
- Rendement du dividende
- Croissance du dividende (5 ans)
- Ratio de distribution
- Stabilité des paiements
- Historique de versement

### Indicateurs de Performance
- Rendement total (10 ans)
- Ratio de Sharpe
- Drawdown maximum
- Alpha et Beta
- Corrélation avec le marché

## 🎯 Système de Score Composite

Le score composite (0-100) est calculé avec les pondérations suivantes :
- **25%** : Prédiction ML du rendement futur
- **20%** : Performance historique
- **15%** : Rendement du dividende
- **15%** : Croissance du dividende
- **10%** : Ratio risque/rendement (Sharpe)
- **10%** : Faible volatilité
- **5%** : Faible drawdown

## 📁 Structure du Projet

```
dividends/
├── stocks.ipynb              # Notebook principal
├── README.md                 # Ce fichier
├── requirements.txt          # Dépendances Python
├── checkpoints/             # Sauvegardes automatiques
│   ├── initial_data_*.pkl
│   ├── after_eda_*.pkl
│   └── final_results_*.pkl
├── dividend_data_cache/     # Cache des données Yahoo Finance
│   ├── *_history_*.pkl
│   └── *_info_*.pkl
└── dividend_analysis_data.pkl  # Données consolidées
```

## 🔧 Configuration

### Actions analysées (modifiable dans le notebook)
```python
dividend_stocks = ['KO', 'PEP', 'JNJ', 'PG', 'XOM', 'CVX', 'ABBV', 
                  'MRK', 'VZ', 'T', 'IBM', 'CSCO', 'INTC', 'WMT', 'HD']
```

### Période d'analyse
- Par défaut : 10 ans d'historique
- Modifiable via le paramètre `period` dans `yf.download()`

## 📊 Exemples de Résultats

### Top 3 Recommandations (exemple)
```
1. HD (Home Depot) - Score: 85.2 - Allocation: 40%
2. WMT (Walmart) - Score: 82.7 - Allocation: 35%
3. JNJ (Johnson & Johnson) - Score: 79.3 - Allocation: 25%
```

### Métriques de Performance du Modèle
- R² Score : ~0.65-0.75
- RMSE : ~8-12%
- Feature Importance : RSI, rendement dividende, volatilité

## ⚠️ Avertissements

- **Risque d'investissement** : Les prédictions sont basées sur des données historiques. Les performances passées ne garantissent pas les résultats futurs.
- **Conseil professionnel** : Consultez un conseiller financier avant toute décision d'investissement.
- **Données en temps réel** : Les données sont mises à jour via Yahoo Finance. Vérifiez la fraîcheur des données.

## 🤝 Contribution

Les contributions sont bienvenues ! Pour contribuer :
1. Fork le projet
2. Créez une branche (`git checkout -b feature/amelioration`)
3. Committez vos changements (`git commit -am 'Ajout de fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Créez une Pull Request

## 📝 Licence

Ce projet est à des fins éducatives et de recherche. Utilisez à vos propres risques.

## 👤 Auteur

**Benjamin Tunaru**
- GitHub: [@bentunaru](https://github.com/bentunaru)

## 🙏 Remerciements

- Yahoo Finance pour l'API de données financières
- La communauté scikit-learn et XGBoost
- Tous les contributeurs open source

---

**Note** : Ce projet est en développement actif. Les fonctionnalités et performances peuvent évoluer. 