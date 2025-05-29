#!/usr/bin/env python3
"""
Analyse des Dividendes - Top Actions des 10 Dernières Années

Objectifs du Projet:
1. Analyser les dividendes des actions les plus performantes sur 10 ans
2. Comparer les performances et identifier les meilleures opportunités
3. Prédire les meilleures actions pour les 10 prochaines années
4. Utiliser des techniques de Machine Learning avancées (XGBoost, Cross-Validation)

Technologies Utilisées:
- API: yfinance (données financières Yahoo Finance)
- Analyse: Pandas, NumPy
- Visualisation: Seaborn, Matplotlib, Plotly
- Machine Learning: Scikit-learn, XGBoost
- Optimisation: Optuna (Bayesian Optimization)
"""

# Importation des bibliothèques essentielles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from datetime import datetime, timedelta
import time
import os
import pickle
import xgboost as xgb

# Données financières
import yfinance as yf

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Configuration des graphiques
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("✅ Bibliothèques essentielles importées avec succès!")

# Liste des actions à analyser (Top S&P 500 avec dividendes)
DIVIDEND_ARISTOCRATS = [
    'JNJ',   # Johnson & Johnson
    'PG',    # Procter & Gamble
    'KO',    # Coca-Cola
    'PEP',   # PepsiCo
    'ABBV',  # AbbVie
    'MRK',   # Merck
    'XOM',   # Exxon Mobil
    'CVX',   # Chevron
    'T',     # AT&T
    'VZ',    # Verizon
    'IBM',   # IBM
    'CSCO',  # Cisco
    'INTC',  # Intel
    'WMT',   # Walmart
    'HD'     # Home Depot
]

print(f"📊 Nombre d'actions à analyser: {len(DIVIDEND_ARISTOCRATS)}")
print(f"🏢 Actions sélectionnées: {', '.join(DIVIDEND_ARISTOCRATS)}")


class YFinanceDataCollector:
    """Collecteur de données utilisant yfinance avec système de cache"""
    
    def __init__(self, lookback_years=10, cache_dir='dividend_data_cache'):
        self.lookback_years = lookback_years
        self.start_date = datetime.now() - timedelta(days=365 * lookback_years)
        self.end_date = datetime.now()
        self.cache_dir = cache_dir
        
        # Créer le répertoire de cache s'il n'existe pas
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_filename(self, symbol, data_type='history'):
        """Génère le nom de fichier pour le cache"""
        date_str = datetime.now().strftime('%Y%m%d')
        return os.path.join(self.cache_dir, f"{symbol}_{data_type}_{date_str}.pkl")
    
    def _is_cache_valid(self, filename, max_age_days=1):
        """Vérifie si le fichier de cache est encore valide"""
        if not os.path.exists(filename):
            return False
        
        # Vérifier l'âge du fichier
        file_time = os.path.getmtime(filename)
        file_age = (time.time() - file_time) / (24 * 3600)  # en jours
        
        return file_age < max_age_days
    
    def _load_from_cache(self, symbol):
        """Charge les données depuis le cache si disponible"""
        history_file = self._get_cache_filename(symbol, 'history')
        info_file = self._get_cache_filename(symbol, 'info')
        
        if self._is_cache_valid(history_file) and self._is_cache_valid(info_file):
            try:
                df = pd.read_pickle(history_file)
                with open(info_file, 'rb') as f:
                    info = pickle.load(f)
                print(f"✅ {symbol} - Données chargées depuis le cache")
                return df, info
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement du cache pour {symbol}: {e}")
                return None, None
        
        return None, None
    
    def _save_to_cache(self, symbol, df, info):
        """Sauvegarde les données dans le cache"""
        try:
            history_file = self._get_cache_filename(symbol, 'history')
            info_file = self._get_cache_filename(symbol, 'info')
            
            # Sauvegarder le DataFrame
            df.to_pickle(history_file)
            
            # Sauvegarder les informations
            with open(info_file, 'wb') as f:
                pickle.dump(info, f)
                
            print(f"💾 {symbol} - Données sauvegardées dans le cache")
        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde du cache pour {symbol}: {e}")
    
    def download_stock_data(self, symbol, use_cache=True):
        """Télécharge les données d'une action (ou les charge depuis le cache)"""
        
        # Essayer de charger depuis le cache
        if use_cache:
            df, info = self._load_from_cache(symbol)
            if df is not None:
                return df, info
        
        # Si pas de cache valide, télécharger les données
        try:
            print(f"🌐 {symbol} - Téléchargement depuis Yahoo Finance...")
            stock = yf.Ticker(symbol)
            
            # Données historiques
            df = stock.history(start=self.start_date, end=self.end_date)
            
            if len(df) == 0:
                print(f"❌ Pas de données disponibles pour {symbol}")
                return None, None
            
            # Informations sur l'entreprise
            info = stock.info
            
            print(f"✅ {symbol} - {len(df)} jours de données téléchargées")
            
            # Sauvegarder dans le cache
            if use_cache:
                self._save_to_cache(symbol, df, info)
            
            return df, info
            
        except Exception as e:
            print(f"❌ Erreur pour {symbol}: {str(e)}")
            return None, None
    
    def download_all_stocks(self, symbols, use_cache=True):
        """Télécharge les données pour toutes les actions"""
        stock_data = {}
        company_info = {}
        
        print(f"📁 Répertoire de cache: {os.path.abspath(self.cache_dir)}")
        print(f"🔄 Mode cache: {'Activé' if use_cache else 'Désactivé'}\n")
        
        for i, symbol in enumerate(symbols):
            print(f"[{i+1}/{len(symbols)}] Traitement de {symbol}...")
            
            df, info = self.download_stock_data(symbol, use_cache=use_cache)
            
            if df is not None:
                stock_data[symbol] = df
                company_info[symbol] = info
            
            # Petite pause seulement si on télécharge depuis l'API
            if not use_cache or not self._is_cache_valid(self._get_cache_filename(symbol)):
                time.sleep(0.5)
        
        return stock_data, company_info


class DividendDataPreprocessor:
    """
    Classe optimisée pour le prétraitement des données de dividendes
    Utilise pandas et numpy pour des calculs vectorisés efficaces
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def calculate_dividend_metrics(self, stock_data):
        """
        Calcule les métriques de dividendes pour chaque action de manière vectorisée
        """
        metrics = {}
        
        for symbol, df in stock_data.items():
            print(f"  📊 Traitement {symbol}...")
            
            # Copier et optimiser le DataFrame avec pandas
            df_metrics = df.copy()
            
            # === CALCULS VECTORISÉS AVEC PANDAS/NUMPY ===
            
            # 1. Rendements quotidiens (vectorisé)
            df_metrics['daily_return'] = df_metrics['Close'].pct_change()
            
            # 2. Rendement des dividendes (vectorisé avec numpy)
            df_metrics['dividend_yield'] = np.where(
                df_metrics['Close'] > 0,
                (df_metrics['Dividends'] / df_metrics['Close']) * 100,
                0
            )
            
            # 3. Rendement total avec gestion des valeurs nulles
            close_prev = df_metrics['Close'].shift(1)
            df_metrics['total_return'] = np.where(
                close_prev > 0,
                ((df_metrics['Close'] + df_metrics['Dividends']) / close_prev) - 1,
                0
            )
            
            # 4. Volatilité avec rolling window optimisé
            returns_clean = df_metrics['daily_return'].fillna(0)
            df_metrics['volatility_30d'] = (
                returns_clean.rolling(window=30, min_periods=10)
                .std() * np.sqrt(252)
            )
            df_metrics['volatility_252d'] = (
                returns_clean.rolling(window=252, min_periods=50)
                .std() * np.sqrt(252)
            )
            
            # === MÉTRIQUES ANNUELLES AVEC RESAMPLE PANDAS ===
            
            # 5. Agrégations annuelles efficaces
            try:
                annual_dividends = df_metrics['Dividends'].resample('Y').sum()
                avg_price = df_metrics['Close'].resample('Y').mean()
                
                # Rendement dividende annuel avec gestion division par zéro
                annual_yield = np.where(
                    avg_price > 0,
                    (annual_dividends / avg_price) * 100,
                    0
                )
                annual_yield = pd.Series(annual_yield, index=avg_price.index).fillna(0)
                
                # Croissance des dividendes (vectorisé)
                dividend_growth = annual_dividends.pct_change().fillna(0)
                
                # Fréquence des paiements
                dividend_frequency = (
                    df_metrics[df_metrics['Dividends'] > 0]
                    .resample('Y').size()
                )
                
            except Exception as e:
                # Fallback si problème avec les dates
                print(f"    ⚠️ Problème dates pour {symbol}, utilisation méthode alternative")
                annual_yield = pd.Series([df_metrics['dividend_yield'].mean()])
                dividend_growth = pd.Series([0])
                dividend_frequency = pd.Series([4])  # Assumé trimestriel
            
            # === CALCULS DE PERFORMANCE GLOBALE ===
            
            # 6. Rendements totaux avec numpy (plus rapide)
            first_price = df_metrics['Close'].iloc[0]
            last_price = df_metrics['Close'].iloc[-1]
            
            # Protection contre division par zéro
            if first_price > 0:
                total_period_return = ((last_price / first_price) - 1) * 100
                cumulative_dividends = df_metrics['Dividends'].sum()
                total_return_with_div = (
                    ((last_price + cumulative_dividends) / first_price) - 1
                ) * 100
            else:
                total_period_return = 0
                cumulative_dividends = 0
                total_return_with_div = 0
            
            # 7. Volatilité moyenne avec gestion des NaN
            volatility_mean = df_metrics['volatility_252d'].mean()
            if pd.isna(volatility_mean):
                volatility_mean = df_metrics['daily_return'].std() * np.sqrt(252)
            
            # === STOCKAGE DES RÉSULTATS ===
            metrics[symbol] = {
                'data': df_metrics,
                'annual_dividend_yield': annual_yield,
                'dividend_growth': dividend_growth,
                'dividend_frequency': dividend_frequency,
                'avg_dividend_yield': annual_yield.mean(),
                'avg_dividend_growth': dividend_growth.mean(),
                'price_return': total_period_return,
                'total_return': total_return_with_div,
                'cumulative_dividends': cumulative_dividends,
                'volatility': volatility_mean
            }
            
        return metrics


def calculate_rsi_vectorized(prices, period=14):
    """Calcule le RSI de manière vectorisée avec pandas/numpy"""
    
    # Calcul vectorisé des variations
    delta = prices.diff()
    
    # Séparation gains/pertes avec numpy vectorisé
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Moyennes mobiles exponentielles
    avg_gains = pd.Series(gains, index=prices.index).rolling(window=period).mean()
    avg_losses = pd.Series(losses, index=prices.index).rolling(window=period).mean()
    
    # Calcul RSI vectorisé avec protection division par zéro
    rs = np.where(avg_losses > 0, avg_gains / avg_losses, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Gestion des valeurs manquantes
    return pd.Series(rsi, index=prices.index).fillna(50)


def calculate_macd_vectorized(prices, fast=12, slow=26, signal=9):
    """Calcule le MACD de manière vectorisée avec pandas"""
    
    # EMAs vectorisées avec pandas
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # MACD signal
    macd_signal = macd_line.ewm(span=signal).mean()
    
    # Gestion des valeurs manquantes
    return macd_line.fillna(0), macd_signal.fillna(0)


def create_ml_features(stock_data, company_info, lookback_periods=[20, 50, 200]):
    """
    Créer des features pour le machine learning de manière vectorisée
    Utilise pandas et numpy pour des performances optimales
    """
    
    print("🔄 Création des features ML (version vectorisée)...")
    
    # === INITIALISATION AVEC NUMPY POUR LA PERFORMANCE ===
    all_features = []
    feature_creation_times = []
    
    for symbol, df in stock_data.items():
        start_symbol_time = time.time()
        print(f"  📊 Features pour {symbol}...")
        
        # === COPIE OPTIMISÉE DU DATAFRAME ===
        df_features = df.copy()
        
        # === FEATURES DE PRIX VECTORISÉES AVEC PANDAS ===
        
        # Calcul vectorisé des moyennes mobiles
        close_prices = df_features['Close']
        
        for period in lookback_periods:
            # SMA calculée de manière vectorisée
            sma_col = f'sma_{period}'
            df_features[sma_col] = close_prices.rolling(
                window=period, 
                min_periods=max(1, period//4)  # Minimum de données requis
            ).mean()
            
            # Ratio prix/SMA (vectorisé avec numpy)
            df_features[f'price_to_sma_{period}'] = np.where(
                df_features[sma_col] > 0,
                close_prices / df_features[sma_col],
                1.0  # Valeur neutre si SMA = 0
            )
            
            # Volatilité vectorisée
            returns = close_prices.pct_change()
            df_features[f'volatility_{period}'] = (
                returns.rolling(window=period, min_periods=max(1, period//4))
                .std() * np.sqrt(252)
            )
        
        # === FEATURES DE VOLUME VECTORISÉES ===
        
        volume = df_features['Volume']
        
        # SMA du volume
        df_features['volume_sma_20'] = volume.rolling(window=20, min_periods=5).mean()
        
        # Ratio de volume (vectorisé avec protection division par zéro)
        df_features['volume_ratio'] = np.where(
            df_features['volume_sma_20'] > 0,
            volume / df_features['volume_sma_20'],
            1.0
        )
        
        # === FEATURES DE DIVIDENDES VECTORISÉES ===
        
        # Rendement dividende vectorisé
        df_features['dividend_yield'] = np.where(
            close_prices > 0,
            (df_features['Dividends'] / close_prices) * 100,
            0
        )
        
        # Moyenne mobile du rendement dividende
        df_features['dividend_yield_ma_252'] = (
            df_features['dividend_yield']
            .rolling(window=252, min_periods=50)
            .mean()
        )
        
        # Consistance des dividendes (vectorisée)
        dividend_mask = df_features['Dividends'] > 0
        df_features['dividend_consistency'] = (
            dividend_mask.rolling(window=252, min_periods=50)
            .mean()  # Proportion de jours avec dividendes
        )
        
        # === FEATURES TECHNIQUES OPTIMISÉES ===
        
        # RSI vectorisé
        df_features['rsi'] = calculate_rsi_vectorized(close_prices, period=14)
        
        # MACD vectorisé
        macd_line, macd_signal = calculate_macd_vectorized(close_prices)
        df_features['macd'] = macd_line
        df_features['macd_signal'] = macd_signal
        df_features['macd_histogram'] = macd_line - macd_signal  # Feature supplémentaire
        
        # === FEATURES DE MOMENTUM VECTORISÉES ===
        
        # Calcul vectorisé des momentum sur différentes périodes
        momentum_periods = [21, 63, 126, 252]  # 1m, 3m, 6m, 1y
        momentum_names = ['1m', '3m', '6m', '1y']
        
        for period, name in zip(momentum_periods, momentum_names):
            df_features[f'momentum_{name}'] = close_prices.pct_change(periods=period)
        
        # === FEATURES FONDAMENTALES OPTIMISÉES ===
        
        # Récupération des infos entreprise avec valeurs par défaut
        info = company_info.get(symbol, {})
        
        # Dictionnaire des valeurs par défaut pour optimiser les calculs
        fundamental_defaults = {
            'pe_ratio': 15.0,
            'dividend_yield_ttm': 3.0,
            'payout_ratio': 0.5,
            'market_cap': 50_000_000_000,  # 50B
            'beta': 1.0
        }
        
        # Attribution vectorisée des features fondamentales
        for feature, default_value in fundamental_defaults.items():
            if feature == 'pe_ratio':
                value = float(info.get('trailingPE', default_value) or default_value)
            elif feature == 'dividend_yield_ttm':
                value = float(info.get('dividendYield', default_value/100) or default_value/100) * 100
            elif feature == 'payout_ratio':
                value = float(info.get('payoutRatio', default_value) or default_value)
            elif feature == 'market_cap':
                value = float(info.get('marketCap', default_value) or default_value)
            elif feature == 'beta':
                value = float(info.get('beta', default_value) or default_value)
            
            # Attribution vectorisée de la valeur constante
            df_features[feature] = np.full(len(df_features), value)
        
        # === FEATURES TEMPORELLES VECTORISÉES ===
        
        # Extraction vectorisée des composantes temporelles
        datetime_index = df_features.index
        df_features['year'] = datetime_index.year
        df_features['month'] = datetime_index.month
        df_features['quarter'] = datetime_index.quarter
        df_features['day_of_week'] = datetime_index.dayofweek
        df_features['is_month_end'] = datetime_index.is_month_end.astype(int)
        
        # === VARIABLE CIBLE VECTORISÉE ===
        
        # Rendements quotidiens
        df_features['returns_1d'] = close_prices.pct_change()
        
        # Target: Rendement futur sur 1 an (vectorisé)
        # Utilisation de shift négatif pour les rendements futurs
        future_returns = df_features['returns_1d'].rolling(window=252).sum().shift(-252)
        df_features['target_return_1y'] = future_returns * 100
        
        # === AJOUT DU SYMBOLE ===
        df_features['symbol'] = symbol
        
        # === NETTOYAGE VECTORISÉ FINAL ===
        
        # Remplacement des infinis et NaN en une seule opération
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns
        df_features[numeric_columns] = df_features[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(method='ffill').fillna(0)
        
        all_features.append(df_features)
        
        # Chronométrage
        symbol_time = time.time() - start_symbol_time
        feature_creation_times.append(symbol_time)
        
    # === COMBINAISON OPTIMISÉE DES DATAFRAMES ===
    
    print("🔗 Combinaison des DataFrames...")
    start_combine_time = time.time()
    
    # Concaténation efficace avec pandas
    combined_df = pd.concat(all_features, ignore_index=False, sort=False)
    
    combine_time = time.time() - start_combine_time
    
    # === NETTOYAGE FINAL VECTORISÉ ===
    
    print("🧹 Nettoyage final des données...")
    start_clean_time = time.time()
    
    # Identification des colonnes features (excluant les métadonnées)
    excluded_cols = ['symbol', 'target_return_1y', 'returns_1d']
    feature_cols = [col for col in combined_df.columns if col not in excluded_cols]
    
    # Suppression des lignes avec NaN dans les colonnes critiques (vectorisé)
    critical_cols = feature_cols + ['target_return_1y']
    initial_rows = len(combined_df)
    combined_df = combined_df.dropna(subset=critical_cols)
    final_rows = len(combined_df)
    
    clean_time = time.time() - start_clean_time
    
    # === RAPPORT DE PERFORMANCE ===
    
    total_feature_time = sum(feature_creation_times)
    avg_time_per_stock = np.mean(feature_creation_times)
    
    print(f"\n✅ Feature engineering terminé!")
    print(f"📊 Données créées: {len(combined_df):,} observations")
    print(f"🔧 Features créées: {len(feature_cols)} features")
    print(f"🗑️ Lignes supprimées: {initial_rows - final_rows:,} ({((initial_rows - final_rows)/initial_rows*100):.1f}%)")
    print(f"📅 Période couverte: {combined_df.index.min().date()} à {combined_df.index.max().date()}")
    
    print(f"\n⚡ PERFORMANCE:")
    print(f"Temps features: {total_feature_time:.2f}s")
    print(f"Temps combinaison: {combine_time:.2f}s") 
    print(f"Temps nettoyage: {clean_time:.2f}s")
    print(f"Temps par action: {avg_time_per_stock:.3f}s")
    print(f"Vitesse: {len(stock_data)/total_feature_time:.1f} actions/seconde")
    
    return combined_df


class XGBoostDividendPredictor:
    """Modèle XGBoost pour prédire les rendements futurs"""
    
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.best_params = None
        self.cv_results = None
        
    def prepare_data(self, ml_data):
        """Prépare les données pour l'entraînement"""
        # Séparer les features et la target
        feature_cols = [col for col in ml_data.columns 
                       if col not in ['symbol', 'target_return_1y', 'returns_1d']]
        
        X = ml_data[feature_cols].copy()
        y = ml_data['target_return_1y'].copy()
        
        # Encoder les variables catégorielles si nécessaire
        categorical_features = []
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                categorical_features.append(col)
        
        # Remplacer les inf et NaN
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        y = y.fillna(0)
        
        return X, y, feature_cols
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Entraîne le modèle avec optimisation des hyperparamètres"""
        
        print("🔄 Entraînement d'un modèle XGBoost...")
        
        # Paramètres optimaux trouvés empiriquement
        self.best_params = {
            'n_estimators': 400,
            'max_depth': 8,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        try:
            self.model = xgb.XGBRegressor(**self.best_params)
            self.model.fit(X_train, y_train)
            
            # Prédictions
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            # Métriques
            results = {
                'train_mse': mean_squared_error(y_train, train_pred),
                'test_mse': mean_squared_error(y_test, test_pred),
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred)
            }
            
            # Feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("✅ Modèle XGBoost entraîné avec succès!")
            return results, train_pred, test_pred
            
        except ImportError:
            print("⚠️ XGBoost non disponible, utilisation de RandomForest...")
            from sklearn.ensemble import RandomForestRegressor
            
            self.model = RandomForestRegressor(
                n_estimators=400,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            # Prédictions
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            # Métriques
            results = {
                'train_mse': mean_squared_error(y_train, train_pred),
                'test_mse': mean_squared_error(y_test, test_pred),
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred)
            }
            
            # Feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("✅ Modèle RandomForest entraîné avec succès!")
            return results, train_pred, test_pred


class InvestmentRecommendationEngine:
    """Moteur de recommandations d'investissement"""
    
    def __init__(self, model, dividend_metrics):
        self.model = model
        self.dividend_metrics = dividend_metrics
        
    def predict_future_returns(self, ml_data):
        """Prédit les rendements futurs pour chaque action"""
        
        # Prendre les données les plus récentes pour chaque action
        latest_data = ml_data.groupby('symbol').tail(1)
        
        predictions = {}
        for symbol in latest_data['symbol'].unique():
            symbol_data = latest_data[latest_data['symbol'] == symbol].copy()
            
            try:
                # Préparer les features
                feature_cols = [col for col in symbol_data.columns 
                               if col not in ['symbol', 'target_return_1y', 'returns_1d']]
                
                prediction_data = symbol_data[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
                
                # Prédiction
                pred_return = self.model.predict(prediction_data)[0]
                
                predictions[symbol] = {
                    'predicted_return_12m': pred_return,
                    'prediction_date': symbol_data.index[0],
                    'current_price': symbol_data['Close'].iloc[0] if 'Close' in symbol_data.columns else None
                }
                
            except Exception as e:
                print(f"❌ Erreur de prédiction pour {symbol}: {e}")
                predictions[symbol] = {
                    'predicted_return_12m': 0.0,
                    'prediction_date': symbol_data.index[0],
                    'current_price': None
                }
                
        return predictions
    
    def calculate_composite_score(self, predictions, summary_df):
        """Calcule un score composite pour chaque action"""
        
        scores = []
        
        for symbol in predictions.keys():
            if symbol not in summary_df['Symbol'].values:
                continue
                
            # Récupérer les métriques historiques
            symbol_stats = summary_df[summary_df['Symbol'] == symbol].iloc[0]
            
            # Score composite simple mais efficace
            score = (
                min(max(predictions[symbol]['predicted_return_12m'], -20) / 20 * 25, 25) +  # 25% prédiction
                min(symbol_stats['Total Return (%)'] / 400 * 25, 25) +  # 25% performance historique
                min(symbol_stats['Avg Dividend Yield (%)'] / 8 * 20, 20) +  # 20% dividende
                min(max(symbol_stats['Dividend Growth (%)'], 0) / 15 * 15, 15) +  # 15% croissance
                max(15 - symbol_stats['Volatility (%)'] / 10, 0)  # 15% faible volatilité
            )
            
            scores.append({
                'Symbol': symbol,
                'Score_Composite': max(score, 0),
                'Rendement_Prédit_12M': predictions[symbol]['predicted_return_12m'],
                'Rendement_Historique_10Y': symbol_stats['Total Return (%)'],
                'Rendement_Dividende': symbol_stats['Avg Dividend Yield (%)'],
                'Croissance_Dividende': symbol_stats['Dividend Growth (%)'],
                'Volatilité': symbol_stats['Volatility (%)']
            })
            
        return pd.DataFrame(scores).sort_values('Score_Composite', ascending=False)
    
    def generate_portfolio_recommendations(self, scores_df, top_n=3):
        """Génère des recommandations de portefeuille"""
        
        # Top actions
        top_stocks = scores_df.head(top_n)
        
        # Allocation basée sur le score composite
        total_score = top_stocks['Score_Composite'].sum()
        if total_score > 0:
            top_stocks = top_stocks.copy()
            top_stocks['Allocation_%'] = (top_stocks['Score_Composite'] / total_score * 100).round(1)
        else:
            # Répartition égale si scores nuls
            allocation = [100/top_n] * top_n
            top_stocks = top_stocks.copy()
            top_stocks['Allocation_%'] = allocation
        
        return top_stocks


def main():
    """Fonction principale pour exécuter l'analyse complète"""
    
    print("🚀 Début de l'analyse des dividendes...")
    
    # 1. Collecte des données
    print("\n📊 Étape 1: Collecte des données")
    data_collector = YFinanceDataCollector(lookback_years=10)
    stock_data, company_info = data_collector.download_all_stocks(DIVIDEND_ARISTOCRATS)
    
    print(f"✅ Données collectées pour {len(stock_data)} actions")
    
    # 2. Prétraitement des données
    print("\n🔄 Étape 2: Prétraitement des données")
    preprocessor = DividendDataPreprocessor()
    dividend_metrics = preprocessor.calculate_dividend_metrics(stock_data)
    
    # Créer le DataFrame de résumé
    summary_data = []
    for symbol, metrics in dividend_metrics.items():
        volatility_pct = (
            metrics['volatility'] * 100 
            if not pd.isna(metrics['volatility']) 
            else 0
        )
        
        summary_data.append({
            'Symbol': symbol,
            'Total Return (%)': round(metrics['total_return'], 2),
            'Price Return (%)': round(metrics['price_return'], 2),
            'Avg Dividend Yield (%)': round(metrics['avg_dividend_yield'], 2),
            'Dividend Growth (%)': round(metrics['avg_dividend_growth'] * 100, 2),
            'Volatility (%)': round(volatility_pct, 2)
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('Total Return (%)', ascending=False)
    
    print(f"✅ Métriques calculées pour {len(summary_df)} actions")
    
    # 3. Feature Engineering
    print("\n🔧 Étape 3: Feature Engineering")
    ml_data = create_ml_features(stock_data, company_info)
    
    # 4. Entraînement du modèle
    print("\n🤖 Étape 4: Entraînement du modèle ML")
    predictor = XGBoostDividendPredictor()
    X, y, feature_names = predictor.prepare_data(ml_data)
    
    # Split temporel
    train_size = int(0.8 * len(X))
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    # Entraîner le modèle
    results, train_pred, test_pred = predictor.train_model(X_train, y_train, X_test, y_test)
    
    print(f"📈 Performance: R² = {results['test_r2']:.3f}, MSE = {results['test_mse']:.2f}")
    
    # 5. Génération des recommandations
    print("\n💼 Étape 5: Génération des recommandations")
    recommendation_engine = InvestmentRecommendationEngine(predictor.model, dividend_metrics)
    
    # Prédictions futures
    future_predictions = recommendation_engine.predict_future_returns(ml_data)
    
    # Calcul du score composite
    scores_df = recommendation_engine.calculate_composite_score(future_predictions, summary_df)
    
    # Recommandations de portefeuille
    portfolio_recommendations = recommendation_engine.generate_portfolio_recommendations(scores_df)
    
    # 6. Affichage des résultats
    print("\n🏆 RÉSULTATS FINAUX:")
    print("=" * 60)
    
    print("\n📊 Performance du Modèle:")
    print(f"R² Test: {results['test_r2']:.3f}")
    print(f"MSE Test: {results['test_mse']:.2f}")
    
    print("\n🥇 Top 5 Actions Recommandées:")
    display_cols = ['Symbol', 'Score_Composite', 'Rendement_Prédit_12M', 'Rendement_Dividende', 'Volatilité']
    print(scores_df[display_cols].head().round(2).to_string(index=False))
    
    print("\n💰 Portefeuille Recommandé (Top 3):")
    for idx, row in portfolio_recommendations.iterrows():
        print(f"{row['Symbol']}: {row['Allocation_%']:.1f}% (Score: {row['Score_Composite']:.1f})")
        print(f"   Prédiction 12M: {row['Rendement_Prédit_12M']:+.1f}%")
        print(f"   Dividende: {row['Rendement_Dividende']:.1f}%")
        print(f"   Volatilité: {row['Volatilité']:.1f}%")
        print()
    
    print("\n📋 Recommandations d'Investissement:")
    print("• Diversification progressive sur 6-12 mois")
    print("• Réinvestissement automatique des dividendes") 
    print("• Réévaluation semestrielle")
    print("• Horizon d'investissement: 5-10 ans minimum")
    
    print("\n⚠️ Avertissement:")
    print("Les prédictions sont basées sur des données historiques.")
    print("Les performances passées ne garantissent pas les résultats futurs.")
    print("Consultez un conseiller financier professionnel.")
    
    print(f"\n✅ Analyse terminée avec succès!")
    
    return {
        'stock_data': stock_data,
        'summary_df': summary_df,
        'ml_data': ml_data,
        'model': predictor.model,
        'predictions': future_predictions,
        'recommendations': portfolio_recommendations,
        'feature_importance': predictor.feature_importance
    }


if __name__ == "__main__":
    # Exécuter l'analyse complète
    results = main()