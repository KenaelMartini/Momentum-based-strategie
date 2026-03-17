# ============================================================
# ibkr_data.py — Data Layer | Connexion IBKR & Récupération
# ============================================================
# RÔLE DE CE FICHIER :
# Ce fichier est le "pont" entre ton code Python et Interactive
# Brokers. Il gère :
#   1. La connexion à TWS
#   2. La définition des contrats (actions, futures)
#   3. Le téléchargement des données historiques
#   4. Le nettoyage et la normalisation des données
#   5. La sauvegarde locale (cache) pour éviter les re-téléchargements
#
# CONCEPT CLÉ — L'API IBKR fonctionne de manière ASYNCHRONE :
# Tu envoies une requête, et TWS te répond via un "callback"
# (une fonction qui se déclenche quand la réponse arrive).
# ib_insync simplifie tout ça avec une syntaxe synchrone (plus lisible).
# ============================================================

import time
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
from ib_insync import IB, Stock, Future, ContFuture, util

# On importe nos paramètres de configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID,
    STOCK_UNIVERSE, FUTURES_UNIVERSE,
    DATA_FREQUENCY, IBKR_HISTORY_DURATION, IBKR_DATA_TYPE,
    DATA_PATH, PROCESSED_DATA_PATH
)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION DU LOGGING
# ─────────────────────────────────────────────────────────────
# Le logging est ESSENTIEL en trading quant.
# On ne print() jamais en production — on utilise des logs.
# Pourquoi ? Les logs sont horodatés, filtrables, et peuvent
# être sauvegardés dans des fichiers pour audit ultérieur.
#
# Niveaux de log (du moins grave au plus grave) :
#   DEBUG    → Détails techniques fins
#   INFO     → Informations générales de fonctionnement
#   WARNING  → Quelque chose d'inattendu mais pas bloquant
#   ERROR    → Une erreur s'est produite
#   CRITICAL → Erreur fatale, le programme ne peut pas continuer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CLASSE PRINCIPALE : IBKRDataFetcher
# ─────────────────────────────────────────────────────────────
# On encapsule toute la logique dans une CLASSE.
# Pourquoi une classe et pas des fonctions simples ?
#   1. La connexion IBKR est un ÉTAT — on doit la garder ouverte
#   2. On peut réutiliser la même connexion pour plusieurs requêtes
#   3. Plus propre, plus maintenable, plus professionnel
#
# CONCEPT POO : __init__ est le "constructeur" — il s'exécute
# automatiquement quand tu crées un objet IBKRDataFetcher().

class IBKRDataFetcher:
    """
    Gère la connexion à IBKR TWS et le téléchargement
    des données historiques pour actions et futures.
    """

    def __init__(self):
        # IB() est l'objet principal d'ib_insync.
        # Il représente la connexion à TWS.
        self.ib = IB()

        # On crée les dossiers de données s'ils n'existent pas
        # exist_ok=True évite une erreur si le dossier existe déjà
        os.makedirs(DATA_PATH, exist_ok=True)
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

        logger.info("IBKRDataFetcher initialisé.")

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 1 : Connexion à TWS
    # ─────────────────────────────────────────────────────────
    def connect(self) -> bool:
        """
        Se connecte à TWS sur le port Paper Trading.
        Retourne True si succès, False sinon.

        PRÉREQUIS : TWS doit être ouvert et configuré pour
        accepter les connexions API (File → Global Config →
        API → Settings → Enable ActiveX and Socket Clients).
        """
        try:
            # connect() établit la connexion TCP/IP avec TWS
            # host  : adresse IP de TWS (127.0.0.1 = même machine)
            # port  : port d'écoute de TWS
            # clientId : identifiant unique de cette session
            self.ib.connect(
                host=IBKR_HOST,
                port=IBKR_PORT,
                clientId=IBKR_CLIENT_ID
            )
            logger.info(
                f"✅ Connecté à IBKR TWS | "
                f"Host: {IBKR_HOST} | Port: {IBKR_PORT} | "
                f"Account: {self.ib.wrapper.accounts}"
            )
            return True

        except Exception as e:
            # On capture TOUTES les exceptions pour ne jamais
            # planter silencieusement — on log et on retourne False
            logger.error(f"❌ Échec de connexion à IBKR : {e}")
            logger.error(
                "Vérifie que TWS est ouvert et que l'API est activée "
                "(File → Global Config → API → Enable Socket Clients)"
            )
            return False

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 2 : Déconnexion
    # ─────────────────────────────────────────────────────────
    def disconnect(self):
        """
        Ferme proprement la connexion avec TWS.
        TOUJOURS appeler cette méthode à la fin du script
        pour libérer le socket et éviter les conflits de clientId.
        """
        self.ib.disconnect()
        logger.info("🔌 Déconnecté de IBKR TWS.")

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 3 : Créer un contrat ACTION
    # ─────────────────────────────────────────────────────────
    def _create_stock_contract(self, symbol: str) -> Stock:
        """
        Crée un objet "contrat" IBKR pour une action.

        CONCEPT IBKR — Les CONTRATS :
        IBKR ne travaille pas avec des "tickers" simples.
        Chaque instrument financier est un CONTRAT avec des
        attributs précis : symbole, exchange, devise, type...
        Cette précision évite les ambiguïtés (ex: "AAPL" peut
        exister sur plusieurs bourses mondiales).

        Args:
            symbol: Ticker de l'action (ex: "AAPL")

        Returns:
            Objet Stock (contrat IBKR)
        """
        # Stock(symbol, exchange, currency)
        # SMART = "Smart Routing" d'IBKR — il trouve automatiquement
        # le meilleur exchange pour exécuter l'ordre.
        # Pour le data historique, SMART fonctionne pour les US stocks.
        return Stock(symbol, "SMART", "USD")

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 4 : Créer un contrat FUTURE CONTINU
    # ─────────────────────────────────────────────────────────
    def _create_continuous_future_contract(self, symbol: str) -> ContFuture:
        """
        Crée un contrat FUTURE CONTINU pour les commodités.

        CONCEPT — FUTURES ET ROLL :
        Un future est un contrat avec une date d'EXPIRATION.
        Ex: le contrat "CLZ24" = Crude Oil, expiration décembre 2024.
        Après expiration, il faut passer au contrat suivant = "ROLL".

        PROBLÈME : Si on concatène les contrats bruts, on a des
        "sauts" artificiels dans les prix au moment du roll.
        Ces sauts faussent complètement le calcul des rendements.

        SOLUTION : Le CONTRAT CONTINU (ContFuture chez IBKR) est
        un contrat synthétique qui gère automatiquement les rolls
        et ajuste les prix pour avoir une série cohérente.
        C'est le standard pour backtester sur des futures.

        EXCHANGES des futures commodités :
        CL, NG → NYMEX | GC, SI, HG → COMEX | ZC, ZW, ZS → CBOT
        """
        # Dictionnaire de mapping symbole → exchange
        # On le définit localement car c'est propre à cette méthode
        exchange_map = {
            "CL": "NYMEX",  # Crude Oil
            "NG": "NYMEX",  # Natural Gas
            "GC": "COMEX",  # Gold
            "SI": "COMEX",  # Silver
            "HG": "COMEX",  # Copper
            "ZC": "CBOT",   # Corn
            "ZW": "CBOT",   # Wheat
            "ZS": "CBOT",   # Soybean
        }

        exchange = exchange_map.get(symbol, "NYMEX")

        # ContFuture = Continuous Future (contrat continu)
        return ContFuture(symbol, exchange, currency="USD")

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 5 : Télécharger les données historiques d'un actif
    # ─────────────────────────────────────────────────────────
    def _fetch_historical_bars(
        self,
        contract,
        duration: str = IBKR_HISTORY_DURATION,
        bar_size: str = DATA_FREQUENCY,
        what_to_show: str = IBKR_DATA_TYPE
    ) -> Optional[pd.DataFrame]:
        """
        Télécharge les données OHLCV historiques pour un contrat.

        CONCEPT — OHLCV :
        Chaque "barre" (candle) contient 5 informations :
          O = Open  (prix d'ouverture de la période)
          H = High  (prix le plus haut de la période)
          L = Low   (prix le plus bas de la période)
          C = Close (prix de clôture de la période)
          V = Volume (nombre de titres échangés)
        On travaille principalement avec les prix de CLÔTURE
        pour calculer les rendements.

        Args:
            contract   : objet contrat IBKR (Stock ou ContFuture)
            duration   : durée d'historique ("10 Y", "6 M", etc.)
            bar_size   : taille de chaque barre ("1 day", "1 hour")
            what_to_show: type de prix ("TRADES", "MIDPOINT", "BID", "ASK")

        Returns:
            DataFrame pandas avec colonnes [open, high, low, close, volume]
            ou None si erreur
        """
        try:
            # reqHistoricalData() = requête principale IBKR
            # endDateTime = "" → jusqu'à maintenant
            # useRTH = True → Regular Trading Hours seulement
            #   (exclut le pre-market et after-hours)
            #   Pour des stratégies daily, on veut les heures officielles.
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,       # Regular Trading Hours
                formatDate=1,      # Dates en format lisible
                keepUpToDate=False # On veut juste l'historique statique
            )

            if not bars:
                logger.warning(
                    f"⚠️ Aucune donnée reçue pour {contract.symbol}"
                )
                return None

            # util.df() convertit la liste de barres IBKR en DataFrame pandas
            # C'est LA structure de données centrale en data science Python
            df = util.df(bars)

            # On garde seulement les colonnes utiles
            df = df[["date", "open", "high", "low", "close", "volume"]]

            # On transforme la colonne date en index temporel
            # C'est le standard pour les séries temporelles financières
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            # On trie par date croissante (plus ancien → plus récent)
            df.sort_index(inplace=True)

            logger.info(
                f"✅ {contract.symbol} : {len(df)} barres récupérées "
                f"({df.index[0].date()} → {df.index[-1].date()})"
            )

            return df

        except Exception as e:
            logger.error(
                f"❌ Erreur récupération données {contract.symbol} : {e}"
            )
            return None

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 6 : Nettoyage et validation des données
    # ─────────────────────────────────────────────────────────
    def _clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Nettoie et valide les données brutes IBKR.

        CONCEPT — DATA CLEANING :
        Les données financières brutes contiennent souvent des
        anomalies qui fausseraient nos calculs :
          - Valeurs manquantes (NaN) : jours fériés, pannes, etc.
          - Prix aberrants : erreurs de saisie, "fat fingers"
          - Volume nul : jours sans échanges
          - Doublons : parfois l'API renvoie deux fois la même barre

        En institution, le data cleaning est une étape critique.
        Des données sales = des signaux erronés = des trades perdants.

        Args:
            df     : DataFrame brut d'IBKR
            symbol : nom de l'actif (pour les logs)

        Returns:
            DataFrame nettoyé
        """
        initial_rows = len(df)

        # --- ÉTAPE 1 : Supprimer les doublons d'index ---
        # Si deux barres ont la même date, on garde la dernière
        df = df[~df.index.duplicated(keep="last")]

        # --- ÉTAPE 2 : Gérer les valeurs manquantes ---
        # On compte les NaN avant traitement (pour le log)
        nan_count = df.isnull().sum().sum()

        # forward fill (ffill) : remplace NaN par la dernière valeur connue
        # C'est la méthode standard pour les prix financiers.
        # Logique : si le marché n'a pas tradé ce jour-là (jour férié),
        # le prix reste le même que la veille.
        df = df.ffill()

        # Si des NaN persistent au début de la série (avant le 1er prix),
        # on les supprime — on ne peut pas interpoler dans le vide.
        df = df.dropna()

        # --- ÉTAPE 3 : Supprimer les prix aberrants ---
        # Un prix ne peut pas être ≤ 0 (sauf en cas de bug ou short squeeze extrême)
        df = df[df["close"] > 0]
        df = df[df["open"] > 0]
        df = df[df["high"] > 0]
        df = df[df["low"] > 0]

        # --- ÉTAPE 4 : Vérifier la cohérence OHLC ---
        # High doit être ≥ max(Open, Close) et Low ≤ min(Open, Close)
        # On filtre les barres où ce n'est pas le cas (erreurs de données)
        valid_ohlc = (
            (df["high"] >= df[["open", "close"]].max(axis=1)) &
            (df["low"]  <= df[["open", "close"]].min(axis=1))
        )
        invalid_count = (~valid_ohlc).sum()
        if invalid_count > 0:
            logger.warning(
                f"⚠️ {symbol} : {invalid_count} barres OHLC incohérentes supprimées"
            )
        df = df[valid_ohlc]

        # --- ÉTAPE 5 : Détecter les rendements aberrants ---
        # Un rendement journalier de ±50% est suspect pour une action liquide.
        # Cela peut indiquer un split, un dividend ajustement non géré, ou
        # une erreur de données.
        # ATTENTION : On log seulement, on ne supprime pas automatiquement
        # car ça peut être un vrai mouvement extrême (ex: biotech FDA approval)
        returns = df["close"].pct_change()
        extreme_returns = (returns.abs() > 0.50).sum()
        if extreme_returns > 0:
            logger.warning(
                f"⚠️ {symbol} : {extreme_returns} rendements >50% détectés. "
                f"Vérifier si les données sont ajustées pour les splits/dividendes."
            )

        final_rows = len(df)
        logger.info(
            f"🧹 {symbol} nettoyé : {initial_rows} → {final_rows} barres "
            f"({nan_count} NaN comblés)"
        )

        return df

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 7 : Sauvegarder / Charger depuis le cache local
    # ─────────────────────────────────────────────────────────
    def _save_to_cache(self, df: pd.DataFrame, symbol: str, asset_type: str):
        """
        Sauvegarde les données en CSV local.

        POURQUOI UN CACHE ?
        L'API IBKR a des limites de requêtes (pacing limits).
        Si tu re-lances ton script 10 fois, tu vas te faire
        throttler (bloquer temporairement) par IBKR.
        On sauvegarde donc localement et on ne re-télécharge
        que si les données sont obsolètes.
        """
        filename = f"{DATA_PATH}{asset_type}_{symbol}.csv"
        df.to_csv(filename)
        logger.debug(f"💾 Données sauvegardées : {filename}")

    def _load_from_cache(
        self, symbol: str, asset_type: str
    ) -> Optional[pd.DataFrame]:
        """
        Charge les données depuis le cache local si elles existent
        et sont récentes (moins de 24h).

        Returns:
            DataFrame si cache valide, None sinon
        """
        filename = f"{DATA_PATH}{asset_type}_{symbol}.csv"

        if not os.path.exists(filename):
            return None  # Pas de cache → on doit télécharger

        # Vérifier la fraîcheur du cache
        # Si le fichier a plus de 24h, on re-télécharge
        file_age_hours = (
            time.time() - os.path.getmtime(filename)
        ) / 3600

        if file_age_hours > 24:
            logger.info(
                f"🔄 Cache {symbol} expiré ({file_age_hours:.1f}h) → re-téléchargement"
            )
            return None

        # Chargement du CSV avec parsing automatique des dates
        df = pd.read_csv(filename, index_col="date", parse_dates=True)
        logger.info(f"📂 {symbol} chargé depuis le cache local")
        return df

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 8 : Pipeline complète pour les ACTIONS
    # ─────────────────────────────────────────────────────────
    def fetch_stocks_data(
        self,
        symbols: list = None,
        use_cache: bool = True
    ) -> dict:
        """
        Pipeline principale pour télécharger les données de toutes les actions.

        CONCEPT — PIPELINE :
        Une pipeline de données est une séquence d'étapes qui
        transforment les données brutes en données prêtes à l'usage :
          Requête IBKR → Nettoyage → Cache → Retour

        Args:
            symbols   : liste de tickers (défaut: STOCK_UNIVERSE de config)
            use_cache : utiliser le cache local si disponible

        Returns:
            dict { "AAPL": DataFrame, "MSFT": DataFrame, ... }
        """
        if symbols is None:
            symbols = STOCK_UNIVERSE

        all_data = {}  # Dictionnaire résultat : symbol → DataFrame

        logger.info(f"📥 Téléchargement données pour {len(symbols)} actions...")

        for i, symbol in enumerate(symbols):
            logger.info(f"  [{i+1}/{len(symbols)}] Traitement de {symbol}...")

            # Vérifier le cache d'abord
            if use_cache:
                cached = self._load_from_cache(symbol, "stock")
                if cached is not None:
                    all_data[symbol] = cached
                    continue  # On passe au symbole suivant

            # Pas de cache valide → on télécharge depuis IBKR
            contract = self._create_stock_contract(symbol)

            # IBKR nécessite que le contrat soit "qualifié" avant usage
            # qualifyContracts() vérifie que le contrat existe et remplit
            # les champs manquants (conId, exchange précis, etc.)
            try:
                self.ib.qualifyContracts(contract)
            except Exception as e:
                logger.error(f"❌ Contrat invalide pour {symbol} : {e}")
                continue

            # Téléchargement
            df = self._fetch_historical_bars(contract)

            if df is None:
                logger.warning(f"⚠️ {symbol} ignoré (données indisponibles)")
                continue

            # Nettoyage
            df = self._clean_data(df, symbol)

            # Sauvegarde en cache
            self._save_to_cache(df, symbol, "stock")

            all_data[symbol] = df

            # IBKR PACING LIMITS — IMPORTANT :
            # IBKR limite à ~50 requêtes par 10 minutes.
            # Si tu envoies trop vite, tu reçois une erreur "pacing violation".
            # On attend 0.5s entre chaque requête pour éviter ça.
            time.sleep(0.5)

        logger.info(
            f"✅ Données actions récupérées : {len(all_data)}/{len(symbols)} succès"
        )
        return all_data

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 9 : Pipeline complète pour les FUTURES
    # ─────────────────────────────────────────────────────────
    def fetch_futures_data(
        self,
        symbols: list = None,
        use_cache: bool = True
    ) -> dict:
        """
        Pipeline pour télécharger les données des futures continus.

        DIFFÉRENCE AVEC LES STOCKS :
        - On utilise ContFuture au lieu de Stock
        - Les futures ont une mécanique de ROLL à gérer
        - Les exchanges sont différents (NYMEX, COMEX, CBOT)
        - Le type de données est TRADES (même que stocks)

        Args:
            symbols   : liste de symboles futures
            use_cache : utiliser le cache local

        Returns:
            dict { "CL": DataFrame, "GC": DataFrame, ... }
        """
        if symbols is None:
            symbols = FUTURES_UNIVERSE

        all_data = {}

        logger.info(
            f"📥 Téléchargement données pour {len(symbols)} futures continus..."
        )

        for i, symbol in enumerate(symbols):
            logger.info(f"  [{i+1}/{len(symbols)}] Traitement de {symbol}...")

            if use_cache:
                cached = self._load_from_cache(symbol, "future")
                if cached is not None:
                    all_data[symbol] = cached
                    continue

            contract = self._create_continuous_future_contract(symbol)

            try:
                self.ib.qualifyContracts(contract)
            except Exception as e:
                logger.error(
                    f"❌ Contrat future invalide pour {symbol} : {e}"
                )
                continue

            # Pour les futures, on utilise TRADES également
            # mais certains futures illiquides nécessitent MIDPOINT
            df = self._fetch_historical_bars(
                contract,
                what_to_show="TRADES"
            )

            if df is None:
                logger.warning(f"⚠️ {symbol} ignoré")
                continue

            df = self._clean_data(df, symbol)
            self._save_to_cache(df, symbol, "future")
            all_data[symbol] = df

            time.sleep(0.5)  # Respect des pacing limits

        logger.info(
            f"✅ Données futures récupérées : {len(all_data)}/{len(symbols)} succès"
        )
        return all_data

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 10 : Construire la matrice des prix de clôture
    # ─────────────────────────────────────────────────────────
    def build_price_matrix(self, data_dict: dict) -> pd.DataFrame:
        """
        Combine les DataFrames individuels en UNE SEULE matrice de prix.

        POURQUOI ?
        La plupart des calculs de momentum et de portefeuille sont
        vectorisés — on travaille sur TOUTES les colonnes (actifs)
        simultanément plutôt qu'en boucle sur chaque actif.
        C'est BEAUCOUP plus rapide (numpy calcule en C sous le capot).

        Exemple de sortie :
                    AAPL    MSFT    GOOGL   CL      GC
        2020-01-02  294.6   160.6   1368.2  61.1    1527.3
        2020-01-03  296.2   158.9   1361.5  62.3    1551.2
        ...

        Args:
            data_dict : dictionnaire { symbol: DataFrame }

        Returns:
            DataFrame avec une colonne par actif, une ligne par date
        """
        price_dict = {}

        for symbol, df in data_dict.items():
            if df is not None and "close" in df.columns:
                price_dict[symbol] = df["close"]

        if not price_dict:
            logger.error("❌ Aucune donnée valide pour construire la matrice")
            return pd.DataFrame()

        # pd.DataFrame(dict) crée un DataFrame où chaque valeur
        # du dict devient une colonne
        price_matrix = pd.DataFrame(price_dict)

        # On aligne toutes les séries sur les mêmes dates
        # (certains actifs peuvent avoir des jours fériés différents)
        # forward fill pour gérer les jours où certains actifs
        # n'ont pas tradé (ex: futures vs stocks ont des calendriers différents)
        price_matrix = price_matrix.ffill()

        # On retire les lignes où TOUS les actifs ont NaN
        price_matrix = price_matrix.dropna(how="all")

        logger.info(
            f"📊 Matrice des prix construite : "
            f"{price_matrix.shape[0]} jours × {price_matrix.shape[1]} actifs"
        )

        # Sauvegarde de la matrice traitée
        price_matrix.to_csv(f"{PROCESSED_DATA_PATH}price_matrix.csv")
        logger.info(f"💾 Matrice sauvegardée dans {PROCESSED_DATA_PATH}price_matrix.csv")

        return price_matrix


# ─────────────────────────────────────────────────────────────
# SCRIPT PRINCIPAL — Point d'entrée
# ─────────────────────────────────────────────────────────────
# Cette structure if __name__ == "__main__" est un standard Python.
# Le code ici s'exécute SEULEMENT si tu lances ce fichier directement
# (python ibkr_data.py) et PAS si tu l'importes depuis un autre fichier.
# C'est une bonne pratique car ça permet d'utiliser ce fichier
# à la fois comme module importable ET comme script standalone.

if __name__ == "__main__":

    print("=" * 60)
    print("  IBKR DATA FETCHER — Stratégie Momentum")
    print("=" * 60)

    # 1. Initialisation
    fetcher = IBKRDataFetcher()

    # 2. Connexion à TWS
    if not fetcher.connect():
        print("❌ Impossible de se connecter à TWS. Arrêt.")
        exit(1)

    try:
        # 3. Téléchargement des données actions
        print("\n📥 Récupération des données actions...")
        stocks_data = fetcher.fetch_stocks_data(use_cache=True)

        # 4. Téléchargement des données futures
        print("\n📥 Récupération des données futures...")
        futures_data = fetcher.fetch_futures_data(use_cache=True)

        # 5. Fusion en matrice complète
        print("\n📊 Construction de la matrice des prix...")
        all_data = {**stocks_data, **futures_data}  # Fusion des deux dicts
        price_matrix = fetcher.build_price_matrix(all_data)

        # 6. Résumé
        print("\n" + "=" * 60)
        print("  RÉSUMÉ")
        print("=" * 60)
        print(f"  Actions récupérées : {len(stocks_data)}")
        print(f"  Futures récupérés  : {len(futures_data)}")
        print(f"  Matrice finale     : {price_matrix.shape}")
        print(f"  Période couverte   : "
              f"{price_matrix.index[0].date()} → "
              f"{price_matrix.index[-1].date()}")
        print("\n  Aperçu de la matrice :")
        print(price_matrix.tail())

    finally:
        # Le bloc finally s'exécute TOUJOURS, même si une erreur survient.
        # On est ainsi sûr de toujours fermer la connexion proprement.
        fetcher.disconnect()