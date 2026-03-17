# ============================================================
# config.py — Configuration globale | Stratégie Momentum
# ============================================================
# RÔLE DE CE FICHIER :
# En quant trading professionnel, on ne "hardcode" (écrit en dur)
# jamais de valeurs directement dans le code métier.
# Toute la configuration est centralisée ici pour deux raisons :
#   1. LISIBILITÉ  : on voit tous les paramètres d'un coup d'œil
#   2. MAINTENANCE : modifier un paramètre = toucher UN seul fichier
# ============================================================


# ─────────────────────────────────────────────────────────────
# SECTION 1 — CONNEXION IBKR
# ─────────────────────────────────────────────────────────────
# TWS (Trader Workstation) tourne en local sur ta machine.
# Python se connecte à TWS via un socket TCP/IP (protocole réseau).
# TWS agit comme un "pont" entre ton code et les marchés IBKR.
#
# PORTS PAR DÉFAUT :
#   TWS Paper Trading  → 7497   ← on utilise celui-ci
#   TWS Live Trading   → 7496
#   IB Gateway Paper   → 4002
#   IB Gateway Live    → 4001
#
# CLIENT_ID : identifiant unique de ta session Python.
# Si tu lances plusieurs scripts en même temps (ex: data + execution),
# chaque script doit avoir un CLIENT_ID différent.

IBKR_HOST      = "127.0.0.1"  # localhost — TWS est sur ta machine
IBKR_PORT      = 7497          # Paper Trading
IBKR_CLIENT_ID = 1             # ID de cette session


# ─────────────────────────────────────────────────────────────
# SECTION 2 — UNIVERS D'ACTIFS
# ─────────────────────────────────────────────────────────────
# L'"univers" = l'ensemble des actifs sur lesquels la stratégie
# peut potentiellement prendre des positions.
#
# POURQUOI CES ACTIONS ?
# On choisit des large caps US très liquides pour deux raisons :
#   1. Données historiques abondantes et fiables
#   2. Coûts de transaction faibles (spreads serrés, volume élevé)
# En institution, l'univers peut contenir des milliers d'actifs.

STOCK_UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Finance
    "JPM", "GS", "BAC", "MS", "BLK",
    # Santé
    "JNJ", "PFE", "UNH", "ABBV",
    # Énergie
    "XOM", "CVX",
    # Consommation
    "WMT", "HD", "MCD", "NKE",
    # Industrie
    "CAT", "BA", "GE",
]

# Futures sur commodités — symboles IBKR (racine du contrat)
# Les futures ont une date d'expiration → on utilisera le
# "continuous contract" (contrat perpétuel synthétique) pour
# avoir un historique long sans rupture. On verra ça dans ibkr_data.py
FUTURES_UNIVERSE = [
    "CL",  # Crude Oil (WTI) — NYMEX
    "GC",  # Gold           — COMEX
    "SI",  # Silver         — COMEX
    "ZC",  # Corn           — CBOT
    "ZW",  # Wheat          — CBOT
    "ZS",  # Soybean        — CBOT
    "NG",  # Natural Gas    — NYMEX
    "HG",  # Copper         — COMEX
]


# ─────────────────────────────────────────────────────────────
# SECTION 3 — PARAMÈTRES DU SIGNAL MOMENTUM
# ─────────────────────────────────────────────────────────────
# Le signal momentum mesure la performance passée d'un actif.
# On utilise PLUSIEURS fenêtres temporelles et on les combine
# pour avoir un signal plus robuste (moins de bruit).
#
# FENÊTRES EN JOURS DE TRADING (≈252 jours/an) :
#   21  jours ≈ 1 mois
#   63  jours ≈ 3 mois
#   126 jours ≈ 6 mois
#   252 jours ≈ 12 mois

MOMENTUM_WINDOWS = [21, 63, 126, 252]

# Poids de chaque fenêtre dans le signal composite.
# On donne plus de poids aux fenêtres longues car elles
# capturent mieux le "vrai" momentum vs le bruit court terme.
# RÈGLE : les poids doivent sommer à 1.0
# MOMENTUM_WEIGHTS = {
#     21:  0.10,   # 1 mois  → 10%
#     63:  0.20,   # 3 mois  → 20%
#     126: 0.30,   # 6 mois  → 30%
#     252: 0.40,   # 12 mois → 40%
# } # avant opti

MOMENTUM_WEIGHTS = {
    21:  0.30,   # 1 mois  → 30%
    63:  0.30,   # 3 mois  → 30%
    126: 0.25,   # 6 mois  → 25%
    252: 0.15,   # 12 mois → 15%
} # après opti


# SKIP PERIOD — Pourquoi on ignore le dernier mois ?
# À très court terme (< 1 mois), le marché est en MEAN REVERSION :
# ce qui vient de monter fort a tendance à légèrement reculer
# (prise de profits, market impact, etc.).
# Ce phénomène s'appelle le "short-term reversal" (Jegadeesh 1990).
# Si on l'inclut, il "pollue" notre signal momentum et réduit
# le Sharpe Ratio. On le skip donc systématiquement.
# SKIP_DAYS = 21  # On ignore les 21 derniers jours de trading
SKIP_DAYS = 10 # après opti


# Fréquence de rebalancement du portefeuille
# "monthly" = on reconstruit le portefeuille 1 fois par mois
# C'est le standard académique pour le momentum.
# Plus fréquent → plus de coûts de transaction
# Moins fréquent → signal moins frais
REBALANCING_FREQUENCY = "monthly"  # options: "daily", "weekly", "monthly"


# ─────────────────────────────────────────────────────────────
# SECTION 4 — CONSTRUCTION DU PORTEFEUILLE
# ─────────────────────────────────────────────────────────────
# On construit un portefeuille LONG/SHORT :
#   LONG  → on achète les actifs avec le meilleur momentum (winners)
#   SHORT → on vend les actifs avec le pire momentum (losers)
# Ce type de portefeuille est "market neutral" en théorie :
# il performe peu importe si le marché global monte ou descend.

# LONG_QUANTILE  = 0.80  # On va LONG sur le top 20% des actifs
# SHORT_QUANTILE = 0.20  # On va SHORT sur le bottom 20% des actifs avant opti

LONG_QUANTILE  = 0.70
SHORT_QUANTILE = 0.30 # après opti


# Capital initial pour le backtest et le paper trading
INITIAL_CAPITAL = 100_000  # 100 000 USD

# Taille maximale d'une position (en % du capital total)
# Règle de diversification : aucun actif ne peut représenter
# plus de 10% du portefeuille. Protège contre le risque
# idiosyncratique (risque spécifique à un seul actif).
MAX_POSITION_SIZE = 0.10  # 10%

# Levier maximum autorisé
# 1.0 = pas de levier (on ne peut pas investir plus que le capital)
# 2.0 = levier x2 (on peut investir 2x le capital disponible)
MAX_LEVERAGE = 1.5


# ─────────────────────────────────────────────────────────────
# SECTION 5 — PARAMÈTRES DU BACKTEST
# ─────────────────────────────────────────────────────────────

BACKTEST_START = "2015-01-01"
BACKTEST_END   = "2024-12-31"
# 10 ans de données → on capture plusieurs cycles de marché :
#   2015-2016 : volatilité, chine
#   2017-2019 : bull market
#   2020      : crash COVID + rebond violent
#   2021      : bull market post-covid
#   2022      : bear market (hausse des taux)
#   2023-2024 : rebond + AI boom
# Plus ton backtest couvre de régimes de marché différents,
# plus tes résultats sont crédibles.

# COÛTS DE TRANSACTION
# En BASIS POINTS (bps) : 1 bps = 0.01% = 0.0001
# 10 bps = 0.10% par trade (aller simple)
# C'est une estimation conservative pour des large caps liquides.
# En institution, on modélise aussi le MARKET IMPACT (le fait que
# ton propre ordre fait bouger le prix) — on l'ajoutera plus tard.
TRANSACTION_COST_BPS = 10   # 10 basis points par trade

# SLIPPAGE : différence entre le prix théorique et le prix réel d'exécution
# En backtest, on suppose qu'on trade au prix de CLÔTURE.
# En réalité, on trade à un prix légèrement moins favorable.
SLIPPAGE_BPS = 5  # 5 basis points de slippage estimé


# ─────────────────────────────────────────────────────────────
# SECTION 6 — PARAMÈTRES DE RISQUE
# ─────────────────────────────────────────────────────────────

# Taux sans risque annualisé — utilisé pour calculer le Sharpe Ratio
# Sharpe = (Rendement_portfolio - Taux_sans_risque) / Volatilité
# On utilise le taux des T-Bills américains 3 mois (~5% en 2024)
RISK_FREE_RATE = 0.05  # 5% annualisé

# STOP-LOSS au niveau du portefeuille
# Si le portefeuille perd plus de X% depuis son dernier sommet
# (drawdown), on arrête de trader et on sort toutes les positions.
MAX_PORTFOLIO_DRAWDOWN = 0.15  # 15% de drawdown maximum

# STOP-LOSS au niveau d'une position individuelle
# Si une position perd plus de X%, on la ferme immédiatement.
MAX_POSITION_LOSS = 0.15  # 15% de perte max par position

# Volatilité cible du portefeuille (annualisée)
# On va scaler les positions pour que la vol du portfolio soit
# proche de cette cible — technique appelée "vol targeting"
# Très utilisée en CTA et trend following.
TARGET_VOLATILITY = 0.15  # 15% de volatilité annualisée cible


# ─────────────────────────────────────────────────────────────
# SECTION 7 — DONNÉES HISTORIQUES
# ─────────────────────────────────────────────────────────────

# Type de données qu'on récupère
DATA_FREQUENCY = "1 day"   # Données journalières
                            # Format IBKR : "1 day", "1 hour", "5 mins"

# Durée d'historique à récupérer via l'API IBKR
# Format IBKR : "1 Y" = 1 an, "6 M" = 6 mois, "10 Y" = 10 ans
# ATTENTION : IBKR limite l'historique gratuit à ~1 an pour certains actifs
IBKR_HISTORY_DURATION = "10 Y"  # 10 ans d'historique

# Type de prix à utiliser
# "TRADES" = prix des transactions réelles (recommandé pour les actions)
# "MIDPOINT" = milieu du spread bid/ask (pour forex/futures)
IBKR_DATA_TYPE = "TRADES"

# Chemin local pour sauvegarder les données téléchargées
# On sauvegarde localement pour éviter de re-télécharger à chaque run
DATA_PATH = "./data/raw/"
PROCESSED_DATA_PATH = "./data/processed/"