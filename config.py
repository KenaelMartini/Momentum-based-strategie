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
# Profil aligné sur le run figé stats_20260323_190605 (identique à 191330 sur la courbe) :
# pas de snapshot git — reconstitué depuis les raisons de suspension dans ce CSV.
MAX_PORTFOLIO_DRAWDOWN = 0.15  # 15% — inchangé vs commit initial ; évite coupes « pic risk reset » trop tôt

# Réentrée après circuit breaker
SUSPENSION_COOLDOWN_CALENDAR_DAYS = 12
SUSPENSION_REENTRY_DD_FROM_EXIT = -0.05
SUSPENSION_REENTRY_FAST_CALENDAR_DAYS = 7
SUSPENSION_REENTRY_FAST_DD_FROM_EXIT = -0.015
SUSPENSION_REENTRY_REQUIRE_REGIME_CONFIRMATION = True
SUSPENSION_REENTRY_ALLOWED_RISK_REGIMES = ("BULL", "NORMAL")
SUSPENSION_REENTRY_MIN_CONSECUTIVE_RISK_DAYS = 3
SUSPENSION_REENTRY_RAMP_ENABLED = True
SUSPENSION_REENTRY_RAMP_SCALES = (0.35, 0.55, 0.75, 1.0)
# Désactivé sur 190605 : aucune suspension_reason POST_REENTRY_RECUT_* dans ce run.
SUSPENSION_POST_REENTRY_RECUT_ENABLED = False
SUSPENSION_POST_REENTRY_RECUT_SESSION_DAYS = 5
SUSPENSION_POST_REENTRY_RECUT_LOSS = 0.015
SUSPENSION_POST_REENTRY_RECUT_MAX_CALENDAR_WAIT_NO_INVEST = 45
# False sur le run de référence (défaut moteur risk) ; True modifie dates de fill / recut.
REBALANCE_FILL_SAME_BAR = False
# Désactivé sur 190605 : aucune REBALANCE_WINDOW_LOSS_CUT dans ce run.
REBALANCE_WINDOW_LOSS_CUT_ENABLED = False
REBALANCE_WINDOW_LOSS_CUT_SESSION_DAYS = 5
REBALANCE_WINDOW_LOSS_CUT_LOSS = 0.02
FAST_DRAWDOWN_CUT_ENABLED = True
FAST_DRAWDOWN_CUT_THRESHOLD = 0.065
FAST_DRAWDOWN_CUT_WINDOW_DAYS = 7
FAST_DRAWDOWN_CUT_WINDOW_DAYS_RISK_OFF = 7
# Horizon lent additionnel (dégradation progressive en stress/crise)
FAST_DRAWDOWN_CUT_THRESHOLD_LONG = 0.055
FAST_DRAWDOWN_CUT_WINDOW_DAYS_LONG = 10
# Confirmation anti-bruit : N jours consécutifs de breach avant suspension
FAST_DRAWDOWN_CUT_CONFIRM_DAYS = 2
# Profil C : pas de fast cut **court** en BULL/NORMAL (évite sorties type oct. 2021) ;
# l’horizon **long** reste actif dans tous les régimes (grind lent même si le risk est encore optimiste).
FAST_DRAWDOWN_CUT_ONLY_UNDER_STRESS = True

# Sous le pic global : si plus de N jours consécutifs en DD<0 ET DD <= seuil,
# on multiplie risk_scaling (réduction d’expo sans tout liquider).
PROLONGED_UNDERWATER_ENABLED = True
PROLONGED_UNDERWATER_MIN_DAYS = 12
PROLONGED_UNDERWATER_MIN_DD = -0.09
PROLONGED_UNDERWATER_RISK_SCALE_MULT = 0.92

# STOP-LOSS au niveau d'une position individuelle
# Si une position perd plus de X%, on la ferme immédiatement.
MAX_POSITION_LOSS = 0.15  # 15% de perte max par position

# Volatilité cible du portefeuille (annualisée)
# On va scaler les positions pour que la vol du portfolio soit
# proche de cette cible — technique appelée "vol targeting"
# Très utilisée en CTA et trend following.
TARGET_VOLATILITY = 0.15  # 15% de volatilité annualisée cible

# OVERLAY DU REGIME DE MARCHE
# Couche additionnelle pilotée par le moteur de regime "market".
ENABLE_MARKET_OVERLAY = False
MARKET_OVERLAY_MODE = "disabled"  # options: "disabled", "risk_off_only", "transition_and_risk_off"
TRANSITION_OVERLAY_ENABLED = False
RISK_OFF_MIN_SCALE = 0.15
RISK_OFF_MAX_SCALE = 0.35

# Seuil relatif min. sur |Δw| pour déclencher un trade au rebalancement mensuel.
REBALANCE_THRESHOLD_DEFAULT = 0.032

# Rééquilibrage robuste en phase défensive
# - En RISK_OFF: uniquement réduction de risque (pas d'augmentation d'exposition)
RISK_OFF_ONLY_DERISK_ENABLED = True
# Si un actif passe de long à short (ou inverse), on force l'exécution même si |Δw| < seuil.
REBALANCE_FORCE_SIGN_FLIP_EXECUTION = True

# Cible d'exposition nette (Σw) par régime de marché effectif.
# Permet d'éviter une dérive structurelle trop décorrélée en conservant la protection.
REGIME_NET_EXPOSURE_TARGET_ENABLED = True
REGIME_NET_TARGET_RISK_OFF_MIN = 0.0
REGIME_NET_TARGET_RISK_OFF_MAX = 0.15
REGIME_NET_TARGET_TRANSITION_MIN = 0.0
REGIME_NET_TARGET_TRANSITION_MAX = 0.40
REGIME_NET_TARGET_TREND_MIN = -1.0
REGIME_NET_TARGET_TREND_MAX = 1.0
REGIME_NET_TARGET_RISK_ON_MIN = -1.0
REGIME_NET_TARGET_RISK_ON_MAX = 1.0

# --- Régime marché (features) aligné sur BULL / STRESS / CRISIS (risk manager) ---
# Quand True, l’état utilisé pour les stats / perf par régime et les tilts ci-dessous
# combine le classifieur “macro” et le régime risque discret (plus de RISK_OFF crédible).
ALIGN_MARKET_REGIME_WITH_RISK = True

# Multiplicateur sur les poids après apply_regime_weight_filter, selon régime aligné.
# Activé : réduit fortement l’expo en RISK_OFF (aligné), où les stats montrent souvent
# beaucoup d’épisodes DD longs et un Sharpe faible.
# True + scale bas (0.45–0.65) = forte coupe en RISK_OFF aligné ; sur ce backtest le Max DD global peut empirer (path-dependent).
RISK_INFORMED_EXPOSURE_TILT_ENABLED = False
RISK_INFORMED_SCALE_RISK_OFF = 0.55
RISK_INFORMED_SCALE_TRANSITION_UNDER_STRESS = 0.90
# TREND riche en alpha : filet léger quand le risk voit déjà du STRESS.
RISK_INFORMED_SCALE_TREND_UNDER_STRESS = 0.95

# En phase TREND (régime aligné = TREND) : paliers DD portfolio supplémentaires + reprise.
TREND_DRAWDOWN_TILT_ENABLED = False
TREND_DD_MULT_LE5PCT = 0.97
TREND_DD_MULT_LE8PCT = 0.92
TREND_DD_MULT_LE12PCT = 0.84
TREND_RECOVERY_MULT = 1.05
TREND_RECOVERY_MAX_DD_FOR_BOOST = -0.008
TREND_RECOVERY_MIN_REGIME_SCORE = 0.82

# --- Flat défensif + réentrée par régime (hors circuit breaker DD max) ---
DEFENSIVE_FLAT_ENABLED = False
DEFENSIVE_FLAT_ENTRY_EFFECTIVE_REGIMES = ("RISK_OFF",)
DEFENSIVE_FLAT_ENTRY_MIN_DD = -0.055
DEFENSIVE_FLAT_ENTRY_MIN_CONSECUTIVE_DAYS = 4
DEFENSIVE_FLAT_MIN_CALENDAR_DAYS = 3
DEFENSIVE_FLAT_REENTRY_EFFECTIVE_REGIMES = ("TREND", "RISK_ON")
DEFENSIVE_FLAT_REENTRY_EFFECTIVE_CONSECUTIVE = 2
DEFENSIVE_FLAT_REENTRY_RISK_REGIMES = ("BULL", "NORMAL")
DEFENSIVE_FLAT_REENTRY_RISK_CONSECUTIVE = 2

# GARDE-FOUS D'ITERATION VS BASELINE
# PASS si tous les checks sont satisfaits.
BASELINE_MIN_ACCEPTED_SHARPE = 0.60
BASELINE_MAX_CAGR_DEGRADATION = 0.0025
BASELINE_MAX_MAX_DD_DEGRADATION = 0.0050
BASELINE_MAX_TURNOVER_INCREASE = 0.25


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
