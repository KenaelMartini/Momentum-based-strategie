# ============================================================
# signal.py — Signal Layer | Momentum Cross-Sectionnel & Time-Series
# ============================================================
# RÔLE DE CE FICHIER :
# Transformer une matrice de prix bruts en signaux de trading.
# C'est le "cerveau" de la stratégie — c'est ici qu'on calcule
# mathématiquement qui acheter et qui vendre.
#
# PIPELINE IMPLÉMENTÉE :
#   Prix → Log Returns → Momentum Brut → Score Composite
#   → Volatilité EWMA → Z-Score → Signal CS + TS → Signal Final
#
# DÉPENDANCES :
#   pip install pandas numpy scipy
# ============================================================

import os
import sys
import logging

import numpy as np
import pandas as pd
from scipy import stats  # Pour le ranking et les statistiques

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MOMENTUM_WINDOWS,
    MOMENTUM_WEIGHTS,
    SKIP_DAYS,
    LONG_QUANTILE,
    SHORT_QUANTILE,
    TARGET_VOLATILITY,
)

# Configuration du logger (même système que ibkr_data.py)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# CLASSE PRINCIPALE : MomentumSignalGenerator
# ============================================================
# On encapsule toute la logique du signal dans une classe.
# Elle prend en entrée la matrice des prix et produit en sortie
# les signaux de trading (qui acheter, qui vendre, avec quelle force).

class MomentumSignalGenerator:
    """
    Génère les signaux de trading momentum à partir d'une
    matrice de prix historiques.

    Implémente :
      - Cross-Sectional Momentum (CS-Mom) : ranking entre actifs
      - Time-Series Momentum (TS-Mom)     : chaque actif vs lui-même
      - Signal composite final            : combinaison des deux
    """

    def __init__(self, price_matrix: pd.DataFrame):
        """
        Initialise le générateur avec la matrice des prix.

        Args:
            price_matrix : DataFrame (dates × actifs) avec les prix de clôture
                           Exemple :
                                   AAPL    MSFT    GC
                           2020-01  294.6   160.6  1527.3
                           2020-01  296.2   158.9  1551.2
        """
        # Validation : on vérifie que la matrice n'est pas vide
        if price_matrix.empty:
            raise ValueError("❌ La matrice des prix est vide.")

        if len(price_matrix) < max(MOMENTUM_WINDOWS) + SKIP_DAYS:
            raise ValueError(
                f"❌ Pas assez de données. Minimum requis : "
                f"{max(MOMENTUM_WINDOWS) + SKIP_DAYS} jours. "
                f"Disponible : {len(price_matrix)} jours."
            )

        self.prices = price_matrix.copy()
        self.n_assets = price_matrix.shape[1]
        self.n_days   = price_matrix.shape[0]
        self.assets   = list(price_matrix.columns)

        # Ces attributs seront remplis au fur et à mesure
        # des calculs — on les initialise à None pour l'instant
        self.log_returns      = None  # Log returns journaliers
        self.momentum_raw     = None  # Momentum brut par fenêtre
        self.momentum_score   = None  # Score composite pondéré
        self.ewma_vol         = None  # Volatilité EWMA annualisée
        self.zscore           = None  # Z-score cross-sectionnel
        self.signal_cs        = None  # Signal Cross-Sectionnel
        self.signal_ts        = None  # Signal Time-Series
        self.signal_final     = None  # Signal final combiné

        logger.info(
            f"MomentumSignalGenerator initialisé : "
            f"{self.n_days} jours × {self.n_assets} actifs"
        )

    # ─────────────────────────────────────────────────────────
    # ÉTAPE 1 : Calcul des Log Returns
    # ─────────────────────────────────────────────────────────
    def compute_log_returns(self) -> pd.DataFrame:
        """
        Calcule les rendements logarithmiques journaliers.

        FORMULE :
            r(t) = ln(P_t / P_{t-1})

        POURQUOI LES LOG RETURNS ?
        Propriété d'additivité temporelle :
            r(t1→t3) = r(t1→t2) + r(t2→t3)
        Cette propriété est fausse avec les rendements simples.
        Elle est indispensable pour sommer des rendements sur
        plusieurs périodes (calcul du momentum multi-fenêtres).

        IMPLÉMENTATION :
        np.log(P_t / P_{t-1}) = np.log(P_t) - np.log(P_{t-1})
        pandas.DataFrame.pct_change() donne les rendements simples.
        On utilise np.log(df / df.shift(1)) pour les log returns.

        Returns:
            DataFrame des log returns (même dimensions que prices,
            première ligne = NaN car pas de P_{t-1} pour P_0)
        """
        # df.shift(1) décale le DataFrame d'une ligne vers le bas
        # → prix d'hier aligné avec le prix d'aujourd'hui
        # np.log(P_t / P_{t-1}) = log return journalier
        self.log_returns = np.log(self.prices / self.prices.shift(1))

        # La première ligne est NaN (pas de prix précédent pour le jour 1)
        # C'est normal et attendu — on ne la supprime pas ici car
        # les calculs suivants gèrent les NaN proprement

        logger.info(
            f"✅ Log returns calculés : "
            f"{self.log_returns.notna().all(axis=1).sum()} jours valides"
        )
        return self.log_returns

    # ─────────────────────────────────────────────────────────
    # ÉTAPE 2 : Calcul du Momentum Brut (multi-fenêtres)
    # ─────────────────────────────────────────────────────────
    def compute_raw_momentum(self) -> dict:
        """
        Calcule le momentum brut pour chaque fenêtre temporelle.

        FORMULE :
            Mom(i, t, n) = ln(P_{i, t-skip} / P_{i, t-n})

        Équivalent à la somme des log returns de t-n à t-skip :
            Mom(i, t, n) = Σ r(i, k) pour k de t-n à t-skip

        SKIP_DAYS = 21 : on exclut le dernier mois pour éviter
        le short-term reversal (Jegadeesh 1990).

        STRUCTURE DE SORTIE :
        Un dictionnaire avec une clé par fenêtre :
        {
            21:  DataFrame (dates × actifs) — momentum 1 mois
            63:  DataFrame (dates × actifs) — momentum 3 mois
            126: DataFrame (dates × actifs) — momentum 6 mois
            252: DataFrame (dates × actifs) — momentum 12 mois
        }

        Returns:
            dict { fenêtre: DataFrame momentum brut }
        """
        if self.log_returns is None:
            self.compute_log_returns()

        self.momentum_raw = {}

        for window in MOMENTUM_WINDOWS:
            # On calcule le rendement cumulé sur [t-window, t-skip]
            # En pratique : prix il y a SKIP_DAYS / prix il y a WINDOW jours
            #
            # prices.shift(SKIP_DAYS)  → prix d'il y a SKIP_DAYS jours
            # prices.shift(window)     → prix d'il y a WINDOW jours
            #
            # ln(P_{t-skip} / P_{t-window}) = momentum sur la fenêtre
            # en excluant les SKIP_DAYS derniers jours

            mom = np.log(
                self.prices.shift(SKIP_DAYS) /   # numérateur   : P_{t-skip}
                self.prices.shift(window)          # dénominateur : P_{t-window}
            )

            self.momentum_raw[window] = mom

            logger.info(
                f"  Fenêtre {window:3d}j : "
                f"{mom.notna().all(axis=1).sum()} dates valides calculées"
            )

        logger.info(f"✅ Momentum brut calculé pour {len(MOMENTUM_WINDOWS)} fenêtres")
        return self.momentum_raw

    # ─────────────────────────────────────────────────────────
    # ÉTAPE 3 : Score Composite Pondéré
    # ─────────────────────────────────────────────────────────
    def compute_momentum_score(self) -> pd.DataFrame:
        """
        Combine les momentum bruts de chaque fenêtre en UN score composite.

        FORMULE :
            MomScore(i, t) = Σ_j [ w_j × Mom(i, t, n_j) ]

        Avec Σ w_j = 1 (les poids somment à 1)
        Poids définis dans config.py : MOMENTUM_WEIGHTS

        POURQUOI PONDÉRER ?
        On donne plus de poids aux fenêtres longues (6-12 mois)
        car elles capturent le "vrai" momentum de fond vs le bruit
        des fenêtres courtes.

        Le score composite est plus STABLE et moins sensible
        au choix d'une fenêtre particulière. C'est l'équivalent
        d'une moyenne pondérée de plusieurs indicateurs.

        Returns:
            DataFrame du score composite (dates × actifs)
        """
        if self.momentum_raw is None:
            self.compute_raw_momentum()

        # On initialise la matrice du score à zéro
        # Elle aura la même forme que la matrice des prix
        self.momentum_score = pd.DataFrame(
            0.0,
            index=self.prices.index,
            columns=self.prices.columns
        )

        for window, weight in MOMENTUM_WEIGHTS.items():
            if window not in self.momentum_raw:
                logger.warning(f"⚠️ Fenêtre {window} manquante dans momentum_raw")
                continue

            mom_window = self.momentum_raw[window]

            # On remplace les NaN par 0 pour l'addition pondérée
            # (une fenêtre qui n'a pas assez de données ne pénalise pas le score)
            # fillna(0) : si on n'a pas assez d'historique pour cette fenêtre,
            # on ne la prend pas en compte (contribution nulle)
            self.momentum_score += weight * mom_window.fillna(0)

        # On remet NaN là où TOUTES les fenêtres étaient NaN
        # (dates trop proches du début de l'historique)
        # On utilise le masque de la fenêtre la plus longue
        longest_window = max(MOMENTUM_WINDOWS)
        valid_mask = self.momentum_raw[longest_window].notna().any(axis=1)
        self.momentum_score = self.momentum_score[valid_mask]

        logger.info(
            f"✅ Score composite calculé : "
            f"{len(self.momentum_score)} dates × {self.n_assets} actifs"
        )
        return self.momentum_score

    # ─────────────────────────────────────────────────────────
    # ÉTAPE 4 : Volatilité EWMA (Exponentially Weighted Moving Average)
    # ─────────────────────────────────────────────────────────
    def compute_ewma_volatility(self, lambda_: float = 0.94) -> pd.DataFrame:
        """
        Calcule la volatilité réalisée annualisée avec pondération EWMA.

        FORMULE EWMA (RiskMetrics — JP Morgan, 1994) :
            σ²(t) = λ × σ²(t-1) + (1-λ) × r²(t)

        Avec λ = 0.94 (standard industrie)

        POURQUOI EWMA ET PAS UNE SIMPLE MOYENNE MOBILE ?
        La volatilité des marchés est PERSISTANTE et RÉACTIVE :
        - Après un choc (ex: COVID crash), la vol reste élevée
        - Elle revient progressivement à la normale
        EWMA capte cette dynamique en donnant plus de poids
        aux observations récentes. λ = 0.94 signifie que
        la moitié du poids est sur les ~12 derniers jours.

        ANNUALISATION :
            σ_annuelle = σ_journalière × √252
        Facteur √252 car on suppose des rendements i.i.d.
        (indépendants et identiquement distribués) : la variance
        annuelle = 252 × variance journalière.

        Args:
            lambda_ : facteur de décroissance (0.94 = standard RiskMetrics)

        Returns:
            DataFrame des volatilités annualisées (dates × actifs)
        """
        if self.log_returns is None:
            self.compute_log_returns()

        # ewm() = Exponentially Weighted Moving average dans pandas
        # alpha = 1 - lambda_ (convention pandas)
        # adjust=False : utilise la formule récursive exacte (pas l'approximation)
        # min_periods=20 : on a besoin d'au moins 20 observations pour
        #                  avoir une estimation de variance fiable

        # On calcule la variance EWMA des rendements journaliers
        ewma_variance = (
            self.log_returns
            .fillna(0)                      # NaN → 0 pour le calcul EWMA
            .pow(2)                          # r² (variance des rendements)
            .ewm(
                alpha=1 - lambda_,           # alpha = 1 - λ = 0.06
                adjust=False,                # Formule récursive exacte
                min_periods=20               # Minimum 20 observations
            )
            .mean()                          # Moyenne EWMA des r²
        )

        # Annualisation : σ_annuelle = √(252 × σ²_journalière)
        # On multiplie la VARIANCE par 252, puis on prend la racine
        self.ewma_vol = np.sqrt(ewma_variance * 252)

        # On aligne l'index avec le momentum_score
        if self.momentum_score is not None:
            self.ewma_vol = self.ewma_vol.reindex(self.momentum_score.index)

        # Sécurité : on remplace les volatilités nulles par un minimum
        # Une vol nulle causerait une division par zéro plus tard
        # 1% de vol annuelle est un plancher raisonnable
        self.ewma_vol = self.ewma_vol.clip(lower=0.01)

        logger.info(
            f"✅ Volatilité EWMA calculée (λ={lambda_}) | "
            f"Vol moyenne : {self.ewma_vol.mean().mean():.1%}"
        )
        return self.ewma_vol

    # ─────────────────────────────────────────────────────────
    # ÉTAPE 5 : Z-Score Cross-Sectionnel
    # ─────────────────────────────────────────────────────────
    def compute_cross_sectional_zscore(self) -> pd.DataFrame:
        """
        Normalise le score momentum avec un z-score cross-sectionnel.

        FORMULE :
            z(i, t) = (MomScore(i, t) - μ(t)) / σ(t)

        Où μ(t) et σ(t) sont calculés SUR TOUS LES ACTIFS à la date t.
        C'est un z-score "cross-sectionnel" (en coupe transversale),
        pas temporel.

        EXEMPLE :
        À la date t, on a 20 actifs avec leurs MomScores.
        μ(t) = moyenne des 20 scores
        σ(t) = écart-type des 20 scores
        z(AAPL, t) = (MomScore(AAPL) - μ(t)) / σ(t)

        INTERPRÉTATION :
        z = +2.0 → AAPL est à 2 écarts-types AU-DESSUS de la moyenne
                  → Fort momentum relatif → Signal d'achat fort
        z = -1.5 → Actif 1.5 σ en dessous → Momentum faible → Signal de vente

        POURQUOI NORMALISER ?
        Sans normalisation, les actifs très volatils (crypto, small caps)
        auraient toujours des scores plus élevés en valeur absolue.
        Le z-score rend les scores COMPARABLES entre actifs
        de volatilité différente.

        Returns:
            DataFrame des z-scores (dates × actifs)
        """
        if self.momentum_score is None:
            self.compute_momentum_score()

        # axis=1 → on calcule la moyenne et l'écart-type
        # sur les COLONNES (actifs) pour chaque ligne (date)
        # C'est le calcul cross-sectionnel

        cross_mean = self.momentum_score.mean(axis=1)   # μ(t) : vecteur de dates
        cross_std  = self.momentum_score.std(axis=1)    # σ(t) : vecteur de dates

        # On évite la division par zéro si tous les actifs ont le même score
        cross_std = cross_std.replace(0, np.nan)

        # Soustraction et division vectorisées
        # .sub(cross_mean, axis=0) : soustrait la moyenne de chaque ligne
        # .div(cross_std,  axis=0) : divise par l'écart-type de chaque ligne
        self.zscore = (
            self.momentum_score
            .sub(cross_mean, axis=0)   # MomScore(i,t) - μ(t)
            .div(cross_std,  axis=0)   # / σ(t)
        )

        # On clippe les z-scores extrêmes à ±3σ
        # Un z-score > 3 est statistiquement très rare (0.3% des cas).
        # En pratique, des valeurs > 3 indiquent souvent une anomalie
        # de données plutôt qu'un vrai signal. On les plafonne.
        self.zscore = self.zscore.clip(lower=-3.0, upper=3.0)

        logger.info(
            f"✅ Z-scores calculés | "
            f"Moyenne : {self.zscore.mean().mean():.3f} | "
            f"Écart-type : {self.zscore.std().mean():.3f}"
        )
        return self.zscore

    # ─────────────────────────────────────────────────────────
    # ÉTAPE 6A : Signal Cross-Sectionnel (CS-Mom)
    # ─────────────────────────────────────────────────────────
    def compute_cross_sectional_signal(self) -> pd.DataFrame:
        """
        Génère le signal Cross-Sectionnel à partir du ranking des z-scores.

        LOGIQUE :
        À chaque date :
          1. On classe tous les actifs par z-score (du plus grand au plus petit)
          2. Top LONG_QUANTILE  (20%) → signal +1 (LONG)
          3. Bot SHORT_QUANTILE (20%) → signal -1 (SHORT)
          4. Milieu (60%)             → signal  0 (neutre, pas de position)

        CE TYPE DE PORTEFEUILLE EST "MARKET NEUTRAL" :
        La somme des positions long et short s'annule approximativement.
        Le portefeuille ne dépend pas de la direction du marché global.
        Il performe si les "winners" surperforment les "losers",
        indépendamment que le marché monte ou descende.

        C'est ce qu'on appelle un "long/short equity" ou "dollar neutral
        portfolio" — très prisé en hedge fund.

        Returns:
            DataFrame des signaux CS : valeurs dans {-1, 0, +1}
        """
        if self.zscore is None:
            self.compute_cross_sectional_zscore()

        # On initialise la matrice des signaux à 0 (neutre)
        self.signal_cs = pd.DataFrame(
            0.0,
            index=self.zscore.index,
            columns=self.zscore.columns
        )

        # Pour chaque date (ligne), on calcule les quantiles
        # et on assigne les signaux +1 / -1
        for date in self.zscore.index:
            scores_today = self.zscore.loc[date].dropna()

            if len(scores_today) < 5:
                # Pas assez d'actifs pour avoir un ranking significatif
                continue

            # Calcul des seuils de quantiles
            # quantile(0.80) = valeur en dessous de laquelle se trouvent
            # 80% des actifs → tout ce qui est AU-DESSUS est le top 20%
            upper_threshold = scores_today.quantile(LONG_QUANTILE)   # top 20%
            lower_threshold = scores_today.quantile(SHORT_QUANTILE)  # bot 20%

            # Attribution des signaux
            for asset in scores_today.index:
                score = scores_today[asset]
                if score >= upper_threshold:
                    self.signal_cs.loc[date, asset] = 1.0    # LONG
                elif score <= lower_threshold:
                    self.signal_cs.loc[date, asset] = -1.0   # SHORT
                # else : reste à 0 (neutre)

        # Statistiques du signal pour validation
        long_pct  = (self.signal_cs == 1).mean().mean()
        short_pct = (self.signal_cs == -1).mean().mean()
        logger.info(
            f"✅ Signal CS calculé | "
            f"Long: {long_pct:.1%} | Short: {short_pct:.1%} | "
            f"Neutre: {1-long_pct-short_pct:.1%}"
        )
        return self.signal_cs

    # ─────────────────────────────────────────────────────────
    # ÉTAPE 6B : Signal Time-Series (TS-Mom)
    # ─────────────────────────────────────────────────────────
    def compute_time_series_signal(self) -> pd.DataFrame:
        """
        Génère le signal Time-Series avec vol scaling.

        FORMULE :
            Signal_TS(i, t) = MomScore(i, t) / σ_EWMA(i, t) × (σ_target / σ_cross)

        DÉCOMPOSITION :
          1. sign(MomScore) → direction : long si momentum positif, short si négatif
          2. |MomScore| / σ_EWMA → force du signal ajustée par la volatilité
          3. × σ_target → on scale pour que la contribution au risque soit uniforme

        VOL SCALING — Pourquoi c'est crucial :
        Sans vol scaling, un actif très volatil (ex: Gold) aurait une
        contribution au risque bien plus grande qu'un actif calme (ex: JPM).
        Le vol scaling assure que CHAQUE actif contribue ÉGALEMENT au risque
        du portefeuille. C'est le standard dans tous les CTA.

        Le signal TS est DIRECTIONNEL (pas market neutral) :
        Si tous les actifs ont un momentum positif → on est Long sur tout.
        Si le marché chute → on est Short sur tout.
        Ce signal capture les grandes tendances de marché.

        Returns:
            DataFrame des signaux TS (valeurs continues, pas discrètes)
        """
        if self.momentum_score is None:
            self.compute_momentum_score()
        if self.ewma_vol is None:
            self.compute_ewma_volatility()

        # Alignement des index (sécurité)
        common_index = self.momentum_score.index.intersection(self.ewma_vol.index)
        mom_aligned = self.momentum_score.reindex(common_index)
        vol_aligned = self.ewma_vol.reindex(common_index)

        # Signal brut = score momentum / volatilité EWMA
        # On normalise le momentum par la volatilité de l'actif
        # → donne une mesure de "rendement ajusté au risque" du momentum
        raw_ts_signal = mom_aligned / vol_aligned

        # Vol scaling vers la volatilité cible
        # On veut que le signal moyen corresponde à TARGET_VOLATILITY (15%)
        # La volatilité cross-sectionnelle du signal à chaque date
        signal_vol = raw_ts_signal.std(axis=1)  # vol du signal au travers des actifs
        signal_vol = signal_vol.replace(0, np.nan).ffill()

        # Scaling : on multiplie par (TARGET_VOL / vol_signal)
        # pour que l'amplitude du signal corresponde à notre cible
        scaling_factor = TARGET_VOLATILITY / signal_vol

        self.signal_ts = raw_ts_signal.mul(scaling_factor, axis=0)

        # On clippe les signaux extrêmes (protection)
        # Un signal de 3 signifie déjà 3x la volatilité cible
        self.signal_ts = self.signal_ts.clip(lower=-3.0, upper=3.0)

        logger.info(
            f"✅ Signal TS calculé | "
            f"Signal moyen: {self.signal_ts.mean().mean():.3f} | "
            f"Actifs long: {(self.signal_ts > 0).mean().mean():.1%}"
        )
        return self.signal_ts

    # ─────────────────────────────────────────────────────────
    # ÉTAPE 7 : Signal Final Combiné
    # ─────────────────────────────────────────────────────────
    def compute_final_signal(
        self,
        cs_weight: float = 0.5,
        ts_weight: float = 0.5
    ) -> pd.DataFrame:
        """
        Combine les signaux CS et TS en un signal final unique.

        FORMULE :
            Signal_Final = cs_weight × Signal_CS + ts_weight × Signal_TS

        POURQUOI COMBINER CS ET TS ?
        Les deux signaux sont COMPLÉMENTAIRES :

        CS-Mom : capture les DIVERGENCES entre actifs
                 → "AAPL surperforme MSFT en ce moment"
                 → Market neutral, faible drawdown en marché baissier
                 → Mais rate les grandes tendances directionnelles

        TS-Mom : capture les TENDANCES DIRECTIONNELLES
                 → "L'or est en tendance haussière depuis 6 mois"
                 → Directionnel, capture les mega-trends
                 → Mais peut souffrir lors de retournements brusques

        Combinés à 50/50 : on capture à la fois les divergences
        relatives ET les tendances absolues → meilleur Sharpe Ratio.

        La recherche de Moskowitz, Ooi & Pedersen (2012) montre que
        TS et CS se complètent efficacement en portefeuille.

        Args:
            cs_weight : poids du signal cross-sectionnel (défaut 0.5)
            ts_weight : poids du signal time-series (défaut 0.5)

        Returns:
            DataFrame du signal final combiné
        """
        # Calcul de tous les composants si pas encore fait
        if self.signal_cs is None:
            self.compute_cross_sectional_signal()
        if self.signal_ts is None:
            self.compute_time_series_signal()

        # Validation des poids
        if abs(cs_weight + ts_weight - 1.0) > 1e-6:
            raise ValueError(
                f"❌ Les poids CS ({cs_weight}) + TS ({ts_weight}) "
                f"doivent sommer à 1.0"
            )

        # Alignement sur l'index commun
        common_index = self.signal_cs.index.intersection(self.signal_ts.index)

        cs_aligned = self.signal_cs.reindex(common_index)
        ts_aligned = self.signal_ts.reindex(common_index)

        # Combinaison pondérée
        self.signal_final = (
            cs_weight * cs_aligned +
            ts_weight * ts_aligned
        )

        # Normalisation finale du signal composite
        # On re-normalise pour que les signaux restent dans [-1, +1]
        # en divisant par la valeur absolue maximale par ligne
        # Cela garantit que le signal final a la même amplitude
        # que les signaux individuels
        row_max = self.signal_final.abs().max(axis=1).replace(0, 1)
        self.signal_final = self.signal_final.div(row_max, axis=0)

        logger.info(
            f"✅ Signal final combiné (CS:{cs_weight:.0%} / TS:{ts_weight:.0%}) | "
            f"Valeur moyenne: {self.signal_final.mean().mean():.3f} | "
            f"Actifs avec signal: "
            f"{(self.signal_final.abs() > 0.1).mean().mean():.1%}"
        )
        return self.signal_final

    # ─────────────────────────────────────────────────────────
    # MÉTHODE UTILITAIRE : Pipeline complète en une seule ligne
    # ─────────────────────────────────────────────────────────
    def run_full_pipeline(
        self,
        cs_weight: float = 0.5,
        ts_weight: float = 0.5
    ) -> dict:
        """
        Exécute toute la pipeline du signal en une seule commande.

        Cette méthode appelle séquentiellement toutes les étapes
        dans le bon ordre. C'est la méthode qu'on appellera depuis
        le backtest.

        Args:
            cs_weight : poids signal cross-sectionnel
            ts_weight : poids signal time-series

        Returns:
            dict avec tous les résultats intermédiaires et finaux
        """
        logger.info("🚀 Lancement de la pipeline signal complète...")

        # Étape 1 : Log returns
        self.compute_log_returns()

        # Étape 2 : Momentum brut (toutes fenêtres)
        self.compute_raw_momentum()

        # Étape 3 : Score composite pondéré
        self.compute_momentum_score()

        # Étape 4 : Volatilité EWMA
        self.compute_ewma_volatility()

        # Étape 5 : Z-score cross-sectionnel
        self.compute_cross_sectional_zscore()

        # Étape 6A : Signal Cross-Sectionnel
        self.compute_cross_sectional_signal()

        # Étape 6B : Signal Time-Series
        self.compute_time_series_signal()

        # Étape 7 : Signal Final
        self.compute_final_signal(cs_weight, ts_weight)

        logger.info("✅ Pipeline signal complète !")

        # On retourne tous les résultats dans un dict structuré
        return {
            "log_returns"    : self.log_returns,
            "momentum_raw"   : self.momentum_raw,
            "momentum_score" : self.momentum_score,
            "ewma_vol"       : self.ewma_vol,
            "zscore"         : self.zscore,
            "signal_cs"      : self.signal_cs,
            "signal_ts"      : self.signal_ts,
            "signal_final"   : self.signal_final,
        }

    # ─────────────────────────────────────────────────────────
    # MÉTHODE UTILITAIRE : Résumé du signal à une date donnée
    # ─────────────────────────────────────────────────────────
    def get_signal_snapshot(self, date: str = None) -> pd.DataFrame:
        """
        Affiche un résumé lisible du signal pour une date donnée.

        Très utile pour déboguer et comprendre ce que fait le signal
        à un moment précis. En institution, ce genre de "snapshot"
        est utilisé pour le monitoring quotidien de la stratégie.

        Args:
            date : date au format "YYYY-MM-DD" (défaut : dernière date)

        Returns:
            DataFrame résumé avec toutes les métriques par actif
        """
        if self.signal_final is None:
            logger.error("❌ Pipeline non exécutée. Lancer run_full_pipeline() d'abord.")
            return pd.DataFrame()

        # Dernière date si non spécifiée
        if date is None:
            date = self.signal_final.index[-1]
        else:
            date = pd.to_datetime(date)
            # On trouve la date la plus proche si la date exacte n'existe pas
            date = self.signal_final.index[
                self.signal_final.index.get_indexer([date], method="nearest")[0]
            ]

        # Construction du tableau résumé
        snapshot = pd.DataFrame({
            "MomScore"     : self.momentum_score.loc[date],
            "Z-Score"      : self.zscore.loc[date],
            "Vol_EWMA"     : self.ewma_vol.loc[date],
            "Signal_CS"    : self.signal_cs.loc[date],
            "Signal_TS"    : self.signal_ts.loc[date],
            "Signal_Final" : self.signal_final.loc[date],
        })

        # Ajout d'une colonne "Direction" lisible
        snapshot["Direction"] = snapshot["Signal_Final"].apply(
            lambda x: "🟢 LONG" if x > 0.1 else ("🔴 SHORT" if x < -0.1 else "⚪ NEUTRE")
        )

        # Tri par signal final décroissant (strongest first)
        snapshot = snapshot.sort_values("Signal_Final", ascending=False)

        print(f"\n📊 Snapshot du signal au {date.date()}")
        print("=" * 75)
        print(snapshot.round(4).to_string())
        print("=" * 75)

        return snapshot


# ============================================================
# SCRIPT PRINCIPAL — Test du signal avec données simulées
# ============================================================
# En attendant la connexion IBKR, on teste la pipeline avec
# des données synthétiques réalistes pour valider le code.

if __name__ == "__main__":

    print("=" * 60)
    print("  TEST — MomentumSignalGenerator")
    print("=" * 60)

    # ── Génération de données synthétiques réalistes ──────────
    # On simule 3 ans de prix journaliers pour 10 actifs
    # avec un mouvement brownien géométrique (GBM) :
    # dP = μ × P × dt + σ × P × dW
    # C'est le modèle de Black-Scholes pour les prix d'actions.

    np.random.seed(42)  # Seed fixe pour la reproductibilité

    n_days   = 756   # ~3 ans de trading (252 × 3)
    n_assets = 10    # 10 actifs simulés

    assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
              "GC", "CL", "JPM", "XOM", "NVDA"]

    # Paramètres du GBM pour chaque actif
    # mu = rendement annuel moyen, sigma = volatilité annuelle
    mu    = 0.08   # 8% de rendement annuel moyen
    sigma = 0.20   # 20% de volatilité annuelle

    # Génération des rendements journaliers
    # dt = 1/252 (un jour de trading)
    daily_returns = np.random.normal(
        loc   = mu / 252,           # rendement journalier moyen
        scale = sigma / np.sqrt(252),  # vol journalière
        size  = (n_days, n_assets)
    )

    # Conversion en prix via produit cumulatif
    # P(t) = P(0) × exp(Σ r(k)) — formule du GBM
    prices_array = 100 * np.exp(np.cumsum(daily_returns, axis=0))

    # Création du DataFrame avec dates et noms d'actifs
    dates = pd.bdate_range(start="2021-01-01", periods=n_days)  # jours ouvrés
    price_matrix = pd.DataFrame(prices_array, index=dates, columns=assets)

    print(f"\n📊 Données simulées : {price_matrix.shape}")
    print(price_matrix.tail(3).round(2))

    # ── Exécution de la pipeline complète ─────────────────────
    print("\n🚀 Lancement de la pipeline signal...")
    generator = MomentumSignalGenerator(price_matrix)
    results   = generator.run_full_pipeline(cs_weight=0.5, ts_weight=0.5)

    # ── Affichage du snapshot final ───────────────────────────
    snapshot = generator.get_signal_snapshot()

    # ── Statistiques de validation ────────────────────────────
    print("\n📈 Statistiques du signal final :")
    print(f"  Nombre de dates avec signal : {len(results['signal_final'])}")
    print(f"  % actifs LONG  : {(results['signal_final'] > 0.1).mean().mean():.1%}")
    print(f"  % actifs SHORT : {(results['signal_final'] < -0.1).mean().mean():.1%}")
    print(f"  % actifs NEUTRE: {(results['signal_final'].abs() <= 0.1).mean().mean():.1%}")
    print(f"\n  Volatilité moyenne du portefeuille (signal TS) :")
    vol_by_asset = results["ewma_vol"].mean()
    print(vol_by_asset.round(3).to_string())