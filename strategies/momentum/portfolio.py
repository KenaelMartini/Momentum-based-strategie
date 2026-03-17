# ============================================================
# portfolio.py — Portfolio Construction Layer
# ============================================================
# RÔLE DE CE FICHIER :
# Transformer les signaux abstraits [-1, +1] en positions
# concrètes : nombre d'actions, nombre de contrats futures,
# montants en dollars.
#
# PIPELINE :
#   Signal [-1,+1] → Vol Parity Weights → Normalisation
#   → Contraintes → Dollar Positions → Titres/Contrats
#   → Rebalancing Check → Positions Finales
#
# DÉPENDANCES :
#   pip install pandas numpy
# ============================================================

import os
import sys
import logging
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    INITIAL_CAPITAL,
    MAX_POSITION_SIZE,
    MAX_LEVERAGE,
    TARGET_VOLATILITY,
    TRANSACTION_COST_BPS,
    SLIPPAGE_BPS,
    REBALANCING_FREQUENCY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTES — FUTURES MULTIPLIERS
# ============================================================
# Chaque contrat future contrôle une quantité fixe du sous-jacent.
# Le "multiplier" est la valeur d'un point de prix en dollars.
#
# EXEMPLE CONCRET :
#   CL (Crude Oil) : 1 contrat = 1000 barils
#   Si le prix du baril = 80$ → 1 contrat vaut 80 000$
#   Si le prix monte de 1$ → le contrat gagne 1000$
#
# Ces multipliers sont FIXES et définis par les exchanges.
# Source : CME Group specifications

FUTURES_MULTIPLIERS = {
    "CL": 1000,   # Crude Oil    — 1000 barils    — NYMEX
    "NG": 10000,  # Natural Gas  — 10 000 MMBTU   — NYMEX
    "GC": 100,    # Gold         — 100 onces troy  — COMEX
    "SI": 5000,   # Silver       — 5000 onces troy — COMEX
    "HG": 25000,  # Copper       — 25 000 livres   — COMEX
    "ZC": 5000,   # Corn         — 5000 boisseaux  — CBOT
    "ZW": 5000,   # Wheat        — 5000 boisseaux  — CBOT
    "ZS": 5000,   # Soybean      — 5000 boisseaux  — CBOT
}

# Seuil minimum de rebalancement
# On ne retrade une position que si le changement dépasse ce % du capital
# En dessous, les frais de transaction ne valent pas le coup
REBALANCING_THRESHOLD = 0.02  # 2% du capital minimum pour retrader


# ============================================================
# CLASSE PRINCIPALE : PortfolioConstructor
# ============================================================

class PortfolioConstructor:
    """
    Construit et gère le portefeuille momentum.

    Prend en entrée :
      - Les signaux finaux (output de MomentumSignalGenerator)
      - Les prix courants
      - La volatilité EWMA (output de MomentumSignalGenerator)

    Produit en sortie :
      - Les poids cibles du portefeuille
      - Les positions en dollars
      - Les positions en nombre de titres/contrats
      - Le rapport de rebalancement (ce qu'il faut trader)
    """

    def __init__(self, capital: float = INITIAL_CAPITAL):
        """
        Initialise le constructeur de portefeuille.

        Args:
            capital : capital total disponible en USD
        """
        self.capital = capital

        # Positions ACTUELLES du portefeuille
        # Dictionnaire : { symbol: nb_titres } (positif=long, négatif=short)
        self.current_positions = {}

        # Poids ACTUELS du portefeuille
        # Dictionnaire : { symbol: poids } (entre -1 et +1)
        self.current_weights = {}

        # Historique complet des portefeuilles au fil du temps
        # Liste de DataFrames — un par date de rebalancement
        self.portfolio_history = []

        # Historique des trades effectués
        self.trades_history = []

        logger.info(
            f"PortfolioConstructor initialisé | "
            f"Capital: {capital:,.0f}$ | "
            f"Max position: {MAX_POSITION_SIZE:.0%} | "
            f"Max levier: {MAX_LEVERAGE:.1f}x"
        )

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 1 : Calcul des poids Vol Parity
    # ─────────────────────────────────────────────────────────
    def compute_vol_parity_weights(
        self,
        signals: pd.Series,
        volatilities: pd.Series
    ) -> pd.Series:
        """
        Calcule les poids du portefeuille par Vol Parity Weighting.

        FORMULE :
            w_i = signal_i × (σ_target / σ_i)

        INTUITION :
        On veut que chaque actif contribue ÉGALEMENT au risque.
        Si AAPL a une vol de 20% et XOM une vol de 10% :
          → XOM reçoit 2x plus de capital qu'AAPL
          → Car XOM est 2x moins risqué par dollar investi
          → Résultat : les deux contribuent ~autant au risque

        C'est différent de l'equal weighting (même $ sur tout)
        et du market cap weighting ($ proportionnel à la taille).
        Le vol parity est le standard dans les stratégies CTA.

        Args:
            signals      : Series { symbol: signal [-1,+1] }
            volatilities : Series { symbol: vol annualisée }

        Returns:
            Series des poids bruts (avant normalisation)
        """
        # Alignement — on garde seulement les actifs présents dans les deux
        common_assets = signals.index.intersection(volatilities.index)
        signals      = signals[common_assets]
        volatilities = volatilities[common_assets]

        # Protection contre les volatilités nulles ou manquantes
        # clip(lower=0.01) → vol minimum de 1% pour éviter division par zéro
        volatilities = volatilities.fillna(volatilities.median()).clip(lower=0.01)

        # Calcul des poids bruts
        # signal_i × (σ_target / σ_i)
        # Un actif avec signal=+1 et vol=20% reçoit : 1 × (15%/20%) = 0.75
        # Un actif avec signal=+1 et vol=10% reçoit : 1 × (15%/10%) = 1.50
        raw_weights = signals * (TARGET_VOLATILITY / volatilities)

        return raw_weights

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 2 : Normalisation et contraintes
    # ─────────────────────────────────────────────────────────
    def apply_constraints(self, raw_weights: pd.Series) -> pd.Series:
        """
        Applique les contraintes de risque sur les poids bruts.

        CONTRAINTES APPLIQUÉES DANS L'ORDRE :

        1. POSITION CAP : aucune position > MAX_POSITION_SIZE (10%)
           → Diversification minimale, protection contre le risque
             idiosyncratique (risque spécifique à un actif)

        2. LEVERAGE NORMALIZATION : gross exposure ≤ MAX_LEVERAGE (1.5x)
           → Si la somme des |poids| dépasse 1.5, on scale tout à la baisse
           → Contrôle du levier total du portefeuille

        ORDRE IMPORTANT :
        On applique le position cap AVANT la normalisation du levier.
        Pourquoi ? Si on normalisait d'abord, le position cap pourrait
        être violé après normalisation sur certains actifs.

        Args:
            raw_weights : Series des poids bruts (vol parity)

        Returns:
            Series des poids contraints et normalisés
        """
        weights = raw_weights.copy()

        # --- CONTRAINTE 1 : Position cap individuelle ---
        # clip(-MAX, +MAX) plafonne chaque poids entre -10% et +10%
        weights = weights.clip(
            lower=-MAX_POSITION_SIZE,
            upper=MAX_POSITION_SIZE
        )

        # --- CONTRAINTE 2 : Normalisation du levier ---
        # gross_exposure = somme des valeurs absolues des poids
        # = levier total du portefeuille
        gross_exposure = weights.abs().sum()

        if gross_exposure > MAX_LEVERAGE:
            # On scale tous les poids proportionnellement
            # pour que leur somme absolue = MAX_LEVERAGE
            # Chaque poids est multiplié par MAX_LEVERAGE / gross_exposure
            scaling = MAX_LEVERAGE / gross_exposure
            weights = weights * scaling
            logger.debug(
                f"  Levier réduit : {gross_exposure:.2f}x → {MAX_LEVERAGE:.2f}x "
                f"(scaling: {scaling:.3f})"
            )

        # --- VÉRIFICATION FINALE ---
        # On s'assure que toutes les contraintes sont respectées
        assert weights.abs().max() <= MAX_POSITION_SIZE + 1e-6, \
            "❌ Position cap violé après contraintes"
        assert weights.abs().sum() <= MAX_LEVERAGE + 1e-6, \
            "❌ Levier maximum violé après contraintes"

        return weights

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 3 : Conversion poids → dollars → titres
    # ─────────────────────────────────────────────────────────
    def weights_to_positions(
        self,
        weights: pd.Series,
        prices: pd.Series,
        asset_types: dict = None
    ) -> pd.DataFrame:
        """
        Convertit les poids en positions concrètes.

        ÉTAPES :
          1. Dollar position = poids × capital
          2. Pour les STOCKS  : nb_titres = dollar / prix
          3. Pour les FUTURES : nb_contrats = dollar / (prix × multiplier)

        ARRONDI :
        On ne peut pas acheter 127.3 actions — on arrondit.
        Pour les stocks  : floor() car on ne veut pas dépasser le capital
        Pour les futures : round() car on peut aller dans les deux sens

        RÉSIDU DE CASH :
        L'arrondi crée un petit résidu de cash non investi.
        Ex : on voulait 127.3 actions → on en achète 127
        → Le résidu = 0.3 × prix reste en cash
        C'est normal et inévitable — on le track mais on l'ignore.

        Args:
            weights     : Series des poids contraints { symbol: poids }
            prices      : Series des prix { symbol: prix }
            asset_types : dict { symbol: "stock" ou "future" }
                          Si None, tous traités comme stocks

        Returns:
            DataFrame avec colonnes :
            [weight, dollar_position, shares/contracts, price, asset_type]
        """
        if asset_types is None:
            asset_types = {s: "stock" for s in weights.index}

        positions = []

        for symbol in weights.index:
            if symbol not in prices.index:
                logger.warning(f"⚠️ Prix manquant pour {symbol} — ignoré")
                continue

            weight     = weights[symbol]
            price      = prices[symbol]
            asset_type = asset_types.get(symbol, "stock")

            # Position en dollars
            # Positif = long (on achète), Négatif = short (on vend)
            dollar_position = weight * self.capital

            if asset_type == "future":
                # Pour les futures : on divise par (prix × multiplier)
                # Le multiplier convertit le prix en valeur notionnelle
                multiplier = FUTURES_MULTIPLIERS.get(symbol, 1)
                notional_per_contract = price * multiplier

                if notional_per_contract == 0:
                    logger.warning(f"⚠️ Valeur notionnelle nulle pour {symbol}")
                    continue

                # round() pour les futures (on peut être long ou short)
                nb_contracts = round(dollar_position / notional_per_contract)
                actual_dollar = nb_contracts * notional_per_contract

                positions.append({
                    "symbol"         : symbol,
                    "asset_type"     : "future",
                    "weight"         : weight,
                    "dollar_target"  : dollar_position,
                    "dollar_actual"  : actual_dollar,
                    "quantity"       : nb_contracts,   # nombre de contrats
                    "price"          : price,
                    "multiplier"     : multiplier,
                    "notional"       : actual_dollar,
                })

            else:  # stock
                if price == 0:
                    logger.warning(f"⚠️ Prix nul pour {symbol}")
                    continue

                # int() tronque vers zéro (floor pour positif, ceil pour négatif)
                # On utilise math.floor pour les longs et math.ceil pour les shorts
                if dollar_position >= 0:
                    nb_shares = int(dollar_position / price)    # Long : on tronque
                else:
                    nb_shares = -int(abs(dollar_position) / price)  # Short : idem

                actual_dollar = nb_shares * price

                positions.append({
                    "symbol"        : symbol,
                    "asset_type"    : "stock",
                    "weight"        : weight,
                    "dollar_target" : dollar_position,
                    "dollar_actual" : actual_dollar,
                    "quantity"      : nb_shares,        # nombre d'actions
                    "price"         : price,
                    "multiplier"    : 1,
                    "notional"      : actual_dollar,
                })

        # Conversion en DataFrame pour lisibilité
        positions_df = pd.DataFrame(positions).set_index("symbol")

        # Calcul du cash résiduel (capital non investi à cause des arrondis)
        total_invested = positions_df["dollar_actual"].sum()
        cash_residual  = self.capital - abs(positions_df["dollar_actual"]).sum()

        logger.info(
            f"💼 Positions calculées : {len(positions_df)} actifs | "
            f"Capital investi: {abs(total_invested)/self.capital:.1%} | "
            f"Cash résiduel: {cash_residual:,.0f}$"
        )

        return positions_df

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 4 : Calcul du rebalancement
    # ─────────────────────────────────────────────────────────
    def compute_rebalancing_trades(
        self,
        target_positions: pd.DataFrame,
        current_prices: pd.Series
    ) -> pd.DataFrame:
        """
        Calcule les trades nécessaires pour passer des positions
        actuelles aux positions cibles.

        LOGIQUE :
          delta = position_cible - position_actuelle
          Si |delta × prix| > seuil_rebalancement × capital → on trade
          Sinon → on garde la position actuelle (trop petit pour valoir le coup)

        SEUIL DE REBALANCEMENT :
        On ne retrade pas pour de petites variations.
        Si le changement de position < 2% du capital → on ignore.
        Pourquoi ? Les frais de transaction (10bps) + slippage (5bps)
        = 15bps par trade. Si le trade ne génère pas au moins l'équivalent
        en valeur ajoutée, il détruit de la valeur.

        COÛT DE TRANSACTION :
        On estime le coût de chaque trade :
          Coût = |dollar_traded| × (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000

        Args:
            target_positions : DataFrame des positions cibles (output weights_to_positions)
            current_prices   : Series des prix actuels

        Returns:
            DataFrame des trades à effectuer avec leurs coûts estimés
        """
        trades = []

        # On parcourt toutes les positions cibles
        for symbol in target_positions.index:
            target_qty   = target_positions.loc[symbol, "quantity"]
            target_price = target_positions.loc[symbol, "price"]
            asset_type   = target_positions.loc[symbol, "asset_type"]

            # Position actuelle (0 si on n'a pas encore de position)
            current_qty = self.current_positions.get(symbol, 0)

            # Delta = ce qu'il faut trader
            delta_qty = target_qty - current_qty

            if delta_qty == 0:
                continue  # Pas de changement → pas de trade

            # Valeur du trade en dollars
            multiplier  = FUTURES_MULTIPLIERS.get(symbol, 1) if asset_type == "future" else 1
            dollar_trade = abs(delta_qty) * target_price * multiplier

            # Seuil de rebalancement : on ignore les micro-trades
            threshold_dollars = REBALANCING_THRESHOLD * self.capital
            if dollar_trade < threshold_dollars:
                logger.debug(
                    f"  {symbol} : trade {dollar_trade:,.0f}$ < seuil "
                    f"{threshold_dollars:,.0f}$ → ignoré"
                )
                continue

            # Estimation du coût de transaction
            # TRANSACTION_COST_BPS + SLIPPAGE_BPS = coût total en bps
            total_cost_bps = TRANSACTION_COST_BPS + SLIPPAGE_BPS
            estimated_cost = dollar_trade * total_cost_bps / 10_000

            # Direction du trade
            direction = "BUY" if delta_qty > 0 else "SELL"

            trades.append({
                "symbol"         : symbol,
                "asset_type"     : asset_type,
                "direction"      : direction,
                "current_qty"    : current_qty,
                "target_qty"     : target_qty,
                "delta_qty"      : delta_qty,
                "price"          : target_price,
                "dollar_trade"   : dollar_trade,
                "estimated_cost" : estimated_cost,
                "multiplier"     : multiplier,
            })

        # Positions à fermer (actifs dans current_positions mais pas dans target)
        for symbol, current_qty in self.current_positions.items():
            if symbol not in target_positions.index and current_qty != 0:
                price = current_prices.get(symbol, 0)
                if price == 0:
                    continue

                multiplier   = FUTURES_MULTIPLIERS.get(symbol, 1)
                dollar_trade = abs(current_qty) * price * multiplier
                estimated_cost = dollar_trade * (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10_000

                trades.append({
                    "symbol"         : symbol,
                    "asset_type"     : "unknown",
                    "direction"      : "CLOSE",
                    "current_qty"    : current_qty,
                    "target_qty"     : 0,
                    "delta_qty"      : -current_qty,
                    "price"          : price,
                    "dollar_trade"   : dollar_trade,
                    "estimated_cost" : estimated_cost,
                    "multiplier"     : multiplier,
                })

        if not trades:
            logger.info("✅ Aucun rebalancement nécessaire")
            return pd.DataFrame()

        trades_df = pd.DataFrame(trades).set_index("symbol")

        # Résumé du rebalancement
        total_traded  = trades_df["dollar_trade"].sum()
        total_cost    = trades_df["estimated_cost"].sum()
        turnover      = total_traded / self.capital

        logger.info(
            f"📋 Rebalancement : {len(trades_df)} trades | "
            f"Volume: {total_traded:,.0f}$ | "
            f"Turnover: {turnover:.1%} | "
            f"Coût estimé: {total_cost:,.0f}$ ({total_cost/self.capital*10000:.1f}bps)"
        )

        return trades_df

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 5 : Métriques du portefeuille
    # ─────────────────────────────────────────────────────────
    def compute_portfolio_metrics(
        self,
        positions: pd.DataFrame
    ) -> dict:
        """
        Calcule les métriques clés du portefeuille actuel.

        MÉTRIQUES :
          - Gross Exposure  : Σ|w_i| — levier total
          - Net Exposure    : Σw_i   — exposition directionnelle
          - Long Exposure   : Σmax(w_i, 0) — capital long
          - Short Exposure  : Σmin(w_i, 0) — capital short
          - Nb Long         : nombre de positions longues
          - Nb Short        : nombre de positions courtes
          - Concentration   : poids de la plus grande position (HHI)

        CONCENTRATION — Herfindahl-Hirschman Index (HHI) :
        HHI = Σ w_i² (somme des carrés des poids)
        HHI = 1/N pour un portefeuille parfaitement diversifié
        HHI = 1   pour une seule position concentrée
        Plus HHI est bas, plus le portefeuille est diversifié.
        Standard utilisé en gestion de portefeuille institutionnelle.

        Args:
            positions : DataFrame des positions (output weights_to_positions)

        Returns:
            dict des métriques
        """
        if positions.empty:
            return {}

        weights = positions["weight"]

        long_weights  = weights[weights > 0]
        short_weights = weights[weights < 0]

        gross_exposure = weights.abs().sum()
        net_exposure   = weights.sum()
        long_exposure  = long_weights.sum()
        short_exposure = short_weights.sum()

        # HHI — mesure de concentration
        # On normalise par rapport à un portefeuille equal-weighted
        # HHI normalisé = 1 si très concentré, 0 si parfaitement diversifié
        n = len(weights[weights != 0])
        hhi = (weights ** 2).sum()
        hhi_normalized = max(0.0, (hhi - 1/n) / (1 - 1/n)) if n > 1 else 1.0

        metrics = {
            "gross_exposure"  : gross_exposure,
            "net_exposure"    : net_exposure,
            "long_exposure"   : long_exposure,
            "short_exposure"  : short_exposure,
            "nb_long"         : len(long_weights),
            "nb_short"        : len(short_weights),
            "nb_total"        : n,
            "max_position"    : weights.abs().max(),
            "hhi_normalized"  : hhi_normalized,
            "capital_at_risk" : gross_exposure * self.capital,
        }

        return metrics

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 6 : Pipeline complète de rebalancement
    # ─────────────────────────────────────────────────────────
    def rebalance(
        self,
        signals: pd.Series,
        volatilities: pd.Series,
        prices: pd.Series,
        date: pd.Timestamp,
        asset_types: dict = None
    ) -> dict:
        """
        Pipeline complète de rebalancement pour une date donnée.

        C'est la méthode principale appelée par le backtest
        à chaque date de rebalancement.

        PIPELINE :
          1. Vol parity weights
          2. Apply constraints
          3. Convert to positions
          4. Compute rebalancing trades
          5. Update current positions
          6. Compute metrics

        Args:
            signals      : Series des signaux finaux { symbol: signal }
            volatilities : Series des vols EWMA { symbol: vol }
            prices       : Series des prix actuels { symbol: prix }
            date         : date du rebalancement
            asset_types  : dict { symbol: "stock"/"future" }

        Returns:
            dict avec positions, trades, métriques
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"📅 Rebalancement du {date.date()}")
        logger.info(f"{'='*50}")

        # Étape 1 : Vol parity weights
        raw_weights = self.compute_vol_parity_weights(signals, volatilities)
        logger.info(f"  Poids bruts calculés : {len(raw_weights)} actifs")

        # Étape 2 : Contraintes
        constrained_weights = self.apply_constraints(raw_weights)
        logger.info(
            f"  Gross exposure : {constrained_weights.abs().sum():.2f}x | "
            f"Net exposure : {constrained_weights.sum():.2f}x"
        )

        # Étape 3 : Conversion en positions concrètes
        positions = self.weights_to_positions(
            constrained_weights, prices, asset_types
        )

        # Étape 4 : Trades nécessaires
        trades = self.compute_rebalancing_trades(positions, prices)

        # Étape 5 : Mise à jour des positions actuelles
        # On met à jour le dictionnaire des positions courantes
        for symbol in positions.index:
            self.current_positions[symbol] = positions.loc[symbol, "quantity"]
            self.current_weights[symbol]   = positions.loc[symbol, "weight"]

        # Étape 6 : Métriques
        metrics = self.compute_portfolio_metrics(positions)

        # Sauvegarde dans l'historique
        portfolio_snapshot = {
            "date"      : date,
            "positions" : positions,
            "trades"    : trades,
            "metrics"   : metrics,
            "capital"   : self.capital,
        }
        self.portfolio_history.append(portfolio_snapshot)

        if not trades.empty:
            trades["date"] = date
            self.trades_history.append(trades)

        # Affichage du résumé
        self._print_portfolio_summary(positions, metrics, date)

        return portfolio_snapshot

    # ─────────────────────────────────────────────────────────
    # MÉTHODE UTILITAIRE : Affichage du portefeuille
    # ─────────────────────────────────────────────────────────
    def _print_portfolio_summary(
        self,
        positions: pd.DataFrame,
        metrics: dict,
        date: pd.Timestamp
    ):
        """Affiche un résumé lisible du portefeuille."""

        print(f"\n{'─'*65}")
        print(f"  PORTEFEUILLE AU {date.date()}")
        print(f"{'─'*65}")
        print(f"  Capital      : {self.capital:>12,.0f} $")
        print(f"  Gross Exp.   : {metrics.get('gross_exposure', 0):>12.2f} x")
        print(f"  Net Exp.     : {metrics.get('net_exposure', 0):>12.2f} x")
        print(f"  Long / Short : {metrics.get('nb_long', 0)} L / {metrics.get('nb_short', 0)} S")
        print(f"  Concentration: {metrics.get('hhi_normalized', 0):>12.3f} (HHI)")
        print(f"{'─'*65}")

        # Tableau des positions
        if not positions.empty:
            display_cols = ["weight", "quantity", "price", "dollar_actual", "asset_type"]
            available    = [c for c in display_cols if c in positions.columns]
            display      = positions[available].copy()
            display = display.sort_values("weight", key=lambda x: x.abs(), ascending=False)
            display["weight"] = display["weight"].map("{:.3f}".format)
            display["dollar_actual"] = display["dollar_actual"].map("{:,.0f}$".format)
            print(display.to_string())

        print(f"{'─'*65}\n")

    # ─────────────────────────────────────────────────────────
    # MÉTHODE UTILITAIRE : Résumé de l'historique
    # ─────────────────────────────────────────────────────────
    def get_history_summary(self) -> pd.DataFrame:
        """
        Retourne un résumé de l'historique des rebalancements.

        Utile pour analyser l'évolution du portefeuille dans le temps
        et calculer les métriques de turnover sur la durée du backtest.
        """
        if not self.portfolio_history:
            return pd.DataFrame()

        rows = []
        for snap in self.portfolio_history:
            m = snap["metrics"]
            rows.append({
                "date"           : snap["date"],
                "gross_exposure" : m.get("gross_exposure", 0),
                "net_exposure"   : m.get("net_exposure", 0),
                "nb_long"        : m.get("nb_long", 0),
                "nb_short"       : m.get("nb_short", 0),
                "max_position"   : m.get("max_position", 0),
                "hhi"            : m.get("hhi_normalized", 0),
            })

        return pd.DataFrame(rows).set_index("date")


# ============================================================
# SCRIPT PRINCIPAL — Test avec données simulées
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  TEST — PortfolioConstructor")
    print("=" * 60)

    # On importe et on relance la pipeline signal pour avoir
    # des signaux réalistes sur lesquels tester le portfolio
    import sys
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    from strategies.momentum.momentum_signal import MomentumSignalGenerator

    # ── Génération de données simulées ────────────────────────
    np.random.seed(42)
    n_days   = 756
    assets   = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                "GC",   "CL",   "JPM",   "XOM",  "NVDA"]

    # Types d'actifs — utile pour le calcul des positions
    asset_types = {
        "AAPL": "stock", "MSFT": "stock", "GOOGL": "stock",
        "AMZN": "stock", "META": "stock", "JPM":   "stock",
        "XOM":  "stock", "NVDA": "stock",
        "GC":   "future", "CL": "future",
    }

    # Prix simulés (GBM)
    daily_returns = np.random.normal(
        loc=0.08/252, scale=0.20/np.sqrt(252),
        size=(n_days, len(assets))
    )
    prices_array = 100 * np.exp(np.cumsum(daily_returns, axis=0))
    dates        = pd.bdate_range(start="2021-01-01", periods=n_days)
    price_matrix = pd.DataFrame(prices_array, index=dates, columns=assets)

    # ── Pipeline signal ────────────────────────────────────────
    print("\n🚀 Calcul des signaux...")
    generator = MomentumSignalGenerator(price_matrix)
    results   = generator.run_full_pipeline()

    # ── Test du portefeuille sur la dernière date ──────────────
    last_date = results["signal_final"].index[-1]
    signals   = results["signal_final"].loc[last_date].dropna()
    vols      = results["ewma_vol"].loc[last_date].dropna()
    prices    = price_matrix.loc[last_date]

    print(f"\n📊 Construction du portefeuille au {last_date.date()}...")
    constructor = PortfolioConstructor(capital=100_000)

    snapshot = constructor.rebalance(
        signals    = signals,
        volatilities = vols,
        prices     = prices,
        date       = last_date,
        asset_types = asset_types
    )

    # ── Métriques finales ──────────────────────────────────────
    print("\n📈 Métriques du portefeuille :")
    for k, v in snapshot["metrics"].items():
        if isinstance(v, float):
            print(f"  {k:<20} : {v:.4f}")
        else:
            print(f"  {k:<20} : {v}")

    # ── Test d'un second rebalancement ────────────────────────
    # On simule un rebalancement 1 mois plus tard
    print("\n\n🔄 Test rebalancement mois suivant...")
    prev_date = results["signal_final"].index[-22]  # ~1 mois avant
    signals2  = results["signal_final"].loc[prev_date].dropna()
    vols2     = results["ewma_vol"].loc[prev_date].dropna()
    prices2   = price_matrix.loc[prev_date]

    snapshot2 = constructor.rebalance(
        signals      = signals2,
        volatilities = vols2,
        prices       = prices2,
        date         = prev_date,
        asset_types  = asset_types
    )

    print("\n📋 Historique des rebalancements :")
    print(constructor.get_history_summary().round(4))