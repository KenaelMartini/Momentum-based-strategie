# Package `event_driven`

Backtest **jour par jour** (pas de look-ahead), aligné sur la Phase 3 du projet.

## Lancer

```bash
python -m event_driven --data ./data/processed/price_matrix.csv --output ./results/event_driven
```

Rapport HTML d'analyse poussee des phases (risk-off/risk-on, DD, exposition/turnover, transitions, episodes) :

```bash
python -m event_driven.regime_phase_report --results-dir ./results/event_driven
```

Sur la vue temporelle, une courbe **benchmark** (buy-and-hold **équipondéré** sur `price_matrix.csv`, même capital initial que le backtest) est superposée à la stratégie. Désactiver : `--no-benchmark`. Autre fichier de prix : `--price-matrix ./chemin/price_matrix.csv`.

Options : `--start`, `--end`, `--live`, `--data`, `--output`.

## Modules

| Fichier | Rôle |
|---------|------|
| `events.py` | Enums + dataclasses (`MarketEvent`, `SignalEvent`, `PortfolioStats`, …) |
| `data_handler.py` | Lecture CSV, bar suivant, historique ≤ T |
| `broker.py` | File d’ordres, slippage, commission |
| `portfolio.py` | Cash, positions, génération d’ordres, stats |
| `signal_generator.py` | `MomentumSignalGenerator` (legacy / research ; le moteur principal utilise `MomentumSignalGeneratorV2` dans `event_driven_risk.py`) |
| `visualizer.py` | Export HTML Plotly 3D |
| `baseline.py` | Garde-fous vs `baseline_event_driven_reference.json` |
| `engine.py` | `EventDrivenEngine` — boucle principale |
| `__main__.py` | CLI |

## Extensions

- **`per_day_callback(stats, engine)`** et **`day_sleep_sec`** sur `EventDrivenEngine` : utilisés par `event_driven_live.py` (Dash) pour pousser les stats jour après jour sans dupliquer la logique métier.

## Régime marché « effectif » vs modèle features

- Le moteur **features** (`RegimeEngine`) produit un état `model=` (TREND, RISK_ON, …) à partir de breadth, VIX, etc.
- Le **risk manager** produit un état discret **BULL / NORMAL / STRESS / CRISIS / …** à partir des filtres portefeuille.
- Si `ALIGN_MARKET_REGIME_WITH_RISK` est **True** (défaut), l’état utilisé dans les stats et dans `market_regime_state` après merge est l’**effectif** : il remonte souvent en **RISK_OFF** quand le risk voit CRISIS/SUSPENDED, ou **TRANSITION** sous STRESS, même si le modèle features reste optimiste. Colonnes CSV : `market_regime_feature`, `market_regime_effective`, `market_regime_align_reason`, `risk_regime_name`.
- **BULL/NORMAL avec un portefeuille qui chute** (ex. début 2023) : le score de régime est `min(trend, vol, corr, dd_score)` sur le **benchmark** (trend/vol/corr) et le **drawdown du book** (`dd_score`). Deux effets fréquents : (1) une grille `dd_score` trop plate entre ~−8 % et −12 % laissait un score 0,75 → encore **NORMAL** alors que le book souffre ; (2) un rebond de quelques séances **réduit le DD vs pic** → `dd_score` peut repasser haut quelques jours alors que la tendance du portefeuille reste mauvaise. La fonction `_compute_dd_score` dans `event_driven_risk.py` utilise des paliers plus serrés (notamment au-delà de ~−9 % de DD) pour que le filtre drawdown pousse plus tôt vers **STRESS** quand il domine le min.

## Flat défensif + `AWAIT_REGIME` (machine à états)

Indépendant du **circuit breaker** `-15 %` : si `DEFENSIVE_FLAT_ENABLED` est **True** dans `config.py` :

1. **`INVESTED`** — comportement normal.
2. **`DEFENSIVE_FLAT`** — passage en **cash** (liquidation quotidienne comme sous suspension, mais sans `trading_suspended`). Déclenché quand le régime **effectif** est dans `DEFENSIVE_FLAT_ENTRY_EFFECTIVE_REGIMES` (ex. `RISK_OFF`) **et** `current_drawdown <= DEFENSIVE_FLAT_ENTRY_MIN_DD` pendant `DEFENSIVE_FLAT_ENTRY_MIN_CONSECUTIVE_DAYS` jours consécutifs.
3. **`AWAIT_REGIME`** — toujours cash ; après `DEFENSIVE_FLAT_MIN_CALENDAR_DAYS` jours calendaires depuis l’entrée en flat, on surveille la **réentrée** :
   - soit le régime effectif reste dans `DEFENSIVE_FLAT_REENTRY_EFFECTIVE_REGIMES` (ex. `TREND`, `RISK_ON`) pendant `DEFENSIVE_FLAT_REENTRY_EFFECTIVE_CONSECUTIVE` jours ;
   - **ou** le régime risk (`BULL` / `NORMAL` / …) est dans `DEFENSIVE_FLAT_REENTRY_RISK_REGIMES` pendant `DEFENSIVE_FLAT_REENTRY_RISK_CONSECUTIVE` jours.

Aux mois de rebalance en flat, aucun signal momentum n’est calculé (`DEFENSIVE_FLAT` dans les diagnostics).

Colonnes stats : `defensive_flat_phase`, `defensive_flat_reason`. **Défaut : `DEFENSIVE_FLAT_ENABLED = False`** (baseline inchangée).

## Sous l’eau prolongé & suspension (config)

- **`PROLONGED_UNDERWATER_*`** : jours consécutifs avec `drawdown < 0` **et** `current_drawdown <=` seuil → multiplie **`risk_scaling`** (réduction d’expo sans liquidation totale). Cible les séries de DD > 10 jours **profondes**.
- **`SUSPENSION_*`** : après circuit breaker (`MAX_PORTFOLIO_DRAWDOWN`), réentrée si cooldown calendaire + DD depuis la valeur de sortie en cash au-dessus d’un seuil. **`SUSPENSION_REENTRY_FAST_CALENDAR_DAYS = 0`** désactive la voie rapide (comportement proche de l’historique 30j / -5%).
- **Optionnel** (à ajouter dans `config.py` si besoin ; sinon valeurs par défaut dans `event_driven_risk.py`, tout **désactivé** = même logique qu’avant) :
  - réentrée renforcée : `SUSPENSION_REENTRY_REQUIRE_REGIME_CONFIRMATION`, `SUSPENSION_REENTRY_ALLOWED_RISK_REGIMES`, `SUSPENSION_REENTRY_MIN_CONSECUTIVE_RISK_DAYS`, `SUSPENSION_REENTRY_RAMP_*`, `SUSPENSION_POST_REENTRY_GUARD_*` ;
  - **recut post-suspension** (`SUSPENSION_POST_REENTRY_RECUT_*`) : ne s’active qu’après une **réentrée** suite à `trading_suspended` ; ancre = **première séance investie** après cette réentrée ; N séances ; **pire clôture** vs ancre. Ne couvre **pas** une simple baisse après un **rebalance mensuel** sans suspension avant (ex. début janv. 2022).
  - **coupe post-rebalance** (`REBALANCE_WINDOW_LOSS_CUT_*`) : **indépendant** du recut ci-dessus ; ancre = valeur portefeuille **après** chaque rebalance mensuel ; même logique N séances / pire clôture → `suspension_reason=REBALANCE_WINDOW_LOSS_CUT`. C’est ce qui attrape le scénario « rebal 3 jan. puis −2 %+ en quelques séances ».
  - **`REBALANCE_FILL_SAME_BAR`** (défaut True dans `config.py`) : exécuter le rebalance au **même bar** pour que le risk voie les positions (sinon T+1 = fenêtre vide ou décalée).
  - **fast DD cut** : `FAST_DRAWDOWN_CUT_ENABLED`, `FAST_DRAWDOWN_CUT_THRESHOLD`, `FAST_DRAWDOWN_CUT_WINDOW_DAYS`, `FAST_DRAWDOWN_CUT_WINDOW_DAYS_RISK_OFF` ; option horizon lent `FAST_DRAWDOWN_CUT_THRESHOLD_LONG` + `FAST_DRAWDOWN_CUT_WINDOW_DAYS_LONG` et confirmation `FAST_DRAWDOWN_CUT_CONFIRM_DAYS` (N jours de breach consécutifs) → `suspension_reason=FAST_DRAWDOWN_BREACH` quand activé.
  - **profil C** : `FAST_DRAWDOWN_CUT_ONLY_UNDER_STRESS = True` → le fast cut **court** ne s’applique qu’en **STRESS** / **CRISIS**. L’horizon **long** (`THRESHOLD_LONG` / `WINDOW_DAYS_LONG`) s’évalue **dans tous les régimes** (BULL/NORMAL inclus).

Pour **couper fort en RISK_OFF** (constat stats), activer **`RISK_INFORMED_EXPOSURE_TILT_ENABLED`** et baisser **`RISK_INFORMED_SCALE_RISK_OFF`** (ex. 0.48–0.58) : sur l’échantillon actuel le **Max DD global** peut **augmenter** (effets de chemin) ; ajuster ou relever le baseline si tu assumes ce profil.

## Tilts d’exposition (optionnels)

- `RISK_INFORMED_EXPOSURE_TILT_ENABLED` : après `apply_regime_weight_filter`, multiplie les poids quand l’état **aligné** est défensif (ex. RISK_OFF, TRANSITION sous STRESS, TREND sous STRESS). **Désactivé par défaut** pour ne pas casser le Max DD vs baseline ; à activer pour un hedge plus net.
- `TREND_DRAWDOWN_TILT_ENABLED` : dans `apply_regime_weight_filter`, paliers supplémentaires sur le drawdown **portfolio** lorsque l’état aligné est **TREND**. **Désactivé par défaut** ; même remarque.

## Fichiers `regime_performance_*.csv`

Après chaque run, deux tableaux **mêmes métriques** (DD jour, épisodes DD >10j, croissance >5j, Sharpe conditionnel, etc.) :

| Fichier | Colonne de régime | Interprétation |
|---------|-------------------|----------------|
| `regime_performance_model_{ts}.csv` | `market_regime_state_model` ou `market_regime_feature` | **Avant alignement risk** — pur classifieur features (comme l’analyse historique). |
| `regime_performance_effective_{ts}.csv` | `market_regime_effective` | **Nouveau système** — état aligné sur BULL/STRESS/CRISIS. |
| `regime_performance_{ts}.csv` | (copie de l’effectif) | Compatibilité avec les scripts qui lisaient un seul fichier. |

Les logs affichent les **deux** blocs l’un après l’autre pour comparaison.

En plus des métriques sur les rendements **conditionnels** au régime (`max_drawdown` = drawdown max sur la sous-courbe des seuls jours de ce régime), chaque fichier inclut :

- **`mean_portfolio_drawdown`** / **`worst_portfolio_drawdown`** : moyenne et minimum du drawdown **journalier** du portefeuille (vs pic global) sur les jours étiquetés ce régime.
- **`n_dd_episodes_gt_10d`** : dans chaque **plage consécutive** où le régime ne change pas, nombre d’épisodes de jours consécutifs avec `drawdown < 0` de longueur **strictement supérieure à 10** (≥ 11 jours de bourse).
- **`n_growth_episodes_gt_5d`** : même logique pour des jours consécutifs avec `daily_return > 0`, longueur **> 5** (≥ 6 jours).
