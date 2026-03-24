# Notes projet Fist

**Comment lancer le projet :** voir le fichier racine **`README.md`** (points d’entrée officiels et commandes).

---

## Structure actuelle du dépôt (code)

```
Fist/
├── config.py                    ← Paramètres globaux (IBKR, univers, signal, risque, chemins)
├── README.md                    ← Entrées officielles + prérequis
├── baseline_event_driven_reference.json
├── run_backtest.py              ← Pipeline backtest vectorisé (maître)
├── event_driven/                ← Package backtest event-driven + CLI (`python -m event_driven`)
├── docs/event_driven.md         ← Carte du package
├── event_driven_live.py         ← Dashboard animé (Plotly)
├── event_driven_risk.py         ← Test / démo risk manager event-driven
├── optimizer.py                 ← Optimisation walk-forward (démo __main__)
├── validator.py                 ← Validation WF + Monte Carlo
├── visualizer.py                ← Exports graphiques
├── dashboard_live.py            ← Streamlit (streamlit run …)
├── risk_enhanced.py             ← Filtres régime (phase 2B, expérimental / lib)
├── data/
│   ├── ibkr_data.py             ← Connexion IBKR + cache raw / processed
│   ├── raw/                     ← CSV téléchargés (gitignore)
│   └── processed/               ← price_matrix.csv (gitignore)
├── strategies/momentum/
│   ├── momentum_signal.py
│   ├── portfolio.py
│   └── backtest_vectorized.py
├── risk/
│   ├── regime_*.py, overlay.py, rebalance.py, risk_types.py
│   └── risk_manager.py
├── metrics/performance.py
└── tests/                       ← pytest
```

---

## Roadmap (objectifs produit)

### PHASE 1 — Vectorisé
- `strategies/momentum/backtest_vectorized.py` — moteur rapide research  
- `metrics/performance.py` — Sharpe, DD, Calmar…  
- Tests + validation

### PHASE 2 — Event-driven
- `event_driven/` — simulation réaliste jour par jour (`python -m event_driven`)  
- `visualizer.py` / dashboards — lecture des résultats  
- Comparaison avec le vectorisé

### PHASE 2A — Optimisation du signal
- Calibration fenêtres momentum (21/63/126/252)  
- Poids CS vs TS, filtrage univers, walk-forward (éviter overfitting)

### PHASE 2B — Risk management
- Max DD, vol targeting, filtre régime, stops dynamiques  
- Code dispersé entre `risk/` et expérimentations (`risk_enhanced.py`, etc.)

### PHASE 2C — Coûts
- Turnover, seuil rebalancement adaptatif, market impact

### PHASE 2D — Validation statistique
- Walk-forward OOS, Monte Carlo, robustesse paramètres, benchmarks

### PHASE 3 — Ghost test
- Seulement si Sharpe > 0.5 et Max DD < 20 % (critères indicatifs dans ta doc historique)

---

## Discipline recherche — train 1 vs OOS (style institutionnel)

**Règle gelée :** toute exploration de paramètres, hypothèses ou code « perf » se fait **uniquement** sur **train 1** (`RESEARCH_TRAIN_1_*` dans `config.py`, 2015-01-01 → 2019-12-31). Commande : `python -m event_driven --train1`.

**OOS (2020 → fin backtest) :** **interdit** comme terrain d’optimisation ou de choix de paramètres. On n’y lance un run (`python -m event_driven --oos1`) qu’**après** avoir figé une variante validée sur train 1 (+ checks robustesse), **sans** modifier la config entre le choix et l’OOS.

**Baseline train 1 :** fichier racine `baseline_event_driven_train1.json` — comparée automatiquement aux runs `--train1` (garde-fous `BASELINE_*` dans `config.py`).

**Robustesse (train 1) :** stress coûts — `python -m event_driven --train1 --stress-cost-mult 1.2` (commission + slippage × multiplicateur).

**Journal court (à tenir à la main) :** pour chaque idée : hypothèse → changement (un bloc : signal / coûts / risk) → `python -m event_driven --train1` → métriques vs référence → **garder / rejeter**.

**Train 2+ :** bornes futures `RESEARCH_TRAIN_2_START` / `RESEARCH_TRAIN_2_END` dans `config.py` (walk-forward) — ne pas calibrer en regardant l’OOS déjà consommé.

### Diagnostic train 1 (snapshot)

**Référence A (résultats archivés) :** `results/event_driven_train1_baseline/` — dernier run figé : `stats_20260324_203956.csv` (949 jours, 2015-01-01 → 2019-12-31). Baseline JSON : `baseline_event_driven_train1.json` (comparaison PASS au save).

**Sortie `research_diagnose` (même CSV) :** return total ~17 %, max DD ~−13.5 %, turnover annualisé ~8.9, 46 jours de rebalance, 16 rebalances sans ordres, 15 avec gross post-filtre régime « ancien » nul, 0 jour `trading_suspended`. Régimes (comptage jours) : BULL 388, CRISIS 346, NORMAL 115, STRESS 100. Raisons signal : OK 31, INSUFFICIENT_HISTORY 15.

**1–3 leviers prioritaires (train 1) :**

1. **Turnover** élevé (~9×/an) → bloc **signal / coûts** : `REBALANCE_THRESHOLD_DEFAULT`, sensibilité `TRANSACTION_COST_BPS` / `SLIPPAGE_BPS`, run `--stress-cost-mult 1.2`.
2. **Rebalances sans exécution / gross filtré** → bloc **risk** : overlay CRISIS, alignement filtre régime vs jours effectifs (cf. `gross_after_old_regime_filter`, `suspension_reason` dans les exports moteur).
3. **Part CRISIS élevée** → vérifier données + comportement hard-flat ; itérer risk **après** stabilisation du signal.

**Itération par blocs :** une hypothèse à la fois (signal → coûts → risk) ; après chaque changement : `python -m event_driven --train1` ; noter garder/rejeter vs référence A.

**Robustesse avant OOS :** au minimum `--train1 --stress-cost-mult 1.2` (et éventuellement 1.15 / 1.25) ; ne pas optimiser sur l’OOS.

**OOS figé :** une fois la variante choisie sur train 1, un seul `python -m event_driven --oos1` sans modifier `config.py` entre-temps (comparaison baseline OOS optionnelle via `--baseline-json` si besoin).

### Audit ref A (gel config / traçabilité)

- **Git HEAD** au cycle d’itération : `4c0dc73e8679825f382a9e568102ec6f03cf4b28` (noter tout nouveau commit si la ref A change).
- **Baseline JSON train 1 :** `baseline_event_driven_train1.json` — inchangé après ce cycle (seuil rebal conservé à 0.02).
- **Sweeps additionnels :** `results/train1_threshold_sweep/`, `results/train1_stress_costs/`, **OOS figé** : `results/train1_oos_frozen_20260324/` (`stats_20260324_214300.csv`).

### Journal itération train 1 — cycle 2026-03-24

**Critère annoncé (bloc signal) :** n’augmenter `REBALANCE_THRESHOLD_DEFAULT` que si le **max DD** ne se dégrade pas de plus de **~2 points** vs référence train 1 (−13,52 % → plancher ≈ −15,5 %), tout en cherchant une baisse de turnover.

| Seuil (rebal) | CAGR train 1 | Sharpe | Max DD | Turnover/an | Trades |
|---------------|-------------|--------|--------|-------------|--------|
| 0,02 (réf.)   | 4,26 %      | −0,040 | −13,52 % | 8,92×     | 343    |
| 0,025         | 3,60 %      | −0,108 | −15,74 % | 8,81×     | 336    |
| 0,03          | 3,28 %      | −0,140 | −16,90 % | 8,70×     | 325    |

**Décision :** **rejeter** 0,025 et 0,03 — gain de turnover trop faible vs dégradation DD / Sharpe. **Conserver `REBALANCE_THRESHOLD_DEFAULT = 0,02`** dans `config.py`.

**CLI recherche :** `python -m event_driven --train1 --rebalance-threshold <x>` pour reproduire le sweep sans toucher au fichier config.

**Stress coûts (train 1, seuil 0,02)** — dossiers `results/train1_stress_costs/mult_*` :

| Mult | CAGR | Sharpe | Max DD | Turnover/an | Trades |
|------|------|--------|--------|-------------|--------|
| ×1,2 | 3,98 % | −0,070 | −13,72 % | 8,92× | 343 |
| ×1,25 | 3,92 % | −0,078 | −13,77 % | 8,92× | 343 |

**Bloc risk (ancienne note signal-only) :** voir **Journal bloc risque** ci-dessous — les leviers CRISIS / breaker ont été **testés** ; `config.py` reste sur la variante **retenue** (hard-flat CRISIS off, MAX_DD 20 %, overlays désactivés, etc.).

**OOS figé (une exécution, `--skip-baseline`, config inchangée après décision train 1) :**

- Commande : `python -m event_driven --oos1 --skip-baseline --output ./results/train1_oos_frozen_20260324`
- Période : 2020-01-01 → 2024-12-31 (1258 jours).
- Métriques (stats CSV, `RISK_FREE_RATE` config) : CAGR ≈ **0,90 %**, Sharpe ≈ **−0,30**, max DD ≈ **−19,5 %**, turnover annualisé ≈ **9,53×**.  
- **Ne pas** retuner sur ces résultats OOS ; toute suite = nouvelle hypothèse sur **train 1** uniquement.

### Journal bloc risque train 1 — plan institutionnel (2026-03-24)

**Critère ex ante (phase 1 CRISIS) :** n’adopter `REGIME_WEIGHT_FILTER_CRISIS_HARD_FLAT = True` que si le **max DD** train 1 **s’améliore** nettement vs ref A **sans** dégrader le CAGR au-delà d’un seuil raisonnable ; sinon **rejeter**.

**Reproductibilité :** `python -m event_driven.research_risk_train1 --output-root ./results/risk_train1_plan` (patchs **en mémoire** uniquement — le fichier `config.py` du dépôt n’a **pas** été modifié pour les sweeps).

| Scénario | CAGR | Sharpe | Max DD | TO/an | Trades | Décision |
|----------|------|--------|--------|-------|--------|----------|
| ref (RAM, = baseline JSON) | 4,26 % | −0,040 | −13,52 % | 8,92× | 343 | référence |
| **P1** `CRISIS_HARD_FLAT=True` | 4,01 % | −0,066 | −14,48 % | 8,93× | 274 | **rejeter** (DD et Sharpe plus mauvais) |
| **P2** `MAX_PORTFOLIO_DRAWDOWN=0,18` | 4,26 % | −0,040 | −13,52 % | 8,92× | 343 | neutre (identique ref sur train 1) |
| **P2** `MAX_PORTFOLIO_DRAWDOWN=0,22` | idem | idem | idem | idem | idem | neutre |
| **P2** `SUSPENSION_COOLDOWN=28j` | idem | idem | idem | idem | idem | neutre |
| **P2** `SUSPENSION_REENTRY_DD_FROM_EXIT=-0,04` | idem | idem | idem | idem | idem | neutre |
| **P3** `RISK_INFORMED_EXPOSURE_TILT=True` | 3,98 % | −0,071 | −14,27 % | 8,56× | 329 | **rejeter** |
| **P3** `ENABLE_MARKET_OVERLAY` + `risk_off_only` | idem ref | idem | idem | idem | idem | neutre (pas d’effet mesurable ici) |
| **P4** `FAST_DRAWDOWN_CUT=True` | idem ref | idem | idem | idem | idem | neutre |
| **P4** `DEFENSIVE_FLAT_ENABLED=True` | 1,75 % | −0,361 | −14,31 % | 7,41× | 311 | **rejeter** |
| **P4** `PROLONGED_UNDERWATER=True` | idem ref | idem | idem | idem | idem | neutre |

**Variante retenue (inchangée vs ref A) :** `REGIME_WEIGHT_FILTER_CRISIS_HARD_FLAT=False`, `MAX_PORTFOLIO_DRAWDOWN=0,20`, `SUSPENSION_*` par défaut, overlays / informed / defensive / prolonged **off** comme dans `config.py` actuel. `baseline_event_driven_train1.json` **non** mis à jour.

**Robustesse coûts (post-décision) :** `python -m event_driven --train1 --stress-cost-mult 1.2 --skip-baseline --output ./results/risk_plan_stress_train1` → CAGR ≈ **3,98 %**, Sharpe ≈ **−0,070**, max DD ≈ **−13,72 %** (949 jours).

**OOS figé (bloc risque, sans retuning) :** `python -m event_driven --oos1 --skip-baseline --output ./results/risk_plan_oos_frozen_20260324b` — stats `stats_20260324_220313.csv` : CAGR ≈ **0,90 %**, Sharpe ≈ **−0,30**, max DD ≈ **−19,5 %**, TO ≈ **9,53×** (1258 jours).
