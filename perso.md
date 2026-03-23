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
