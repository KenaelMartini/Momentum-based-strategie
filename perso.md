quant_strategies/
├── config.py              ← Paramètres globaux (port, IP, account)
├── data/
│   └── ibkr_data.py       ← Connexion IBKR + récupération données
├── strategies/
│   └── momentum/
│       ├── signal.py      ← Construction du signal
│       ├── portfolio.py   ← Sizing & pondération
│       └── backtest.py    ← Moteur de backtest
├── risk/
│   └── risk_manager.py    ← Contrôle du risque
└── metrics/
    └── performance.py     ← Sharpe, Drawdown, etc.

Pour backtest : 

PHASE 1 — Vectorisé (maintenant)
├── backtest_vectorized.py    ← moteur rapide pour le research
├── performance.py            ← métriques Sharpe, DD, Calmar...
└── Test + validation complète

PHASE 2 — Event-Driven (après)
├── backtest_eventdriven.py   ← simulation réaliste jour par jour
├── visualizer.py             ← dashboard live en temps réel
└── Test + comparaison avec vectorisé

Optimisation de la stratégie : 

PHASE 2A — Optimisation du signal
├── Calibration des fenêtres momentum (21/63/126/252)
├── Optimisation des poids CS vs TS
├── Filtrage de l'univers (liquidité, qualité)
└── Walk-forward validation (éviter l'overfitting)

PHASE 2B — Amélioration du risk management  
├── Réduction du Max Drawdown (-34% → < -20%)
├── Optimisation du vol targeting
├── Ajout d'un filtre de régime de marché
└── Stop-loss dynamiques

PHASE 2C — Réduction des coûts ( Phase passer car la phase 2.B fait déja en grande partie ce que l'on attendait de la 2.C )
├── Optimisation du turnover (76% → < 40%)
├── Seuil de rebalancement adaptatif
└── Modélisation du market impac 

PHASE 2D — Validation statistique
├── Walk-forward backtest (out-of-sample)
├── Monte Carlo stress testing
├── Analyse de robustesse des paramètres
└── Comparaison avec benchmarks institutionnels

PHASE 3 — Ghost test
└── Seulement si Sharpe > 0.5 et Max DD < 20%






Problèmes a fix : 

- Gestion de régimes, identification des régimes mais transition trop lente, si crise -> réduire sans attendre le rebal mensuel.
- Réentrée après CB 30 jours après mais n'a jamais de réel confirmation du marché -> ajout de confirmation du marché, on doit mettre en place un pré CB. Réduction progressive du risque dès -5%, ne pas attendre 10% ou 15% de pertes. La réentrer après CB doit etre basée sur le régime + vol + momentum, pas sur le temps. 
- Cap levier, 1.5x uniquement si volatilité basse confirmer sinon max 1.2x
- Kill switch intraday, si perte journalière > X -> stop immédiat
- Réfléchir a comment réduire le rebal mensuel ( turnover/an = 1145.7% )
