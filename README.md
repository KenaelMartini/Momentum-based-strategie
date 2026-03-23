# Fist — Momentum multi-actifs (backtest & IBKR)

Stratégie **momentum** long/short sur actions et futures, backtest **vectorisé** et **event-driven**, métriques de performance, risque (vol targeting, régimes), connexion **Interactive Brokers** pour les données.

Toute exécution se fait depuis la **racine du dépôt** (`cd` ici), pour que les imports (`config`, `risk`, `event_driven`, etc.) et les chemins `./data/`, `./results/` fonctionnent.

---

## Prérequis

- Python 3.11+ (3.13 utilisé en local)
- Données : soit cache `data/processed/price_matrix.csv`, soit **TWS / IB Gateway** (paper, port défaut dans `config.py`)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Points d’entrée officiels

| Rôle | Commande typique |
|------|------------------|
| **Pipeline backtest vectorisé** (données → signal → perf → fichiers dans `results/`) | `python run_backtest.py --use-cache` (équivalent : `python -m run_backtest --use-cache`) |
| **Backtest event-driven** (simulation jour par jour, exports HTML/CSV, garde-fous baseline) | `python -m event_driven` — options : `--start`, `--end`, `--data`, `--output`, `--live` |
| **Dashboard event-driven animé** (Plotly Dash) | `python event_driven_live.py` — `--port`, `--speed`, etc. |
| **Récupération IBKR** (construit `data/raw/*.csv` + `price_matrix.csv`) | `python data/ibkr_data.py` |
| **Graphiques** (matplotlib / plotly / export streamlit) | `python visualizer.py --format all` |
| **Dashboard Streamlit** (résultats déjà produits) | `streamlit run dashboard_live.py` |
| **Optimisation walk-forward** (démo dans `__main__`, Optuna si installé) | `python optimizer.py` |
| **Validation walk-forward + Monte Carlo** | `python validator.py` (nécessite le cache `price_matrix.csv`) |
| **Tests** | `python -m pytest tests/ -q` |

### Détail rapide — backtest principal

```bash
python run_backtest.py --use-cache --start 2015-01-01 --end 2024-12-31
```

### Détail rapide — event-driven

```bash
python -m event_driven --data ./data/processed/price_matrix.csv --output ./results/event_driven
```

---

## Arborescence utile (code)

```
config.py                 # Paramètres globaux (univers, signal, risque, chemins données)
data/ibkr_data.py         # Fetch IBKR + écriture raw / processed
strategies/momentum/      # Signal, portefeuille, backtest vectorisé
risk/                     # Régimes, overlay, rebalance, types
metrics/performance.py    # Sharpe, drawdown, etc.
event_driven/             # Package : moteur event-driven + CLI (`python -m event_driven`) — voir docs/event_driven.md
run_backtest.py           # Orchestration backtest vectorisé
tests/                    # pytest (baseline, overlay risque)
baseline_event_driven_reference.json   # Référence métrique pour les tests baseline
```

Les sorties générées (`results/`, `logs/`, CSV dans `data/raw` / `data/processed`) sont en principe **hors Git** (voir `.gitignore`).

---

## Modules à la racine (secondaires / historique)

Ces fichiers restent à la racine pour ne pas casser les imports existants ; à traiter comme **outils** ou **expérimentations**, pas comme entrées “produit” du jour le jour :

- `event_driven_risk.py` — démo / test du gestionnaire de risque event-driven  
- `risk_enhanced.py` — filtres de régime (phase 2B), plutôt bibliothèque / expérimentation  
- `risk/risk_manager.py` — peut être lancé en script de démo (`__main__`)

Pour la roadmap long terme (phases 2A–3), voir **`perso.md`**.
