# Offline bandit replay v0 (2026-07-02) — single-turn retrieve-or-not / arm choice

Rung 1 of the skill-distillation bridge: can a *cheap trained policy* allocate
retrieval per-question better than fixed policies, evaluated by offline replay of
the paired 2026-07-02 arms (no new LLM calls)? Reward = correct − λ·k-tokens.
50/50 train/test split, seed 0; all numbers are test-half. Oracle = per-question
argmax over recorded outcomes — a noise-inflated ceiling (argmax over Bernoulli
draws), reported for scale only.

## barexam-70b  (n_test=200, arms: llm_only, judge, ce, scope)

| policy | acc / ktok / reward @ lam=0.0 | acc / ktok / reward @ lam=0.005 | acc / ktok / reward @ lam=0.01 | acc / ktok / reward @ lam=0.02 | acc / ktok / reward @ lam=0.05 |
|---|---|---|---|---|---|
| fixed:llm_only | 0.775 / 0.70 / 0.775 | 0.775 / 0.70 / 0.772 | 0.775 / 0.70 / 0.768 | 0.775 / 0.70 / 0.761 | 0.775 / 0.70 / 0.740 |
| fixed:judge | 0.755 / 1.53 / 0.755 | 0.755 / 1.53 / 0.747 | 0.755 / 1.53 / 0.740 | 0.755 / 1.53 / 0.724 | 0.755 / 1.53 / 0.679 |
| fixed:ce | 0.755 / 1.48 / 0.755 | 0.755 / 1.48 / 0.748 | 0.755 / 1.48 / 0.740 | 0.755 / 1.48 / 0.725 | 0.755 / 1.48 / 0.681 |
| fixed:scope | 0.760 / 1.73 / 0.760 | 0.760 / 1.73 / 0.751 | 0.760 / 1.73 / 0.743 | 0.760 / 1.73 / 0.725 | 0.760 / 1.73 / 0.673 |
| oracle (ceiling, noise-inflated) | 0.885 / 0.79 / 0.885 | 0.885 / 0.77 / 0.881 | 0.885 / 0.77 / 0.877 | 0.885 / 0.77 / 0.870 | 0.885 / 0.77 / 0.846 |
| gate:judge-max (tuned on train) | 0.775 / 0.91 / 0.775 | 0.775 / 0.88 / 0.771 | 0.775 / 0.88 / 0.766 | 0.775 / 0.88 / 0.757 | 0.775 / 0.88 / 0.731 |
| contextual (per-action logistic) | 0.735 / 1.11 / 0.735 | 0.730 / 1.05 / 0.725 | 0.735 / 0.98 / 0.725 | 0.765 / 0.89 / 0.747 | 0.760 / 0.78 / 0.721 |

- lam=0 headline: contextual 0.735 vs best fixed (`llm_only`) 0.775 — McNemar b/c=5/13, p=0.096
- contextual action agrees with per-question oracle action on 42.0% of test questions (oracle itself noise-inflated)
- gate:judge-max 0.775 vs best fixed — b/c=2/2, p=1.000

## barexam-8b  (n_test=200, arms: llm_only, judge, ce, scope)

| policy | acc / ktok / reward @ lam=0.0 | acc / ktok / reward @ lam=0.005 | acc / ktok / reward @ lam=0.01 | acc / ktok / reward @ lam=0.02 | acc / ktok / reward @ lam=0.05 |
|---|---|---|---|---|---|
| fixed:llm_only | 0.555 / 0.69 / 0.555 | 0.555 / 0.69 / 0.552 | 0.555 / 0.69 / 0.548 | 0.555 / 0.69 / 0.541 | 0.555 / 0.69 / 0.520 |
| fixed:judge | 0.645 / 1.46 / 0.645 | 0.645 / 1.46 / 0.638 | 0.645 / 1.46 / 0.630 | 0.645 / 1.46 / 0.616 | 0.645 / 1.46 / 0.572 |
| fixed:ce | 0.585 / 1.42 / 0.585 | 0.585 / 1.42 / 0.578 | 0.585 / 1.42 / 0.571 | 0.585 / 1.42 / 0.557 | 0.585 / 1.42 / 0.514 |
| fixed:scope | 0.650 / 1.66 / 0.650 | 0.650 / 1.66 / 0.642 | 0.650 / 1.66 / 0.633 | 0.650 / 1.66 / 0.617 | 0.650 / 1.66 / 0.567 |
| oracle (ceiling, noise-inflated) | 0.855 / 0.98 / 0.855 | 0.855 / 0.89 / 0.851 | 0.855 / 0.89 / 0.846 | 0.855 / 0.89 / 0.837 | 0.855 / 0.89 / 0.811 |
| gate:judge-max (tuned on train) | 0.650 / 1.65 / 0.650 | 0.650 / 1.65 / 0.642 | 0.650 / 1.65 / 0.634 | 0.650 / 1.65 / 0.617 | 0.620 / 1.42 / 0.549 |
| contextual (per-action logistic) | 0.625 / 1.61 / 0.625 | 0.625 / 1.61 / 0.617 | 0.625 / 1.61 / 0.609 | 0.620 / 1.60 / 0.588 | 0.615 / 1.59 / 0.535 |

- lam=0 headline: contextual 0.625 vs best fixed (`scope`) 0.650 — McNemar b/c=6/11, p=0.332
- contextual action agrees with per-question oracle action on 5.0% of test questions (oracle itself noise-inflated)
- gate:judge-max 0.650 vs best fixed — b/c=0/0, p=1.000

## housing-70b  (n_test=250, arms: llm_only, judge, ce, scope)

| policy | acc / ktok / reward @ lam=0.0 | acc / ktok / reward @ lam=0.005 | acc / ktok / reward @ lam=0.01 | acc / ktok / reward @ lam=0.02 | acc / ktok / reward @ lam=0.05 |
|---|---|---|---|---|---|
| fixed:llm_only | 0.536 / 0.25 / 0.536 | 0.536 / 0.25 / 0.535 | 0.536 / 0.25 / 0.534 | 0.536 / 0.25 / 0.531 | 0.536 / 0.25 / 0.524 |
| fixed:judge | 0.672 / 3.63 / 0.672 | 0.672 / 3.63 / 0.654 | 0.672 / 3.63 / 0.636 | 0.672 / 3.63 / 0.599 | 0.672 / 3.63 / 0.491 |
| fixed:ce | 0.636 / 2.60 / 0.636 | 0.636 / 2.60 / 0.623 | 0.636 / 2.60 / 0.610 | 0.636 / 2.60 / 0.584 | 0.636 / 2.60 / 0.506 |
| fixed:scope | 0.652 / 2.74 / 0.652 | 0.652 / 2.74 / 0.638 | 0.652 / 2.74 / 0.625 | 0.652 / 2.74 / 0.597 | 0.652 / 2.74 / 0.515 |
| oracle (ceiling, noise-inflated) | 0.796 / 0.93 / 0.796 | 0.796 / 0.76 / 0.792 | 0.796 / 0.76 / 0.788 | 0.796 / 0.76 / 0.781 | 0.796 / 0.76 / 0.758 |
| gate:judge-max (tuned on train) | 0.644 / 3.30 / 0.644 | 0.636 / 3.11 / 0.620 | 0.636 / 3.11 / 0.605 | 0.636 / 3.11 / 0.574 | 0.540 / 0.28 / 0.526 |
| contextual (per-action logistic) | 0.660 / 3.42 / 0.660 | 0.660 / 3.30 / 0.644 | 0.660 / 3.12 / 0.629 | 0.608 / 2.41 / 0.560 | 0.572 / 0.72 / 0.536 |

- lam=0 headline: contextual 0.660 vs best fixed (`judge`) 0.672 — McNemar b/c=4/7, p=0.549
- contextual action agrees with per-question oracle action on 18.0% of test questions (oracle itself noise-inflated)
- gate:judge-max 0.644 vs best fixed — b/c=2/9, p=0.065

## housing-8b  (n_test=250, arms: llm_only, judge, ce, scope)

| policy | acc / ktok / reward @ lam=0.0 | acc / ktok / reward @ lam=0.005 | acc / ktok / reward @ lam=0.01 | acc / ktok / reward @ lam=0.02 | acc / ktok / reward @ lam=0.05 |
|---|---|---|---|---|---|
| fixed:llm_only | 0.648 / 0.25 / 0.648 | 0.648 / 0.25 / 0.647 | 0.648 / 0.25 / 0.645 | 0.648 / 0.25 / 0.643 | 0.648 / 0.25 / 0.635 |
| fixed:judge | 0.624 / 3.72 / 0.624 | 0.624 / 3.72 / 0.605 | 0.624 / 3.72 / 0.587 | 0.624 / 3.72 / 0.550 | 0.624 / 3.72 / 0.438 |
| fixed:ce | 0.664 / 2.65 / 0.664 | 0.664 / 2.65 / 0.651 | 0.664 / 2.65 / 0.638 | 0.664 / 2.65 / 0.611 | 0.664 / 2.65 / 0.532 |
| fixed:scope | 0.636 / 2.81 / 0.636 | 0.636 / 2.81 / 0.622 | 0.636 / 2.81 / 0.608 | 0.636 / 2.81 / 0.580 | 0.636 / 2.81 / 0.495 |
| oracle (ceiling, noise-inflated) | 0.892 / 1.12 / 0.892 | 0.892 / 0.96 / 0.887 | 0.892 / 0.96 / 0.882 | 0.892 / 0.96 / 0.873 | 0.888 / 0.88 / 0.844 |
| gate:judge-max (tuned on train) | 0.628 / 1.56 / 0.628 | 0.624 / 1.49 / 0.617 | 0.640 / 0.55 / 0.634 | 0.640 / 0.55 / 0.629 | 0.652 / 0.29 / 0.637 |
| contextual (per-action logistic) | 0.652 / 1.32 / 0.652 | 0.652 / 1.21 / 0.646 | 0.644 / 1.06 / 0.633 | 0.640 / 0.82 / 0.624 | 0.640 / 0.48 / 0.616 |

- lam=0 headline: contextual 0.652 vs best fixed (`ce`) 0.664 — McNemar b/c=27/30, p=0.791
- contextual action agrees with per-question oracle action on 47.6% of test questions (oracle itself noise-inflated)
- gate:judge-max 0.628 vs best fixed — b/c=14/23, p=0.188

## medqa-70b  (n_test=637, arms: llm_only, raw, hyde, scope)

| policy | acc / ktok / reward @ lam=0.0 | acc / ktok / reward @ lam=0.005 | acc / ktok / reward @ lam=0.01 | acc / ktok / reward @ lam=0.02 | acc / ktok / reward @ lam=0.05 |
|---|---|---|---|---|---|
| fixed:llm_only | 0.846 / 0.65 / 0.846 | 0.846 / 0.65 / 0.843 | 0.846 / 0.65 / 0.840 | 0.846 / 0.65 / 0.833 | 0.846 / 0.65 / 0.814 |
| fixed:raw | 0.813 / 1.67 / 0.813 | 0.813 / 1.67 / 0.805 | 0.813 / 1.67 / 0.796 | 0.813 / 1.67 / 0.780 | 0.813 / 1.67 / 0.730 |
| fixed:hyde | 0.845 / 1.60 / 0.845 | 0.845 / 1.60 / 0.837 | 0.845 / 1.60 / 0.829 | 0.845 / 1.60 / 0.813 | 0.845 / 1.60 / 0.765 |
| fixed:scope | 0.843 / 1.61 / 0.843 | 0.843 / 1.61 / 0.835 | 0.843 / 1.61 / 0.827 | 0.843 / 1.61 / 0.811 | 0.843 / 1.61 / 0.762 |
| oracle (ceiling, noise-inflated) | 0.923 / 0.72 / 0.923 | 0.923 / 0.72 / 0.919 | 0.923 / 0.72 / 0.916 | 0.923 / 0.72 / 0.909 | 0.923 / 0.72 / 0.887 |
| contextual (per-action logistic) | 0.843 / 1.52 / 0.843 | 0.849 / 1.44 / 0.842 | 0.849 / 1.31 / 0.836 | 0.852 / 0.97 / 0.833 | 0.846 / 0.65 / 0.814 |

- lam=0 headline: contextual 0.843 vs best fixed (`llm_only`) 0.846 — McNemar b/c=23/25, p=0.885
- contextual action agrees with per-question oracle action on 11.1% of test questions (oracle itself noise-inflated)
