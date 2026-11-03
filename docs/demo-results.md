# Demo evaluation results: withdrawn

The table that was here (Ensemble day-1 RMSE 0.107 °C, skill 1.000 on a
14-day window) came from an ensemble fitted on its own test window. It was
in-sample and is withdrawn.

Current numbers, from 365 daily walk-forward origins with bootstrap
intervals, and what they do and do not support: [EVALUATION-AUDIT.md](EVALUATION-AUDIT.md).
Short version: a regularised linear model beats persistence by about 8% RMSE
at days 1 to 3; gradient boosting does not measurably beat it; scores are
against the Open-Meteo archive, not a sensor.
