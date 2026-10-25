# METAR optional data must not crash the poller

Based on main27cb203. No changes to held comparison-date or daily-fetch candidates.

Actualpoll_once->SQLite regression tests: valid25Creading with singular dewpoint-243.04 previously raisesZeroDivisionError; now stores temperature with unknown humidity. Five malformed timestamps(nonfinite and outofrange) previously raise; now returnno_timebefore databasecreation. TestsinitiallyhadincorrectSQLcolumn/name, corrected before finalsource-onlyRED/GREEN. Finalbaseline6fail12pass->candidate18pass.

Fullpandas2.3.3environment340PASS. Actuallockedpandas3.1environment339PASS1FAIL: preexistingtests/test_results_freshness.py::test_saved_ensemble_scores_match_training_evaluation. No dependency or evaluation changes. Mergeholdremains; not fullgreen.
