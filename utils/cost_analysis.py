"""
utils/cost_analysis.py
------------------------
Compute clinical cost savings from MHO feature selection.
Maps selected feature indices → diagnostic test costs (INR).
"""

from data_configs import DISEASE_CONFIGS


def compute_cost_analysis(feature_names: list, selected_indices,
                           disease_name: str) -> dict:
    """
    Parameters
    ----------
    feature_names    : list[str]  – all feature column names (before selection)
    selected_indices : array-like – indices chosen by the optimiser
    disease_name     : str        – key in DISEASE_CONFIGS

    Returns
    -------
    dict with keys:
      total_tests, selected_tests, removed_tests, test_reduction_pct
      total_cost, selected_cost, saved_cost, cost_reduction_pct
      selected_names, removed_names
    """
    config       = DISEASE_CONFIGS[disease_name]
    costs        = config.get("feature_costs", {})
    default_cost = config.get("default_cost", 200)

    def _cost(name: str) -> int:
        return costs.get(name, default_cost)

    total_cost    = sum(_cost(f) for f in feature_names)
    selected_names = [feature_names[i] for i in selected_indices]
    removed_names  = [f for f in feature_names if f not in selected_names]
    selected_cost  = sum(_cost(f) for f in selected_names)
    saved_cost     = total_cost - selected_cost

    n_total    = len(feature_names)
    n_selected = len(selected_names)

    test_reduction_pct = (
        (n_total - n_selected) / n_total * 100 if n_total > 0 else 0.0
    )
    cost_reduction_pct = (
        saved_cost / total_cost * 100 if total_cost > 0 else 0.0
    )

    return {
        "total_tests":         n_total,
        "selected_tests":      n_selected,
        "removed_tests":       n_total - n_selected,
        "test_reduction_pct":  round(test_reduction_pct, 1),
        "total_cost":          total_cost,
        "selected_cost":       selected_cost,
        "saved_cost":          saved_cost,
        "cost_reduction_pct":  round(cost_reduction_pct, 1),
        "selected_names":      selected_names,
        "removed_names":       removed_names,
    }
