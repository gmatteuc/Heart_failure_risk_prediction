"""
Heart Failure Analysis Utilities
================================

This module serves as the main entry point for the Heart Failure Analysis project.
It aggregates functionality from:
- hf_analysis: Computation and modeling
- hf_visualization: Plotting and visualization

Usage
-----
>>> import heart_failure_utils as hfu
>>> # 1. Setup
>>> palette, _, _ = hfu.set_plot_style()
>>>
>>> # 2. EDA
>>> hfu.plot_numeric_distributions(df, numeric_cols=['age', 'creatinine_phosphokinase'])
>>>
>>> # 3. Survival Analysis
>>> km_data = hfu.compute_kaplan_meier(df, duration_col='time', event_col='DEATH_EVENT')
>>> hfu.plot_kaplan_meier(km_data)
"""

# Import everything from sub-modules to expose them at top-level
from hf_analysis import (
    compute_kaplan_meier,
    compute_grouped_kaplan_meier,
    compute_time_to_event_stats,
    compute_death_rate,
    compute_cox_model,
    train_eval_survival_models,
    compute_survival_feature_importance,
    compute_km_statistics,
    compute_variable_associations,
    train_eval_models,
    tune_xgboost,
    compute_feature_importance
)

from hf_visualization import (
    set_plot_style,
    plot_numeric_distributions,
    plot_correlation_heatmap,
    plot_pairplot,
    plot_categorical_distributions,
    plot_survival_model_performance,
    plot_survival_feature_importance,
    plot_cox_weights,
    plot_kaplan_meier,
    plot_grouped_kaplan_meier,
    plot_time_to_event_distribution,
    plot_variable_associations,
    plot_model_performance,
    plot_feature_importance
)

# =============================================================================
# Notebook Wrappers
# =============================================================================

def analyze_kaplan_meier(df, duration_col='time', event_col='DEATH_EVENT', plot=True, palette=None, save_path='output', **kwargs):
    """Wrapper for backward compatibility."""
    km = compute_kaplan_meier(df, duration_col, event_col, **kwargs)
    stats = compute_km_statistics(km)
    
    if plot:
        plot_kaplan_meier(km, palette, save_path)
        
    return stats

def analyze_grouped_kaplan_meier(df, group_col, time_col='time', event_col='DEATH_EVENT', 
                               n_bins=4, is_numeric=False, target_day=365, 
                               save_path='output', plot=True, bootstrap_ci=True, n_jobs=1):
    """Wrapper for backward compatibility."""
    res = compute_grouped_kaplan_meier(df, group_col, time_col, event_col, 
                                     n_bins, is_numeric, target_day, bootstrap_ci, n_jobs)
    if plot:
        plot_grouped_kaplan_meier(res, target_day, save_path)
    return res['metrics']

def analyze_time_to_death(df, time_col='time', event_col='DEATH_EVENT', plot=False, palette=None, save_path='output'):
    """Wrapper for backward compatibility."""
    stats = compute_time_to_event_stats(df, time_col, event_col)
    if plot:
        plot_time_to_event_distribution(stats, palette, save_path)
    return stats

def analyze_variable_associations(df, target_col='DEATH_EVENT', numeric_cols=None, 
                                categorical_cols=None, p_threshold=0.05, plot=True, palette=None, save_path='output'):
    """Wrapper for backward compatibility."""
    res = compute_variable_associations(df, target_col, numeric_cols, categorical_cols, p_threshold)
    if plot:
        plot_variable_associations(df, res, target_col, palette, save_path)
    return res

def compare_models_performance(df, target_col='DEATH_EVENT', test_size=0.2, random_state=42, plot=True, palette=None, xgb_params=None, tune_xgb=False, save_path='output'):
    """Wrapper for backward compatibility."""
    res = train_eval_models(df, target_col, test_size, random_state, xgb_params, tune_xgb)
    if plot:
        plot_model_performance(res, palette, save_path)
    return res

def analyze_feature_importance(model_data, random_state=42, palette=None, save_path='output'):
    """Wrapper for backward compatibility."""
    imps = compute_feature_importance(model_data, random_state)
    plot_feature_importance(imps, palette, save_path)

def analyze_survival_feature_importance(results, palette=None, random_state=42, save_path='output'):
    """Wrapper for survival feature importance."""
    imps = compute_survival_feature_importance(results, random_state)
    plot_survival_feature_importance(imps, palette, save_path)

def compare_survival_models_performance(df, time_col='time', event_col='DEATH_EVENT', test_size=0.2, random_state=42, plot=True, palette=None, tune_xgb=False, save_path='output'):
    """Wrapper for survival model comparison."""
    res = train_eval_survival_models(df, time_col, event_col, test_size, random_state, tune_xgb)
    if plot:
        plot_survival_model_performance(res, palette, save_path)
    return res

