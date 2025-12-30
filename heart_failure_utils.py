"""
Heart Failure Analysis Utilities
================================

This module provides a set of functions for analyzing heart failure clinical records.
It includes tools for:
- Exploratory Data Analysis (EDA)
- Survival Analysis (Kaplan-Meier estimates, Stratified analysis)
- Statistical Testing (Mann-Whitney U, Chi-square)
- Machine Learning (Model training, evaluation, feature importance)

Structure
---------
The module is organized into two main categories:
1. Computation Functions: Pure functions that process data and return results/metrics.
   Prefixes: `compute_`, `get_`, `train_`
2. Plotting Functions: Functions that visualize pre-computed data.
   Prefixes: `plot_`

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
>>>
>>> # 4. Grouped Analysis
>>> grouped_data = hfu.compute_grouped_kaplan_meier(df, group_col='sex', n_jobs=-1)
>>> hfu.plot_grouped_kaplan_meier(grouped_data)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator
from cycler import cycler
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from joblib import Parallel, delayed
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

# =============================================================================
# Configuration & Style
# =============================================================================

def set_plot_style():
    """
    Sets the matplotlib and seaborn style for the project.
    
    Returns:
        tuple: (selected_palette, div_cmap, gradient_cmap)
    """
    plt.style.use('ggplot')
    
    # Custom red-based color cycle
    selected_colors = [
        "#7F0000",  # deep red
        "#B30000",  # strong red
        "#CC4C4C",  # medium red
        "#E57373",  # soft red
        "#FF9999",  # pastel red
        "#660000",  # dark wine red
        "#990000",  # bright red
        "#FFCCCC",  # very light red
    ]
    
    selected_color_cycle = cycler('color', selected_colors)
    selected_palette = sns.color_palette(selected_colors)
    
    mpl.rcParams['axes.prop_cycle'] = selected_color_cycle
    sns.set_palette(selected_palette)
    
    # Custom colormaps
    div_cmap = LinearSegmentedColormap.from_list(
        "nc_div", ["#E57373", "#000000", "#FF001E"], N=256
    )
    gradient_cmap = LinearSegmentedColormap.from_list(
        "nc_gradient", ["#7F0000", "#FF9999"], N=256
    )
    
    pd.set_option('display.max_columns', 200)
    
    return selected_palette, div_cmap, gradient_cmap

# =============================================================================
# Exploratory Data Analysis (Visualization)
# =============================================================================

def plot_numeric_distributions(df, cols, hue=None, save_path='output'):
    """
    Plots side-by-side histograms (count and density) for numeric columns.
    
    Args:
        df (pd.DataFrame): Input data.
        cols (list): List of numeric column names to plot.
        hue (str, optional): Column name for color encoding.
        save_path (str, optional): Directory to save figures. Defaults to 'output'.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
    for col in cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Raw count histogram
        sns.histplot(data=df, x=col, kde=True, bins=100, hue=hue, ax=axes[0])
        axes[0].set_title(f'{col} (count)')
        
        # Normalised histogram per group
        sns.histplot(data=df, x=col, kde=True, bins=100, hue=hue,
                     stat='density', common_norm=False, ax=axes[1])
        axes[1].set_title(f'{col} (normalised)')
        
        plt.tight_layout()
        
        if save_path:
            fname = f"histogram_{col.lower().replace(' ', '_')}.png"
            plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')
        
        plt.show()

def plot_correlation_heatmap(df, cols, target_col=None, cmap="coolwarm", save_path='output'):
    """
    Plots correlation heatmaps. If target_col is provided, plots split by target class.
    
    Args:
        df (pd.DataFrame): Input data.
        cols (list): Numeric columns for correlation.
        target_col (str, optional): Binary target column to split analysis.
        cmap (str or Colormap): Colormap for heatmap.
        save_path (str, optional): Path to save the figure. Defaults to 'output'.
    """
    corr_all = df[cols].corr()
    vmin, vmax = -0.4, 0.4
    
    if target_col:
        # Assuming binary target 0/1
        corr_0 = df[df[target_col] == 0][cols].corr()
        corr_1 = df[df[target_col] == 1][cols].corr()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        sns.heatmap(corr_all, annot=True, square=True, cmap=cmap, center=0, vmin=vmin, vmax=vmax, ax=axes[0])
        axes[0].set_title('All patients')
        
        sns.heatmap(corr_0, annot=True, square=True, cmap=cmap, center=0, vmin=vmin, vmax=vmax, ax=axes[1])
        axes[1].set_title(f'{target_col}=0 (Alive)')
        
        sns.heatmap(corr_1, annot=True, square=True, cmap=cmap, center=0, vmin=vmin, vmax=vmax, ax=axes[2])
        axes[2].set_title(f'{target_col}=1 (Deceased)')
        
        fig.suptitle('Correlation heatmaps by outcome', fontsize=14)
    else:
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_all, annot=True, square=True, cmap=cmap, center=0, vmin=vmin, vmax=vmax)
        plt.title('Correlation heatmap')
        
    plt.tight_layout(rect=[0, 0, 1, 0.95] if target_col else None)
    
    if save_path:
        if os.path.isdir(os.path.dirname(save_path) or ".") and not save_path.endswith('.png'):
             os.makedirs(save_path, exist_ok=True)
             save_path_full = os.path.join(save_path, "correlation_heatmaps.png")
        else:
             save_path_full = save_path
        plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_pairplot(df, cols, hue=None, save_path='output'):
    """
    Creates a pairplot for selected variables.
    """
    g = sns.pairplot(df, vars=cols, hue=hue)
    g.fig.suptitle(f"Pairwise relationships (color coded by {hue})", fontsize=14, y=1.02)
    
    if save_path:
        if os.path.isdir(os.path.dirname(save_path) or ".") and not save_path.endswith('.png'):
             os.makedirs(save_path, exist_ok=True)
             save_path_full = os.path.join(save_path, "pairplot.png")
        else:
             save_path_full = save_path
        g.savefig(save_path_full, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_categorical_distributions(df, cols, hue=None, palette=None, save_path='output'):
    """
    Plots categorical variables: raw counts and outcome-normalized proportions.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
    for col in cols:
        prop_df = (
            df.groupby(col, observed=True)[hue]
            .value_counts(normalize=True)
            .rename("proportion")
            .reset_index()
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Raw counts
        sns.countplot(data=df, x=col, hue=hue, palette=palette, ax=axes[0])
        axes[0].set_title(f"{col} - raw counts")
        axes[0].tick_params(axis='x', rotation=45)
        for container in axes[0].containers:
            axes[0].bar_label(container)

        # Proportions
        sns.barplot(data=prop_df, x=col, y="proportion", hue=hue, palette=palette, ax=axes[1])
        axes[1].set_title(f"{col} - normalized proportions")
        axes[1].tick_params(axis='x', rotation=45)
        for container in axes[1].containers:
            axes[1].bar_label(container, fmt='%.2f')

        plt.tight_layout()
        
        if save_path:
            fname = f"barplot_{col}.png"
            plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')
        
        plt.show()

# =============================================================================
# Survival Analysis (Computation)
# =============================================================================

def _compute_km_arrays(df, duration_col, event_col, event_value):
    """Internal helper to compute basic KM arrays (Vectorized)."""
    # Group by time and count events (d_i) and censored (c_i)
    # We assume event_value indicates the event of interest (death)
    
    # 1. Filter relevant columns and sort
    df_subset = df[[duration_col, event_col]].sort_values(by=duration_col)
    
    # 2. Aggregate
    # d_i: count where event_col == event_value
    # c_i: count where event_col != event_value
    grp = df_subset.groupby(duration_col)[event_col].agg(
        d_i=lambda x: (x == event_value).sum(),
        c_i=lambda x: (x != event_value).sum()
    ).sort_index()
    
    times = grp.index.values
    d_i = grp['d_i'].values
    c_i = grp['c_i'].values
    
    # 3. Calculate At Risk (n_i)
    # n_i at step k is Total - sum(d_j + c_j for j < k)
    n_total = len(df_subset)
    removed = d_i + c_i
    removed_cumsum = np.cumsum(removed)
    # Shift to get removed *before* current time
    removed_prev = np.insert(removed_cumsum[:-1], 0, 0)
    at_risk = n_total - removed_prev
    
    # 4. Calculate Survival Probability
    # S(t) = S(t-1) * (1 - d_i / n_i)
    # We use numpy for vectorization
    
    # Handle division by zero if at_risk becomes 0
    with np.errstate(divide='ignore', invalid='ignore'):
        term = 1.0 - d_i / at_risk
        term[at_risk == 0] = 1.0 # No change if no one at risk
        
    survival_prob = np.cumprod(term)
    
    # 5. Prepend t=0, S=1
    final_times = np.insert(times, 0, 0)
    final_probs = np.insert(survival_prob, 0, 1.0)
    
    return final_times, final_probs

def _bootstrap_worker(df, duration_col, event_col, event_value, grid_times):
    """Internal helper for parallel bootstrapping."""
    resampled_df = df.sample(frac=1, replace=True)
    r_times, r_survivals = _compute_km_arrays(resampled_df, duration_col, event_col, event_value)
    
    # Interpolate to grid
    indices = np.searchsorted(r_times, grid_times, side='right') - 1
    indices = np.clip(indices, 0, len(r_survivals) - 1)
    return r_survivals[indices]

def compute_kaplan_meier(df, duration_col='time', event_col='DEATH_EVENT', event_value=1, 
                        bootstrap_ci=True, n_boot=200, n_jobs=1):
    """
    Computes Kaplan-Meier survival curve and optional bootstrap confidence intervals.
    
    Args:
        df (pd.DataFrame): Data.
        duration_col (str): Time column.
        event_col (str): Event column.
        event_value (int/str): Value indicating event occurrence.
        bootstrap_ci (bool): Whether to compute 95% CI.
        n_boot (int): Number of bootstrap iterations.
        n_jobs (int): Number of parallel jobs (-1 for all cores).
        
    Returns:
        dict: Contains 'times', 'survivals', 'ci_lower', 'ci_upper'.
    """
    times, survivals = _compute_km_arrays(df, duration_col, event_col, event_value)
    
    ci_lower, ci_upper = None, None
    
    if bootstrap_ci:
        grid_times = times
        
        if n_jobs != 1:
            boot_survivals = Parallel(n_jobs=n_jobs)(
                delayed(_bootstrap_worker)(df, duration_col, event_col, event_value, grid_times)
                for _ in range(n_boot)
            )
        else:
            boot_survivals = [
                _bootstrap_worker(df, duration_col, event_col, event_value, grid_times)
                for _ in range(n_boot)
            ]
            
        boot_survivals = np.array(boot_survivals)
        ci_lower = np.percentile(boot_survivals, 2.5, axis=0)
        ci_upper = np.percentile(boot_survivals, 97.5, axis=0)
    
    return {
        'times': times,
        'survivals': survivals,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def compute_grouped_kaplan_meier(df, group_col, time_col='time', event_col='DEATH_EVENT',
                               n_bins=4, is_numeric=False, target_day=365, 
                               bootstrap_ci=True, n_jobs=1):
    """
    Computes Kaplan-Meier curves for subgroups of data.
    
    Args:
        df (pd.DataFrame): Data.
        group_col (str): Column to group by.
        n_bins (int): Number of bins if grouping by numeric variable.
        is_numeric (bool): Treat group_col as numeric and bin it.
        target_day (int): Day to calculate survival probability for.
        
    Returns:
        dict: Structured results containing 'groups', 'km_data', 'metrics'.
    """
    df = df.copy()
    
    # Handle binning
    if is_numeric:
        binned_col = f"{group_col}_bin"
        try:
            df[binned_col] = pd.qcut(df[group_col], q=n_bins, duplicates='drop')
            group_col = binned_col
            groups = sorted(df[group_col].unique(), key=lambda x: x.left)
            labels = [f"[{iv.left:.2f}, {iv.right:.2f})" for iv in groups]
        except ValueError:
             groups = sorted(df[group_col].unique())
             labels = [str(g) for g in groups]
    else:
        groups = sorted(df[group_col].dropna().unique(), key=lambda x: str(x))
        labels = [str(g) for g in groups]
            
    results = {
        'group_col': group_col,
        'groups': [],
        'metrics': {'median_times': {}, 'survival_at_target': {}}
    }
    
    for idx, group in enumerate(groups):
        group_df = df[df[group_col] == group]
        label = labels[idx]
        
        if group_df.empty:
            continue
            
        # Compute KM
        km = compute_kaplan_meier(group_df, duration_col=time_col, event_col=event_col, 
                                 bootstrap_ci=bootstrap_ci, n_jobs=n_jobs)
        
        # Calculate metrics using the consistent helper
        stats = compute_km_statistics(km)
        median_time = stats['median_time']
            
        # 2. Survival at target day
        # Use step-function lookup (exact KM) instead of interpolation
        surv_prob = km['survivals']
        times = km['times']
        
        # Find index where times <= target_day
        # searchsorted(side='right') gives index i such that times[i-1] <= target_day < times[i]
        idx = np.searchsorted(times, target_day, side='right') - 1
        
        # Clip index to be safe (though 0 is always time=0, surv=1)
        idx = max(0, min(idx, len(surv_prob) - 1))
        surv_at_target = surv_prob[idx]
            
        results['groups'].append({
            'label': label,
            'km_data': km,
            'median_time': median_time,
            'surv_at_target': surv_at_target
        })
        
        results['metrics']['median_times'][label] = median_time
        results['metrics']['survival_at_target'][label] = 1 - surv_at_target # Probability of death
        
    return results

def compute_time_to_event_stats(df, time_col='time', event_col='DEATH_EVENT'):
    """
    Computes statistics for time-to-event for patients who experienced the event.
    """
    df_event = df[df[event_col] == 1]
    times = df_event[time_col].dropna()
    
    mean_time = times.mean()
    median_time = times.median()
    iqr = np.percentile(times, 75) - np.percentile(times, 25)
    
    # Bootstrap CI for mean
    boot_means = [times.sample(frac=1, replace=True).mean() for _ in range(1000)]
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    
    return {
        "mean": mean_time,
        "mean_ci": (ci_lower, ci_upper),
        "median": median_time,
        "iqr": iqr,
        "n": len(times),
        "times": times.values
    }

def compute_death_rate(df, n_days, time_col='time', event_col='DEATH_EVENT'):
    """
    Computes the naive death rate within n_days.
    """
    died_within = (df[event_col] == 1) & (df[time_col] <= n_days)
    survived_beyond = (df[time_col] >= n_days)
    
    valid = died_within | survived_beyond
    filtered_df = df[valid]
    
    if len(filtered_df) == 0:
        return 0.0
    
    # Numerator: People who died within n_days
    # Denominator: People with known status at n_days (died <= n_days OR survived >= n_days)
    n_died_within = died_within[valid].sum()
    n_total = len(filtered_df)
        
    return n_died_within / n_total

# =============================================================================
# Cox Proportional Hazards
# =============================================================================

def compute_cox_model(df, duration_col='time', event_col='DEATH_EVENT', penalizer=0.1):
    """
    Fits a Cox Proportional Hazards model and computes the C-index.
    """
    df = df.copy()
    # Ensure event is integer (0/1) to prevent get_dummies from encoding it
    df[event_col] = df[event_col].astype(int)
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Handle potential boolean columns by casting to int
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)
            
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_encoded, duration_col=duration_col, event_col=event_col)
    
    return cph

def train_eval_survival_models(df, time_col='time', event_col='DEATH_EVENT', test_size=0.2, random_state=42, tune_xgb=False):
    """
    Trains CoxPH and Survival XGBoost, evaluates with CV (C-index).
    """
    df = df.copy()
    df[event_col] = df[event_col].astype(int)
    
    X = df.drop(columns=[time_col, event_col])
    T = df[time_col]
    E = df[event_col]
    
    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()
    
    # Split
    X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
        X, T, E, test_size=test_size, random_state=random_state, stratify=E
    )
    
    # Default XGB params
    xgb_params = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'random_state': random_state
    }

    # Hyperparameter Tuning for Survival XGBoost
    if tune_xgb:
        print("Tuning XGBoost hyperparameters (Survival)...")
        import random
        
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        best_score = -1
        best_params = xgb_params.copy()
        
        # Simple Random Search
        n_iter = 10
        cv_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        
        # Generate random combinations
        keys = list(param_dist.keys())
        combinations = []
        for _ in range(n_iter):
            comb = {k: random.choice(param_dist[k]) for k in keys}
            combinations.append(comb)
            
        for i, params in enumerate(combinations):
            current_params = xgb_params.copy()
            current_params.update(params)
            
            fold_scores = []
            for tr_idx, val_idx in cv_tune.split(X_train, E_train):
                X_tr_t, X_val_t = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                T_tr_t, T_val_t = T_train.iloc[tr_idx], T_train.iloc[val_idx]
                E_tr_t, E_val_t = E_train.iloc[tr_idx], E_train.iloc[val_idx]
                
                y_tr_t = np.where(E_tr_t == 1, T_tr_t, -T_tr_t)
                
                model = xgb.XGBRegressor(**current_params)
                model.fit(X_tr_t, y_tr_t)
                
                pred = model.predict(X_val_t)
                try:
                    score = concordance_index(T_val_t, -pred, E_val_t)
                    fold_scores.append(score)
                except:
                    pass
            
            if fold_scores:
                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = current_params
        
        print(f"Best Survival XGBoost Params: {best_params}")
        xgb_params = best_params

    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cox_scores = []
    xgb_scores = []
    
    # Storage for KM curves across folds
    # We will store interpolated survival probabilities on a common time grid
    max_time = T_train.max()
    time_grid = np.linspace(0, max_time, 100)
    
    km_curves = {
        'CoxPH': {'Low': [], 'High': []},
        'Survival XGBoost': {'Low': [], 'High': []}
    }
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, E_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        T_tr, T_val = T_train.iloc[train_idx], T_train.iloc[val_idx]
        E_tr, E_val = E_train.iloc[train_idx], E_train.iloc[val_idx]
        
        # Helper to compute and store KM for a fold
        def process_fold_km(model_name, risk_scores, T_v, E_v):
            median_risk = np.median(risk_scores)
            low_mask = risk_scores <= median_risk
            high_mask = ~low_mask
            
            for group_name, mask in [('Low', low_mask), ('High', high_mask)]:
                if mask.sum() > 0:
                    kmf = KaplanMeierFitter()
                    kmf.fit(T_v[mask], event_observed=E_v[mask])
                    # Interpolate to common grid
                    # survival_function_ is indexed by time
                    surv_prob = np.interp(time_grid, kmf.survival_function_.index.values, kmf.survival_function_['KM_estimate'].values)
                    # Handle times before first event (prob=1) and after last (constant)
                    # np.interp does constant extrapolation by default if x is sorted, which time_grid is.
                    # But we need to be careful about the start.
                    # If time_grid[0] < min(index), interp gives min(index) value. 
                    # We want 1.0 for t=0.
                    # Let's force 1.0 at t=0 if not present
                    if time_grid[0] < kmf.survival_function_.index.min():
                         # Find indices where time_grid is less than first observed time
                         pre_indices = time_grid < kmf.survival_function_.index.min()
                         surv_prob[pre_indices] = 1.0
                    
                    km_curves[model_name][group_name].append(surv_prob)

        # --- CoxPH ---
        try:
            df_tr = X_tr.copy()
            df_tr[time_col] = T_tr
            df_tr[event_col] = E_tr
            
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(df_tr, duration_col=time_col, event_col=event_col)
            
            pred_cox = cph.predict_partial_hazard(X_val)
            c_index_cox = concordance_index(T_val, -pred_cox, E_val)
            cox_scores.append(c_index_cox)
            
            process_fold_km('CoxPH', pred_cox, T_val, E_val)
            
        except Exception:
            pass 

        # --- XGBoost Survival ---
        y_tr_xgb = np.where(E_tr == 1, T_tr, -T_tr)
        
        xgb_surv = xgb.XGBRegressor(**xgb_params)
        xgb_surv.fit(X_tr, y_tr_xgb)
        
        pred_xgb = xgb_surv.predict(X_val)
        c_index_xgb = concordance_index(T_val, -pred_xgb, E_val)
        xgb_scores.append(c_index_xgb)
        
        process_fold_km('Survival XGBoost', pred_xgb, T_val, E_val)

    # Aggregate Results
    cv_results = [
        {'Model': 'CoxPH', 'Metric': 'C-index', 'Mean Score': np.mean(cox_scores), 'Std Dev': np.std(cox_scores)},
        {'Model': 'Survival XGBoost', 'Metric': 'C-index', 'Mean Score': np.mean(xgb_scores), 'Std Dev': np.std(xgb_scores)}
    ]

    # Final Fit
    # Cox
    df_train_full = X_train.copy()
    df_train_full[time_col] = T_train
    df_train_full[event_col] = E_train
    final_cph = CoxPHFitter(penalizer=0.1)
    final_cph.fit(df_train_full, duration_col=time_col, event_col=event_col)
    
    # XGB
    y_train_full_xgb = np.where(E_train == 1, T_train, -T_train)
    final_xgb = xgb.XGBRegressor(**xgb_params)
    final_xgb.fit(X_train, y_train_full_xgb)
    
    return {
        'cv_results': pd.DataFrame(cv_results),
        'models': {'CoxPH': final_cph, 'Survival XGBoost': final_xgb},
        'feature_names': feature_names,
        'X_train': X_train,
        'T_train': T_train,
        'E_train': E_train,
        'km_curves': km_curves,
        'time_grid': time_grid
    }

def plot_survival_model_performance(results, palette=None, save_path='output'):
    df_res = results['cv_results']
    km_curves = results.get('km_curves')
    time_grid = results.get('time_grid')
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Setup Figure: 2 Rows. Top: Barplot. Bottom: Risk Stratification for each model.
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2)
    
    # --- 1. Bar Plot (Top, spanning both cols) ---
    ax_bar = fig.add_subplot(gs[0, :])
    pivot_mean = df_res.pivot(index='Model', columns='Metric', values='Mean Score')
    pivot_std = df_res.pivot(index='Model', columns='Metric', values='Std Dev')
    
    colors = [palette[0], palette[2]] if palette else None
    pivot_mean.plot(kind='bar', ax=ax_bar, yerr=pivot_std, capsize=4, 
                    color=colors, alpha=0.9, edgecolor='black', rot=0)
    
    ax_bar.set_title('Survival Model Performance (CV Mean C-index ± Std)')
    ax_bar.set_ylabel("C-index")
    ax_bar.set_ylim(0.5, 1.0)
    ax_bar.legend(loc='lower right')
    
    # --- 2. Risk Stratification Plots (Bottom) ---
    # Helper to plot KM for risk groups with CV error bars
    def plot_risk_strat_cv(model_name, ax):
        if model_name not in km_curves:
            return
            
        curves = km_curves[model_name]
        
        # Low Risk
        if curves['Low']:
            low_arr = np.array(curves['Low'])
            low_mean = np.mean(low_arr, axis=0)
            low_std = np.std(low_arr, axis=0)
            
            ax.plot(time_grid, low_mean, color=palette[0] if palette else 'green', label='Low Risk (Mean)')
            ax.fill_between(time_grid, low_mean - low_std, low_mean + low_std, color=palette[0] if palette else 'green', alpha=0.2)
            
        # High Risk
        if curves['High']:
            high_arr = np.array(curves['High'])
            high_mean = np.mean(high_arr, axis=0)
            high_std = np.std(high_arr, axis=0)
            
            ax.plot(time_grid, high_mean, color=palette[1] if palette else 'red', label='High Risk (Mean)')
            ax.fill_between(time_grid, high_mean - high_std, high_mean + high_std, color=palette[1] if palette else 'red', alpha=0.2)
            
        ax.set_title(f'{model_name}: Risk Stratification (CV Mean ± Std)')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Survival Probability')
        ax.set_ylim(0, 1)
        ax.legend()

    # Plot for Cox
    ax_cox = fig.add_subplot(gs[1, 0])
    plot_risk_strat_cv('CoxPH', ax_cox)
        
    # Plot for XGBoost
    ax_xgb = fig.add_subplot(gs[1, 1])
    plot_risk_strat_cv('Survival XGBoost', ax_xgb)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "survival_model_performance.png"), dpi=300, bbox_inches='tight')
    
    plt.show()

def compute_survival_feature_importance(model_data, random_state=42):
    """
    Computes feature importance via CV for survival models.
    """
    X = model_data['X_train']
    T = model_data['T_train']
    E = model_data['E_train']
    feats = model_data['feature_names']
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    cox_coefs = []
    xgb_imps = []
    
    for train_idx, _ in cv.split(X, E):
        X_f, T_f, E_f = X.iloc[train_idx], T.iloc[train_idx], E.iloc[train_idx]
        
        # Cox
        try:
            df_f = X_f.copy()
            df_f['time'] = T_f
            df_f['event'] = E_f
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(df_f, duration_col='time', event_col='event')
            cox_coefs.append(cph.params_)
        except:
            pass
            
        # XGB
        y_f_xgb = np.where(E_f == 1, T_f, -T_f)
        xgb_model = xgb.XGBRegressor(objective='survival:cox', eval_metric='cox-nloglik', random_state=random_state)
        xgb_model.fit(X_f, y_f_xgb)
        xgb_imps.append(xgb_model.feature_importances_)
        
    return {
        'cox': pd.DataFrame(cox_coefs), # columns are feats
        'xgb': pd.DataFrame(xgb_imps, columns=feats)
    }

def plot_survival_feature_importance(importance_data, palette=None, save_path='output'):
    """
    Plots top feature importances for survival models.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Cox
    df_cox = importance_data['cox']
    mean_cox = df_cox.mean()
    std_cox = df_cox.std()
    
    top_idx = mean_cox.abs().sort_values(ascending=False).head(10).index
    top_coefs = mean_cox[top_idx]
    top_std = std_cox[top_idx]
    
    plt.figure(figsize=(10, 6))
    top_coefs.plot(kind='barh', xerr=top_std, capsize=4, 
                   color=palette[0] if palette else 'blue', alpha=0.8, edgecolor='black')
    plt.title('Top 10 Features (CoxPH Coefficients)')
    plt.axvline(0, color='black', lw=0.8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "survival_feature_importance_cox.png"), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # XGB
    df_xgb = importance_data['xgb']
    mean_xgb = df_xgb.mean().sort_values(ascending=False).head(10)
    std_xgb = df_xgb.std()[mean_xgb.index]
    
    plt.figure(figsize=(10, 6))
    mean_xgb.plot(kind='barh', xerr=std_xgb, capsize=4, 
                  color=palette[0] if palette else 'blue', alpha=0.8, edgecolor='black')
    plt.title('Top 10 Features (Survival XGBoost Importance)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "survival_feature_importance_xgb.png"), dpi=300, bbox_inches='tight')
    
    plt.show()

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

def plot_cox_weights(cph, palette=None, save_path='output'):
    """
    Plots the hazard ratios (exp(coef)) and their confidence intervals.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 6))
    cph.plot()
    plt.title("Cox Model Coefficients (Log-Hazard Ratios) with 95% CI")
    plt.tight_layout()
    
    if save_path:
        if os.path.isdir(os.path.dirname(save_path) or ".") and not save_path.endswith('.png'):
             os.makedirs(save_path, exist_ok=True)
             save_path_full = os.path.join(save_path, "cox_weights.png")
        else:
             save_path_full = save_path
        plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
    
    plt.show()

# =============================================================================
# Survival Analysis (Visualization)
# =============================================================================

def compute_km_statistics(km_data):
    """
    Calculates statistics from Kaplan-Meier data (median, plateau, etc.).
    
    Args:
        km_data (dict): Result from compute_kaplan_meier.
        
    Returns:
        dict: Statistics including median_time, prob_1_year, etc.
    """
    times = km_data['times']
    survivals = km_data['survivals']
    
    # Cumulative death probability
    y_orig = 1 - survivals
    x_orig = times
    
    # Smooth the curve
    x_smooth = np.linspace(x_orig.min(), x_orig.max(), 1000)
    y_smooth = np.interp(x_smooth, x_orig, y_orig)
    
    # Stats
    plateau_height = y_orig.max()
    
    # Helper for step function lookup
    def get_prob_at_t(t, times, probs):
        if t > times.max():
            return probs[-1] # Assume constant after last event
        idx = np.searchsorted(times, t, side='right') - 1
        idx = max(0, min(idx, len(probs) - 1))
        return probs[idx]

    # Calculate exact probabilities from step function (y_orig = 1 - survival)
    prob_1_year = get_prob_at_t(365, x_orig, y_orig)
    prob_30_days = get_prob_at_t(30, x_orig, y_orig)
    
    # Median (time where cumulative death >= 50% of plateau)
    # We keep using smoothed curve for finding the intersection time to avoid discrete jumps
    target_val = plateau_height * 0.5
    indices = np.where(y_smooth >= target_val)[0]
    
    if len(indices) > 0:
        idx = indices[0]
        t_cross = x_smooth[idx]
        actual_val = y_smooth[idx]
    else:
        t_cross = None
        actual_val = None
        
    return {
        'median_time': t_cross,
        'median_cumulative_prob': actual_val,
        'plateau_height': plateau_height,
        'prob_1_year': prob_1_year,
        'prob_30_days': prob_30_days,
        'x_smooth': x_smooth,
        'y_smooth': y_smooth,
        'x_orig': x_orig,
        'y_orig': y_orig,
        'target_val': target_val
    }

def plot_kaplan_meier(km_data, palette=None, save_path='output'):
    """
    Plots a single Kaplan-Meier curve with annotations.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    times = km_data['times']
    ci_lower = km_data['ci_lower']
    ci_upper = km_data['ci_upper']
    
    # Calculate stats and smoothed curves
    stats = compute_km_statistics(km_data)
    x_orig = stats['x_orig']
    y_orig = stats['y_orig']
    x_smooth = stats['x_smooth']
    y_smooth = stats['y_smooth']
    t_cross = stats['median_time']
    actual_val = stats['median_cumulative_prob']
    plateau_height = stats['plateau_height']
    prob_30_days = stats.get('prob_30_days')
    target_val = stats['target_val']
    
    plt.figure(figsize=(7, 5))
    color = palette[1] if palette else "#B30000"
    
    # Plot curves
    plt.step(x_orig, y_orig, where="post", color=color, alpha=0.33, label='Raw curve')
    plt.plot(x_smooth, y_smooth, color=color, label='Smoothed curve')
    
    # Confidence Interval
    if ci_lower is not None:
        y_lo = 1 - ci_upper
        y_hi = 1 - ci_lower
        y_lo_smooth = np.interp(x_smooth, times, y_lo)
        y_hi_smooth = np.interp(x_smooth, times, y_hi)
        plt.fill_between(x_smooth, y_lo_smooth, y_hi_smooth, color=color, alpha=0.1)

    # Annotations
    plt.axhline(target_val, color='gray', linestyle='--', label=f'{int(0.5 * 100)}% of plateau')
    
    if t_cross is not None:
        plt.axvline(t_cross, color='gray', linestyle='--', label=f'Day {t_cross:.0f}')
        plt.scatter(t_cross, actual_val, color='black', zorder=5)
        plt.text(t_cross + 5, actual_val, f"~{t_cross:.0f} days ({actual_val:.1%})", 
                 va='bottom', ha='left', fontsize=9, alpha=0.5)

    if prob_30_days is not None:
        plt.text(30, prob_30_days, f"{prob_30_days:.1%} at 30 days", 
                 va='bottom', ha='left', fontsize=9, color='black', alpha=0.5)

    plt.text(x_orig[-1], plateau_height, f"plateau: {plateau_height:.1%}", 
             va='bottom', ha='right', fontsize=9, color='black', alpha=0.5)

    plt.xlabel("Follow-up period (days)")
    plt.ylabel("Cumulative probability of death")
    plt.title("Cumulative death curve (Kaplan–Meier estimate)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        if os.path.isdir(os.path.dirname(save_path) or ".") and not save_path.endswith('.png'):
             os.makedirs(save_path, exist_ok=True)
             save_path_full = os.path.join(save_path, "kaplan_meier_curve.png")
        else:
             save_path_full = save_path
        plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_grouped_kaplan_meier(grouped_data, target_day=365, save_path='output'):
    """
    Plots grouped KM curves, median times, and survival probabilities.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    groups = grouped_data['groups']
    n_groups = len(groups)
    
    # Colors
    cmap = LinearSegmentedColormap.from_list("custom_grad", ["#FF9999", "#7F0000"], N=256)
    colors = [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]
    
    fig, (ax_km, ax_med, ax_prob) = plt.subplots(
        3, 1, figsize=(7, 10), gridspec_kw={'height_ratios': [2.5, 1, 1]}
    )
    
    labels = []
    med_times = []
    probs_target = []
    
    for idx, g_data in enumerate(groups):
        label = g_data['label']
        km = g_data['km_data']
        
        labels.append(label)
        med_times.append(g_data['median_time'])
        probs_target.append(1 - g_data['surv_at_target']) # Death prob
        
        # Plot Curve (Cumulative Death)
        x = km['times']
        y = 1 - km['survivals']
        
        # Smooth
        if len(x) > 1:
            x_smooth = np.linspace(x.min(), x.max(), 500)
            y_smooth = np.interp(x_smooth, x, y)
            ax_km.plot(x_smooth, y_smooth, label=label, color=colors[idx])
            
            if km['ci_lower'] is not None:
                y_lo = 1 - km['ci_upper']
                y_hi = 1 - km['ci_lower']
                y_lo_s = np.interp(x_smooth, x, y_lo)
                y_hi_s = np.interp(x_smooth, x, y_hi)
                ax_km.fill_between(x_smooth, y_lo_s, y_hi_s, color=colors[idx], alpha=0.2)
        else:
            ax_km.plot(x, y, label=label, color=colors[idx])

    # 1. KM Plot
    ax_km.set_title(f"Cumulative death curves by {grouped_data['group_col']}")
    ax_km.set_xlabel("Days")
    ax_km.set_ylabel("Cumulative probability of death")
    ax_km.grid(True)
    ax_km.legend(loc='lower right')
    
    # 2. Median Time Bar Plot
    x_pos = range(len(labels))
    safe_med = [0 if np.isnan(x) else x for x in med_times]
    
    ax_med.bar(x_pos, safe_med, color=colors, width=0.5)
    ax_med.set_title("Median time to death (days)")
    ax_med.set_ylabel("Days")
    ax_med.set_xticks(x_pos)
    ax_med.set_xticklabels(labels, rotation=45, ha="right")
    
    # Adjust ylim to fit labels
    if safe_med:
        max_val = max(safe_med)
        if max_val > 0:
            ax_med.set_ylim(0, max_val * 1.2)

    for i, val in enumerate(med_times):
        text = f"{val:.0f}" if not np.isnan(val) else "NR"
        # Place text slightly above bar
        offset = max(safe_med) * 0.02 if safe_med and max(safe_med) > 0 else 1
        y_txt = val + offset if not np.isnan(val) else offset
        ax_med.text(i, y_txt, text, ha='center', va='bottom', fontsize=9)
        
    # 3. Prob at Target Bar Plot
    ax_prob.bar(x_pos, probs_target, color=colors, width=0.5)
    ax_prob.set_title(f"Cumulative death probability at {target_day} days")
    ax_prob.set_ylabel("Probability")
    
    # Adjust ylim to fit labels
    if probs_target:
        max_prob = max(probs_target)
        if max_prob > 0:
            ax_prob.set_ylim(0, min(1.05, max_prob * 1.25))
        else:
            ax_prob.set_ylim(0, 1.0)
    
    ax_prob.set_xticks(x_pos)
    ax_prob.set_xticklabels(labels, rotation=45, ha="right")
    
    for i, val in enumerate(probs_target):
        ax_prob.text(i, val + 0.01, f"{val:.1%}", ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    
    if save_path:
        if os.path.isdir(os.path.dirname(save_path) or ".") and not save_path.endswith('.png'):
             os.makedirs(save_path, exist_ok=True)
             fname = f"grouped_kaplan_meier_{grouped_data['group_col']}.png"
             save_path_full = os.path.join(save_path, fname)
        else:
             save_path_full = save_path
        plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_time_to_event_distribution(stats, palette=None, save_path='output'):
    """
    Plots the distribution of time-to-event.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    times = stats['times']
    mean_val = stats['mean']
    median_val = stats['median']
    
    plt.figure(figsize=(8, 5))
    color = palette[1] if palette else "#B30000"
    
    sns.histplot(times, kde=True, color=color, alpha=0.5, bins=20, label='Observed events')
    
    plt.axvline(mean_val, color='black', linestyle='--', label=f"Mean: {mean_val:.1f}")
    plt.axvline(median_val, color='red', linestyle='-.', label=f"Median: {median_val:.0f}")
    
    plt.title(f"Distribution of Time to Event (n={len(times)})")
    plt.xlabel("Time (days)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        if os.path.isdir(os.path.dirname(save_path) or ".") and not save_path.endswith('.png'):
             os.makedirs(save_path, exist_ok=True)
             save_path_full = os.path.join(save_path, "time_to_event_distribution.png")
        else:
             save_path_full = save_path
        plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
    
    plt.show()

# =============================================================================
# Statistical Analysis
# =============================================================================

def compute_variable_associations(df, target_col='DEATH_EVENT', numeric_cols=None, 
                                categorical_cols=None, p_threshold=0.05):
    """
    Computes statistical associations (Mann-Whitney U, Chi-square).
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.drop([target_col, "time"], errors='ignore').tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["category", "object"]).columns.drop([target_col], errors='ignore').tolist()
        
    results = {'numeric': [], 'categorical': []}
    
    # Numeric
    for col in numeric_cols:
        g1 = df[df[target_col] == 1][col]
        g0 = df[df[target_col] == 0][col]
        try:
            stat, p = mannwhitneyu(g1, g0)
            if p < p_threshold:
                results['numeric'].append({'col': col, 'p_value': p})
        except Exception:
            pass
            
    # Categorical
    for col in categorical_cols:
        try:
            ct = pd.crosstab(df[col], df[target_col])
            chi2, p, dof, ex = chi2_contingency(ct)
            if p < p_threshold:
                results['categorical'].append({'col': col, 'p_value': p})
        except Exception:
            pass
            
    return results

def plot_variable_associations(df, associations, target_col='DEATH_EVENT', palette=None, save_path='output'):
    """
    Plots significant associations found by compute_variable_associations.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Numeric
    for item in associations['numeric']:
        col = item['col']
        p = item['p_value']
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x=target_col, y=col, hue=target_col, legend=False, palette=palette[:2] if palette else None)
        plt.title(f"{col} by {target_col} (p={p:.3g})")
        plt.tight_layout()
        
        if save_path:
            fname = f"boxplot_{col}.png"
            plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')
        
        plt.show()
        
    # Categorical
    for item in associations['categorical']:
        col = item['col']
        p = item['p_value']
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=col, hue=target_col, palette=palette[:2] if palette else None)
        plt.title(f"{col} distribution by {target_col} (p={p:.3g})")
        plt.tight_layout()
        
        if save_path:
            fname = f"countplot_{col}.png"
            plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')
        
        plt.show()

# =============================================================================
# Machine Learning
# =============================================================================

def train_eval_models(df, target_col='DEATH_EVENT', test_size=0.2, random_state=42, xgb_params=None, tune_xgb=False):
    """
    Trains Logistic Regression and XGBoost, evaluates with CV and Test set.
    """
    # Drop time column to avoid data leakage (and target)
    drop_cols = [target_col]
    if 'time' in df.columns:
        drop_cols.append('time')
        
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # Ensure target is numeric (it might be categorical)
    if y.dtype.name == 'category':
        y = y.astype(int)

    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Calculate class imbalance for XGBoost
    # scale_pos_weight = count(negative) / count(positive)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # Default XGB params
    final_xgb_params = {
        'scale_pos_weight': scale_pos_weight,
        'eval_metric': 'logloss',
        'random_state': random_state
    }
    
    # Hyperparameter Tuning if requested
    if tune_xgb:
        print("Tuning XGBoost hyperparameters (Classification)...")
        param_dist = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 4, 5, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5],
            'min_child_weight': [1, 3, 5]
        }
        
        xgb_model = xgb.XGBClassifier(**final_xgb_params)
        
        search = RandomizedSearchCV(
            xgb_model, 
            param_distributions=param_dist, 
            n_iter=15, 
            scoring='accuracy', 
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
            verbose=0,
            random_state=random_state,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        print(f"Best XGBoost Params: {search.best_params_}")
        final_xgb_params.update(search.best_params_)

    if xgb_params:
        final_xgb_params.update(xgb_params)

    # Models
    # Use Pipeline for LR to avoid data leakage during CV (scaling inside folds)
    models = {
        'Logistic Regression': make_pipeline(
            StandardScaler(), 
            LogisticRegression(class_weight='balanced', random_state=random_state)
        ),
        'XGBoost': xgb.XGBClassifier(**final_xgb_params)
    }
    
    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    metrics = {'Accuracy': 'accuracy', 'Recall': 'recall', 'ROC-AUC': 'roc_auc'}
    
    cv_results = []
    for name, model in models.items():
        for m_name, scoring in metrics.items():
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            cv_results.append({
                'Model': name, 'Metric': m_name, 
                'Mean Score': scores.mean(), 'Std Dev': scores.std()
            })
            
    # Final Fit and Confusion Matrix
    confusion_matrices = {}
    y_probs = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_probs[name] = model.predict_proba(X_test)[:, 1]
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    
    return {
        'cv_results': pd.DataFrame(cv_results),
        'y_test': y_test,
        'y_probs': y_probs,
        'confusion_matrices': confusion_matrices,
        'X_train': X_train,
        'y_train': y_train,
        'feature_names': feature_names
    }

def tune_xgboost(df, target_col='DEATH_EVENT', n_iter=10, random_state=42):
    """
    Performs RandomizedSearchCV for XGBoost.
    """
    # Drop time column to avoid data leakage (and target)
    drop_cols = [target_col]
    if 'time' in df.columns:
        drop_cols.append('time')
        
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # Ensure target is numeric
    if y.dtype.name == 'category':
        y = y.astype(int)

    X = pd.get_dummies(X, drop_first=True)
    
    # Class imbalance
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'min_child_weight': [1, 3, 5]
    }
    
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight, 
        eval_metric='logloss', 
        random_state=random_state
    )
    
    search = RandomizedSearchCV(
        xgb_model, 
        param_distributions=param_dist, 
        n_iter=n_iter, 
        scoring='accuracy', 
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )
    
    search.fit(X, y)
    
    return search.best_params_, search.best_score_

def plot_model_performance(results, palette=None, save_path='output'):
    """
    Plots CV performance metrics, ROC curves, and Confusion Matrices.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    df_res = results['cv_results']
    y_test = results['y_test']
    y_probs = results['y_probs']
    confusion_matrices = results.get('confusion_matrices', {})
    
    # Layout: 2 rows. Row 1: Bar chart + ROC. Row 2: Confusion Matrices.
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_roc = fig.add_subplot(gs[0, 1])
    
    # 1. Bar Chart
    pivot_mean = df_res.pivot(index='Model', columns='Metric', values='Mean Score')
    pivot_std = df_res.pivot(index='Model', columns='Metric', values='Std Dev')
    
    colors = [palette[0], palette[2], palette[4]] if palette else None
    pivot_mean.plot(kind='bar', ax=ax_bar, yerr=pivot_std, capsize=4, 
                    color=colors, alpha=0.9, edgecolor='black', rot=0)
    ax_bar.set_title('CV Model Performance (Mean ± Std)')
    ax_bar.set_ylabel("Score")
    ax_bar.set_ylim(0, 1.1)
    ax_bar.legend(loc='lower right')
    
    # 2. ROC
    colors_roc = [palette[0], palette[2]] if palette else ['blue', 'red']
    for i, (name, prob) in enumerate(y_probs.items()):
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})', 
                     color=colors_roc[i % len(colors_roc)], linewidth=2)
        
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_title('ROC Curves (Test Set)')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc='lower right')
    
    # 3. Confusion Matrices
    # Create custom colormap if palette is provided
    if palette:
        # Use the deep red from the palette (index 0) and white
        cmap = LinearSegmentedColormap.from_list("custom_reds", ["#FFFFFF", palette[0]], N=256)
    else:
        cmap = 'Reds'

    model_names = list(confusion_matrices.keys())
    for i, name in enumerate(model_names):
        if i < 2:
            ax_cm = fig.add_subplot(gs[1, i])
            cm = confusion_matrices[name]
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Survived', 'Died'])
            disp.plot(ax=ax_cm, cmap=cmap, colorbar=False)
            ax_cm.set_title(f'Confusion Matrix: {name}')
            ax_cm.grid(False)
            
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "model_performance.png"), dpi=300, bbox_inches='tight')
    
    plt.show()

def compute_feature_importance(model_data, random_state=42):
    """
    Computes feature importance via CV for stability.
    """
    X = model_data['X_train']
    y = model_data['y_train']
    feats = model_data['feature_names']
    
    # Calculate class imbalance for XGBoost (same as in train_eval_models)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    xgb_imps = []
    lr_coefs = []
    
    for train_idx, _ in cv.split(X, y):
        X_f, y_f = X.iloc[train_idx], y.iloc[train_idx]
        
        # XGB
        m_xgb = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=random_state)
        m_xgb.fit(X_f, y_f)
        xgb_imps.append(m_xgb.feature_importances_)
        
        # LR
        scaler = StandardScaler()
        X_fs = scaler.fit_transform(X_f)
        m_lr = LogisticRegression(class_weight='balanced', random_state=random_state)
        m_lr.fit(X_fs, y_f)
        lr_coefs.append(m_lr.coef_[0])
        
    return {
        'xgb': pd.DataFrame(xgb_imps, columns=feats),
        'lr': pd.DataFrame(lr_coefs, columns=feats)
    }

def plot_feature_importance(importance_data, palette=None, save_path='output'):
    """
    Plots top feature importances.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # XGB
    df_xgb = importance_data['xgb']
    mean_xgb = df_xgb.mean().sort_values(ascending=False).head(10)
    std_xgb = df_xgb.std()[mean_xgb.index]
    
    plt.figure(figsize=(10, 6))
    mean_xgb.plot(kind='barh', xerr=std_xgb, capsize=4, 
                  color=palette[0] if palette else 'blue', alpha=0.8, edgecolor='black')
    plt.title('Top 10 Features (XGBoost Importance)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "feature_importance_xgb.png"), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # LR
    df_lr = importance_data['lr']
    mean_lr = df_lr.mean()
    std_lr = df_lr.std()
    
    top_idx = mean_lr.abs().sort_values(ascending=False).head(10).index
    top_coefs = mean_lr[top_idx]
    top_std = std_lr[top_idx]
    
    plt.figure(figsize=(10, 6))
    top_coefs.plot(kind='barh', xerr=top_std, capsize=4, 
                   color=palette[0] if palette else 'blue', alpha=0.8, edgecolor='black')
    plt.title('Top 10 Features (Logistic Regression Coefficients)')
    plt.axvline(0, color='black', lw=0.8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "feature_importance_lr.png"), dpi=300, bbox_inches='tight')
    
    plt.show()

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

