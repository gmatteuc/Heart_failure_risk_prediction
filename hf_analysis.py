"""
Heart Failure Analysis - Computation Module
===========================================

This module contains pure functions for data processing, statistical analysis,
survival analysis computations, and machine learning model training.
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from joblib import Parallel, delayed
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import random

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
# Cox Proportional Hazards & Survival Models
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
                    if time_grid[0] < kmf.survival_function_.index.min():
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
            # Note: lifelines.utils.concordance_index expects scores where higher values 
            # indicate longer survival times. Since hazard is inversely related to survival 
            # (higher hazard = shorter survival), we must use the negative hazard.
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
        # Note: XGBoost with 'survival:cox' predicts log-hazard ratios (higher = higher risk).
        # We negate predictions for concordance_index which expects higher scores = longer survival.
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

# =============================================================================
# Machine Learning (Classification)
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
