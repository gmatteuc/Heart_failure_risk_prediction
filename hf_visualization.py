"""
Heart Failure Analysis - Visualization Module
=============================================

This module provides functions for visualizing data and analysis results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

# Import helper from analysis module
from hf_analysis import compute_km_statistics

# =============================================================================
# Plot Saving
# =============================================================================

def _save_plot(save_path, default_filename):
    """
    Internal helper to handle plot saving logic.
    Supports both directory paths (appends default_filename) and file paths.
    """
    if not save_path:
        return
        
    # Check if save_path looks like a file (has extension)
    if save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
        # It's a file path
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        save_path_full = save_path
    else:
        # It's a directory
        os.makedirs(save_path, exist_ok=True)
        save_path_full = os.path.join(save_path, default_filename)
        
    plt.savefig(save_path_full, dpi=300, bbox_inches='tight')

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
        
        fname = f"histogram_{col.lower().replace(' ', '_')}.png"
        _save_plot(save_path, fname)
        
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
    
    _save_plot(save_path, "correlation_heatmaps.png")
    
    plt.show()

def plot_pairplot(df, cols, hue=None, save_path='output'):
    """
    Creates a pairplot for selected variables.
    """
    g = sns.pairplot(df, vars=cols, hue=hue)
    g.fig.suptitle(f"Pairwise relationships (color coded by {hue})", fontsize=14, y=1.02)
    
    _save_plot(save_path, "pairplot.png")
    
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
        
        fname = f"barplot_{col.lower().replace(' ', '_')}.png"
        _save_plot(save_path, fname)
        
        plt.show()

# =============================================================================
# Survival Analysis (Visualization)
# =============================================================================

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
    
    _save_plot(save_path, "survival_model_performance.png")
    
    plt.show()

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
    
    _save_plot(save_path, "survival_feature_importance_cox.png")
    
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
    
    _save_plot(save_path, "survival_feature_importance_xgb.png")
    
    plt.show()

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
    
    _save_plot(save_path, "cox_weights.png")
    
    plt.show()

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
    
    _save_plot(save_path, "kaplan_meier_curve.png")
    
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
        fname = f"grouped_kaplan_meier_{grouped_data['group_col'].lower().replace(' ', '_')}.png"
        _save_plot(save_path, fname)
    
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
    
    _save_plot(save_path, "time_to_event_distribution.png")
    
    plt.show()

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
            fname = f"boxplot_{col.lower().replace(' ', '_')}.png"
            _save_plot(save_path, fname)
        
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
            fname = f"countplot_{col.lower().replace(' ', '_')}.png"
            _save_plot(save_path, fname)
        
        plt.show()

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
    
    _save_plot(save_path, "model_performance.png")
    
    plt.show()

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
    
    _save_plot(save_path, "feature_importance_xgb.png")
    
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
    
    _save_plot(save_path, "feature_importance_lr.png")
    
    plt.show()
