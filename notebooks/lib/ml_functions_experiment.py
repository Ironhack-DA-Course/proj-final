# Data handling
import numpy as np
import pandas as pd
import re

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import graphviz
import optuna
import optuna.visualization as vis


# Metrics
from sklearn.inspection import permutation_importance

#lib
from typing import Optional, Any
#


def bar_labels(axes, rotation=0, location="edge", xlabel=None, ylabel=None):
    '''Add labels to bars in a bar plot and configure axes'''
    for container in axes.containers:
        axes.bar_label(container, label_type=location, rotation=rotation)
    if xlabel:
        axes.set_xlabel(xlabel)
    if ylabel:
        axes.set_ylabel(ylabel)


 # Import Any and Optional

def get_feature_importance(
    model: Any,
    x_test:  pd.DataFrame | np.ndarray,
    y_test:  pd.DataFrame | np.ndarray,
    method: str = "auto",
    feature_names: Optional[list] = None,
    top_n: int = 8,
    plot: bool = True,
    figsize: tuple = (6, 4),
    random_state: int = 0
) -> pd.DataFrame:
    # Determine method automatically if 'auto'
    if method == "auto":
        if hasattr(model, "feature_importances_"):
            method = "builtin"
        elif hasattr(model, "coef_"):
            method = "coefficients"
        else:
            method = "permutation"

    importance_scores = None
    importance_std = None
    method_name = ""
    sort_column = "" # Define a variable to store the column to sort by

    # Calculate importance based on method
    if method == "builtin":
        # Tree-based models (RandomForest, XGBoost, LightGBM, etc.)
        importance_scores = model.feature_importances_
        method_name = "Built-in Feature Importance"
        sort_column = "importance"

    elif method == "coefficients":
        # Linear models (LogisticRegression, LinearRegression, etc.)
        coef = model.coef_
        if coef.ndim > 1:
            # Multi-class classification - take mean of absolute values
            importance_scores = np.mean(np.abs(coef), axis=0)
        else:
            importance_scores = np.abs(coef)

        method_name = "Coefficient-based Importance"
        sort_column = "importance"

    elif method == "permutation":
        # Permutation importance - works with any model
        perm_importance = permutation_importance(
            model, x_test, y_test,
            n_repeats=5,  # Reduced for faster processing
            random_state=random_state,
            n_jobs=-1
        )
        importance_scores = perm_importance.importances_mean
        importance_std = perm_importance.importances_std
        method_name = "Permutation Importance"
        sort_column = "permutation"

    else:
        raise ValueError(f"Unknown method: {method}. Use 'auto', 'builtin', 'permutation', or 'coefficients'")

    feature_names = x_test.columns.tolist() if feature_names is None else feature_names

    # Create results DataFrame
    results_data = {
        "feature": feature_names,
        "importance": importance_scores
    }

    if importance_std is not None:
        results_data["importance_std"] = importance_std
        if method == "permutation":
             results_data["permutation"] = importance_scores
             results_data.pop("importance") # Remove the 'importance' column


    importance_df = pd.DataFrame(results_data).sort_values(sort_column, ascending=False) # Sort by the determined column
    model_name = type(model).__name__

    # Create visualization
    if plot:
        plt.figure(figsize=figsize)

        # Get top N features
        top_features = importance_df.head(top_n)
        ax = sns.barplot(
            data=top_features,
            x=sort_column, # Use the determined column name for the x-axis
            y="feature",
            hue="feature",
            # alpha=0.8,
            legend=False)

        # Customize plot
        ax.set_xlabel("Importance Score", fontsize=10)
        ax.set_ylabel("Features", fontsize=10)
        ax.set_title(f"{model_name} - {method_name} - Top {min(top_n, len(top_features))} Features",
                     fontsize=10, fontweight="bold")
        for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3)
        plt.tight_layout()
        plt.show()
    return importance_df

def plot_metrics_comparison(metrics_dict: dict, metric_names: str | list, figsize=(16, 16)):
    '''Plot multiple metrics in subplots using seaborn with swapped axes'''
    n_metrics = len(metric_names)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() #if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, metric_name in enumerate(metric_names):
        if i < len(axes):
            ax = axes[i]
            df_metric = pd.DataFrame.from_dict(metrics_dict[metric_name], orient="index", columns=[metric_name])
            df_metric = df_metric.sort_values(metric_name, ascending=False)
            df_metric[metric_name] = df_metric[metric_name].round(2)

            sns.barplot(x=metric_name, y=df_metric.index, data=df_metric, ax=ax, color="skyblue")

            ax.set_xlabel(f"{metric_name.title()} (%)")
            ax.set_ylabel("Model")
            ax.set_title(f"{metric_name.title()} Comparison")

            # Add value labels on the bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3)

    # Hide unused subplots
    for i in range(len(metric_names), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(cms, df_metrics: pd.DataFrame | np.ndarray, metric_for_title: str="Recall"):
    '''Plot confusion matrices in rows of up to 4'''
    model_names = list(cms.keys())
    n_models = len(model_names)
    index = 0

    while index < n_models:
        n_cols = min(4, n_models - index)
        fig, axes = plt.subplots(ncols=n_cols, figsize=(3 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]

        for i in range(n_cols):
            if index < n_models:
                model_name = model_names[index]
                sns.heatmap(cms[model_name], annot=True, fmt="d", ax=axes[i], cbar=False)

                # Get metric value for title
                metric_value = df_metrics.loc[model_name, metric_for_title] if model_name in df_metrics.index else "N/A"
                axes[i].set_title(f"{model_name}: {metric_value}%")
                axes[i].set_xlabel("Predicted")
                axes[i].set_ylabel("True")
                index += 1

        plt.tight_layout()
        plt.show()