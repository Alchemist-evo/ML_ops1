"""Exploratory data analysis: saves figures to reports/figures.

Usage: python -m src.eda
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.data_processing import (CATEGORICAL_FEATURES, CLEAN_PATH,  # noqa: E402
                                 NUMERIC_FEATURES, TARGET)

FIG_DIR = Path("reports/figures")
sns.set_theme(style="whitegrid", palette="deep")


def _save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150)
    plt.close()


def class_balance(df):
    ax = sns.countplot(x=TARGET, data=df)
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Class balance")
    _save("class_balance.png")


def histograms(df):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax, bins=20)
        ax.set_title(f"Distribution of {col}")
    axes.ravel()[-1].axis("off")
    _save("histograms_numeric.png")


def categorical_counts(df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=col, hue=TARGET, ax=ax)
        ax.set_title(f"{col} vs target")
    _save("categorical_vs_target.png")


def correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    _save("correlation_heatmap.png")


def boxplots(df):
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax)
    _save("boxplots_numeric.png")


def run():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_PATH)
    for fn in (class_balance, histograms, categorical_counts,
               correlation_heatmap, boxplots):
        fn(df)
    print(df.describe().T.round(2))
    print("\nClass balance:\n", df[TARGET].value_counts(normalize=True).round(3))
    print("\nTop correlations with target:\n",
          df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).head(6).round(3))


if __name__ == "__main__":
    run()
