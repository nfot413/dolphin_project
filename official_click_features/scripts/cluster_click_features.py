from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_CSV = PROJECT_ROOT / "official_click_features" / "features" / "official_click_features.csv"

FIG_DIR = PROJECT_ROOT / "official_click_features" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = PROJECT_ROOT / "official_click_features" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLUMNS = [
    "original_duration_sec",
    "official_ici",
    "rms_energy",
    "peak_amplitude",
    "mean_abs_amplitude",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_85_hz",
    "dominant_frequency_hz",
    "low_freq_est_hz",
    "high_freq_est_hz",
    "bandwidth_est_hz",
    "estimated_peak_count",
    "estimated_click_rate_per_sec",
    "estimated_mean_ici_sec",
    "estimated_min_ici_sec",
]


def load_features() -> tuple[pd.DataFrame, list[str]]:
    if not FEATURE_CSV.exists():
        raise FileNotFoundError(f"Feature CSV not found: {FEATURE_CSV}")

    df = pd.read_csv(FEATURE_CSV)

    available = [c for c in FEATURE_COLUMNS if c in df.columns]

    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=available).reset_index(drop=True)

    print(f"[INFO] Loaded {len(df)} click train samples.")
    print("[INFO] Using features:")
    for col in available:
        print(f"  - {col}")

    return df, available


def plot_histograms(df: pd.DataFrame, feature_cols: list[str]) -> None:
    for col in feature_cols:
        data = df[col].dropna()
        if len(data) == 0:
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=40)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"hist_{col}.png", dpi=200)
        plt.close()

    print(f"[OK] Saved histograms to {FIG_DIR}")


def run_pca(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray, PCA]:
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    result = df.copy()
    result["PC1"] = X_pca[:, 0]
    result["PC2"] = X_pca[:, 1]

    print("[INFO] PCA explained variance:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.3f}")

    plt.figure(figsize=(7, 6))
    plt.scatter(result["PC1"], result["PC2"], s=18)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of official click train features")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_plain.png", dpi=200)
    plt.close()

    return result, X_scaled, pca


def choose_k(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    rows = []

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)

        rows.append({"k": k, "silhouette_score": score})
        print(f"[INFO] k={k}, silhouette={score:.4f}")

    scores = pd.DataFrame(rows)
    scores.to_csv(OUT_DIR / "click_kmeans_silhouette_scores.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(scores["k"], scores["silhouette_score"], marker="o")
    plt.xlabel("Number of clusters k")
    plt.ylabel("Silhouette score")
    plt.title("KMeans cluster number selection for click trains")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "click_kmeans_silhouette_scores.png", dpi=200)
    plt.close()

    return scores


def run_kmeans(pca_df: pd.DataFrame, X_scaled: np.ndarray, k: int) -> pd.DataFrame:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)

    result = pca_df.copy()
    result["cluster"] = labels

    result.to_csv(OUT_DIR / "official_click_features_with_clusters.csv", index=False)

    plt.figure(figsize=(7, 6))
    for cluster_id in sorted(result["cluster"].unique()):
        sub = result[result["cluster"] == cluster_id]
        plt.scatter(sub["PC1"], sub["PC2"], s=18, label=f"Cluster {cluster_id}")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"KMeans clustering of click train features, k={k}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"click_pca_kmeans_k{k}.png", dpi=200)
    plt.close()

    return result


def summarize_clusters(result: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    summary = result.groupby("cluster")[feature_cols].mean()
    counts = result["cluster"].value_counts().sort_index()
    summary.insert(0, "count", counts)

    out_path = OUT_DIR / "click_cluster_feature_summary.csv"
    summary.to_csv(out_path)

    print("[OK] Cluster summary:")
    print(summary)
    print(f"[OK] Saved cluster summary -> {out_path}")

    return summary


def create_cluster_profile_chart(summary: pd.DataFrame) -> None:
    selected = [
        "official_ici",
        "estimated_click_rate_per_sec",
        "estimated_peak_count",
        "rms_energy",
        "spectral_centroid_hz",
        "bandwidth_est_hz",
    ]

    selected = [c for c in selected if c in summary.columns]

    profile = summary[selected].copy()

    normalized = profile.copy()
    for col in selected:
        min_v = normalized[col].min()
        max_v = normalized[col].max()
        if max_v > min_v:
            normalized[col] = (normalized[col] - min_v) / (max_v - min_v)
        else:
            normalized[col] = 0.0

    plt.figure(figsize=(10, 5))
    for cluster_id in normalized.index:
        plt.plot(
            selected,
            normalized.loc[cluster_id],
            marker="o",
            label=f"Cluster {cluster_id}",
        )

    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Normalized feature value")
    plt.title("Click train cluster acoustic profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "click_cluster_profile_chart.png", dpi=200)
    plt.close()


def create_cluster_count_chart(result: pd.DataFrame) -> None:
    counts = result["cluster"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.title("Click train cluster counts")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "click_cluster_counts.png", dpi=200)
    plt.close()


def main() -> None:
    df, feature_cols = load_features()

    plot_histograms(df, feature_cols)

    pca_df, X_scaled, _ = run_pca(df, feature_cols)

    scores = choose_k(X_scaled, 2, 8)
    best_k = int(scores.sort_values("silhouette_score", ascending=False).iloc[0]["k"])

    print(f"[INFO] Best k by silhouette: {best_k}")

    result = run_kmeans(pca_df, X_scaled, best_k)
    summary = summarize_clusters(result, feature_cols)

    create_cluster_profile_chart(summary)
    create_cluster_count_chart(result)

    print("[OK] Done.")
    print(f"[OK] Figures saved in: {FIG_DIR}")
    print(f"[OK] Results saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()