from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_CSV = (
    PROJECT_ROOT
    / "official_whistle_features"
    / "features"
    / "official_whistle_features.csv"
)

OUT_DIR = PROJECT_ROOT / "official_whistle_features" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_DIR = PROJECT_ROOT / "official_whistle_features" / "features"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLUMNS = [
    "duration_sec",
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
]


def load_features() -> pd.DataFrame:
    if not FEATURE_CSV.exists():
        raise FileNotFoundError(f"Feature CSV not found: {FEATURE_CSV}")

    df = pd.read_csv(FEATURE_CSV)

    existing_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not existing_cols:
        raise ValueError("No usable feature columns found.")

    for col in existing_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=existing_cols).reset_index(drop=True)

    print(f"[INFO] Loaded {len(df)} whistle samples.")
    print(f"[INFO] Using features:")
    for col in existing_cols:
        print(f"  - {col}")

    return df, existing_cols


def plot_feature_histograms(df: pd.DataFrame, feature_cols: list[str]) -> None:
    for col in feature_cols:
        plt.figure(figsize=(8, 5))
        plt.hist(df[col].dropna(), bins=30)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"hist_{col}.png", dpi=200)
        plt.close()

    print(f"[OK] Saved feature histograms to {OUT_DIR}")


def run_pca(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = df.copy()
    pca_df["PC1"] = X_pca[:, 0]
    pca_df["PC2"] = X_pca[:, 1]

    print("[INFO] PCA explained variance ratio:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.3f}")

    return pca_df, X_scaled


def plot_pca_plain(pca_df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(pca_df["PC1"], pca_df["PC2"], s=25)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of official whistle features")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pca_plain.png", dpi=200)
    plt.close()


def choose_k_by_silhouette(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    rows = []

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)

        rows.append(
            {
                "k": k,
                "silhouette_score": score,
            }
        )

        print(f"[INFO] k={k}, silhouette={score:.4f}")

    score_df = pd.DataFrame(rows)
    score_df.to_csv(RESULT_DIR / "kmeans_silhouette_scores.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(score_df["k"], score_df["silhouette_score"], marker="o")
    plt.xlabel("Number of clusters k")
    plt.ylabel("Silhouette score")
    plt.title("KMeans cluster number selection")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "kmeans_silhouette_scores.png", dpi=200)
    plt.close()

    return score_df


def run_kmeans(pca_df: pd.DataFrame, X_scaled: np.ndarray, k: int = 3) -> pd.DataFrame:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)

    result_df = pca_df.copy()
    result_df["cluster"] = labels

    result_df.to_csv(RESULT_DIR / "official_whistle_features_with_clusters.csv", index=False)

    plt.figure(figsize=(7, 6))
    for cluster_id in sorted(result_df["cluster"].unique()):
        sub = result_df[result_df["cluster"] == cluster_id]
        plt.scatter(sub["PC1"], sub["PC2"], s=25, label=f"Cluster {cluster_id}")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"KMeans clustering of whistle features, k={k}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"pca_kmeans_k{k}.png", dpi=200)
    plt.close()

    return result_df


def summarize_clusters(result_df: pd.DataFrame, feature_cols: list[str]) -> None:
    summary = result_df.groupby("cluster")[feature_cols].mean()
    counts = result_df["cluster"].value_counts().sort_index()

    summary.insert(0, "count", counts)

    out_path = RESULT_DIR / "cluster_feature_summary.csv"
    summary.to_csv(out_path)

    print("[OK] Cluster summary:")
    print(summary)
    print(f"[OK] Saved cluster summary -> {out_path}")


def main() -> None:
    df, feature_cols = load_features()

    plot_feature_histograms(df, feature_cols)

    pca_df, X_scaled = run_pca(df, feature_cols)
    plot_pca_plain(pca_df)

    score_df = choose_k_by_silhouette(X_scaled, k_min=2, k_max=8)

    best_k = int(score_df.sort_values("silhouette_score", ascending=False).iloc[0]["k"])
    print(f"[INFO] Best k by silhouette score: {best_k}")

    # 你也可以手动改成 k=3 或 k=4
    result_df = run_kmeans(pca_df, X_scaled, k=best_k)

    summarize_clusters(result_df, feature_cols)

    print()
    print("[OK] Done.")
    print(f"[OK] Figures saved in: {OUT_DIR}")
    print(f"[OK] Result CSV files saved in: {RESULT_DIR}")


if __name__ == "__main__":
    main()