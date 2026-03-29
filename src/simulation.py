"""
Script de simulation : Chiarella baseline vs AdaptiveGamma (Option 2).

Paramètres calibrés en temps mensuel (dt = 1 mois) :
    - alpha  = 1/7  ≈ 0.143  : horizon des trend-followers ≈ 7 mois
    - kappa  = 0.08          : rappel fondamental (8 % / mois)
    - beta   = 0.10          : poids de la demande trend
    - gamma0 = 2.0           : saturation baseline
    - sigma_N = 0.05         : bruit de prix (≈ 5 % / mois, ≈ 17 % annuel)
    - sigma_V = 0.02         : bruit fondamental
    - T      = 1000 mois

Usage :
    python src/simulation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Résolution d'import que l'on exécute en standalone ou en module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import AdaptiveGammaModel, ChiarellaModel, DecreasingMispricingGammaModel, MispricingGammaModel, ModelParams, SimulationResult

# ---------------------------------------------------------------------------
# Paramètres standards (temps mensuel)
# ---------------------------------------------------------------------------

MONTHLY_PARAMS = ModelParams(
    kappa=0.08,
    beta=0.10,
    alpha=1 / 7,   # horizon ≈ 7 mois
    gamma0=2.0,
    sigma_N=0.05,
    sigma_V=0.02,
    g=0.0,
    dt=1.0,        # 1 mois par pas
    T=1000.0,      # 1 000 mois de simulation
)

# Fenêtre de volatilité réalisée : 12 mois (1 an glissant)
VOL_WINDOW_MONTHS: int = 12

# Graine commune pour comparer les mêmes réalisations du bruit
SHARED_SEED: int = 2024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def result_to_dataframe(result: SimulationResult, label: str = "") -> pd.DataFrame:
    """Convertit un SimulationResult (trajectoire unique) en DataFrame pandas.

    Parameters
    ----------
    result : SimulationResult
        Résultat d'une simulation avec n_paths=1.
    label : str
        Préfixe ajouté aux colonnes (ex: 'baseline' → 'baseline_P').

    Returns
    -------
    pd.DataFrame
        Colonnes : t, {label_}P, {label_}V, {label_}M, {label_}gamma.
    """
    if result.P.ndim != 1:
        raise ValueError(
            "result_to_dataframe attend une trajectoire unique (n_paths=1). "
            f"Got P.shape={result.P.shape}."
        )
    prefix = f"{label}_" if label else ""
    return pd.DataFrame(
        {
            "t": result.t,
            f"{prefix}P": result.P,
            f"{prefix}V": result.V,
            f"{prefix}M": result.M,
            f"{prefix}gamma": result.gamma,
        }
    )


def run_baseline(seed: int = SHARED_SEED) -> tuple[SimulationResult, pd.DataFrame]:
    """Simule le modèle baseline (γ constant).

    Parameters
    ----------
    seed : int
        Graine aléatoire.

    Returns
    -------
    (SimulationResult, pd.DataFrame)
    """
    model = ChiarellaModel(params=MONTHLY_PARAMS, seed=seed)
    result = model.simulate(n_paths=1)
    df = result_to_dataframe(result, label="baseline")
    return result, df


def run_adaptive(
    theta: float = 1.0,
    sigma_baseline: float | None = None,
    seed: int = SHARED_SEED,
) -> tuple[SimulationResult, pd.DataFrame]:
    """Simule le modèle adaptatif (γ linéaire en σ_réalisée).

    Parameters
    ----------
    theta : float
        Sensibilité γ ↔ volatilité (θ).
    sigma_baseline : float | None
        Volatilité de référence. Si None, estimée via un run baseline interne.
    seed : int
        Graine aléatoire (identique au baseline pour comparer les mêmes chocs).

    Returns
    -------
    (SimulationResult, pd.DataFrame)
    """
    model = AdaptiveGammaModel(
        params=MONTHLY_PARAMS,
        theta=theta,
        vol_window=VOL_WINDOW_MONTHS,
        sigma_baseline=sigma_baseline,
        seed=seed,
    )
    result = model.simulate(n_paths=1)
    df = result_to_dataframe(result, label="adaptive")
    return result, df


def run_mispricing(
    lambda_: float = 1.0,
    seed: int = SHARED_SEED,
) -> tuple[SimulationResult, pd.DataFrame]:
    """Simule le modèle mispricing (γ exponentiel en |P - V|).

    Parameters
    ----------
    lambda_ : float
        Sensibilité exponentielle au mispricing (λ).
    seed : int
        Graine aléatoire.

    Returns
    -------
    (SimulationResult, pd.DataFrame)
    """
    model = MispricingGammaModel(
        params=MONTHLY_PARAMS,
        lambda_=lambda_,
        seed=seed,
    )
    result = model.simulate(n_paths=1)
    df = result_to_dataframe(result, label="mispricing")
    return result, df


def run_decreasing_mispricing(
    lambda_: float = 1.0,
    seed: int = SHARED_SEED,
) -> tuple[SimulationResult, pd.DataFrame]:
    """Simule le modèle mispricing décroissant (γ = γ_0 / (1 + λ·|P−V|)).

    Parameters
    ----------
    lambda_ : float
        Sensibilité de la décroissance au mispricing (λ ≥ 0).
    seed : int
        Graine aléatoire.

    Returns
    -------
    (SimulationResult, pd.DataFrame)
    """
    model = DecreasingMispricingGammaModel(
        params=MONTHLY_PARAMS,
        lambda_=lambda_,
        seed=seed,
    )
    result = model.simulate(n_paths=1)
    df = result_to_dataframe(result, label="decreasing")
    return result, df


def merge_results(df_baseline: pd.DataFrame, df_adaptive: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les DataFrames baseline et adaptatif sur la colonne temporelle.

    Parameters
    ----------
    df_baseline, df_adaptive : pd.DataFrame
        DataFrames produits par `result_to_dataframe`.

    Returns
    -------
    pd.DataFrame
        DataFrame joint avec toutes les colonnes des deux modèles.
    """
    return pd.merge(df_baseline, df_adaptive, on="t", how="inner")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    """Lance les deux simulations et affiche un résumé comparatif.

    Returns
    -------
    pd.DataFrame
        DataFrame fusionné avec les trajectoires des deux modèles.
    """
    print("=" * 60)
    print("Simulation Chiarella étendu — temps mensuel (dt=1 mois)")
    print(f"T = {int(MONTHLY_PARAMS.T)} mois  |  seed = {SHARED_SEED}")
    print("=" * 60)

    # ----- Baseline -----
    print("\n[1/4] Modèle baseline  (γ = γ₀ = 2.0 constant) …")
    res_base, df_base = run_baseline(seed=SHARED_SEED)
    print(f"      → γ  : {res_base.gamma.mean():.4f} (constant)")
    print(f"      → P finale : {res_base.P[-1]:.4f}")

    # ----- Adaptatif -----
    print("\n[2/4] Modèle adaptatif (γ(t) = γ₀·(1+θ·σ/σ_base), θ=1.0) …")
    res_adap, df_adap = run_adaptive(theta=1.0, seed=SHARED_SEED)
    print(f"      → γ  : mean={res_adap.gamma.mean():.4f}  "
          f"min={res_adap.gamma.min():.4f}  max={res_adap.gamma.max():.4f}")
    print(f"      → P finale : {res_adap.P[-1]:.4f}")

    # ----- Mispricing croissant -----
    print("\n[3/4] Modèle mispricing croissant (γ(t) = γ₀·exp(λ·|P-V|), λ=1.0) …")
    res_misp, df_misp = run_mispricing(lambda_=1.0, seed=SHARED_SEED)
    print(f"      → γ  : mean={res_misp.gamma.mean():.4f}  "
          f"min={res_misp.gamma.min():.4f}  max={res_misp.gamma.max():.4f}")
    print(f"      → P finale : {res_misp.P[-1]:.4f}")

    # ----- Mispricing décroissant -----
    print("\n[4/4] Modèle mispricing décroissant (γ(t) = γ₀/(1+λ·|P-V|), λ=1.0) …")
    res_decr, df_decr = run_decreasing_mispricing(lambda_=1.0, seed=SHARED_SEED)
    print(f"      → γ  : mean={res_decr.gamma.mean():.4f}  "
          f"min={res_decr.gamma.min():.4f}  max={res_decr.gamma.max():.4f}")
    print(f"      → P finale : {res_decr.P[-1]:.4f}")

    # ----- Fusion -----
    df = merge_results(df_base, df_adap)
    df = pd.merge(df, df_misp, on="t", how="inner")
    df = pd.merge(df, df_decr, on="t", how="inner")

    # Rendements log-prix pour chaque modèle
    for label in ("baseline", "adaptive", "mispricing", "decreasing"):
        df[f"{label}_ret"] = df[f"{label}_P"].diff()

    # Résumé statistique rapide des rendements
    print("\n--- Statistiques des rendements mensuels ---")
    stats_cols = ["baseline_ret", "adaptive_ret", "mispricing_ret", "decreasing_ret"]
    print(df[stats_cols].describe().round(6).to_string())

    return df


if __name__ == "__main__":
    df = main()
    print(f"\nDataFrame final : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print("Colonnes :", list(df.columns))
