"""
Fonctions d'analyse statistique des trajectoires simulées.

Statistiques disponibles :
    - Rendements log-prix (différences premières)
    - Volatilité réalisée glissante (annualisée ou dans l'unité du modèle)
    - Distribution des rendements : moyenne, vol, skewness, kurtosis excédentaire
    - Tableau comparatif multi-modèles
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Rendements
# ---------------------------------------------------------------------------

def compute_returns(prices: np.ndarray | pd.Series) -> np.ndarray:
    """Calcule les rendements log-prix (différences premières).

    Le rendement au pas i est défini comme :
        r_i = P_i - P_{i-1}

    (P étant déjà le log-prix, la différence est le log-rendement brut.)

    Parameters
    ----------
    prices : array-like, shape (n,)
        Série de log-prix.

    Returns
    -------
    np.ndarray, shape (n-1,)
        Série des rendements (premier élément supprimé par la différence).
    """
    arr = np.asarray(prices, dtype=float)
    return np.diff(arr)


# ---------------------------------------------------------------------------
# Volatilité réalisée
# ---------------------------------------------------------------------------

def rolling_volatility(
    returns: np.ndarray | pd.Series,
    window: int,
    annualization_factor: float = 1.0,
) -> np.ndarray:
    """Volatilité réalisée glissante sur une fenêtre de `window` pas.

    σ_réalisée(t) = std(r_{t-window+1}, …, r_t) × annualization_factor

    Avant d'avoir accumulé `window` observations, la fenêtre est élargie
    progressivement (expanding window).

    Parameters
    ----------
    returns : array-like, shape (n,)
        Série des rendements.
    window : int
        Taille de la fenêtre glissante (en nombre de pas).
    annualization_factor : float
        Facteur d'annualisation. Pour convertir une vol mensuelle en annuelle,
        passer `np.sqrt(12)`. Pour du journalier vers annuel, `np.sqrt(252)`.
        Défaut : 1.0 (pas d'annualisation, vol dans l'unité du modèle).

    Returns
    -------
    np.ndarray, shape (n,)
        Volatilité réalisée glissante, NaN sur les premières valeurs si
        `min_periods` n'est pas satisfait.
    """
    s = pd.Series(np.asarray(returns, dtype=float))
    vol = s.rolling(window=window, min_periods=2).std()
    return (vol * annualization_factor).to_numpy()


# ---------------------------------------------------------------------------
# Statistiques de distribution
# ---------------------------------------------------------------------------

def return_statistics(
    returns: np.ndarray | pd.Series,
    annualization_factor: float = 1.0,
    label: str = "",
) -> dict[str, float]:
    """Statistiques de distribution des rendements.

    Calcule :
        - mean      : moyenne des rendements (annualisée)
        - vol       : écart-type (annualisé)
        - skewness  : asymétrie (sans unité)
        - kurt_exc  : kurtosis excédentaire (= kurtosis - 3, ≡ 0 pour la normale)
        - jarque_bera_pval : p-valeur du test de normalité Jarque-Bera

    Parameters
    ----------
    returns : array-like, shape (n,)
        Série des rendements (NaN ignorés).
    annualization_factor : float
        Facteur d'annualisation pour la moyenne et la vol.
        Exemple : `np.sqrt(12)` pour une vol mensuelle → annuelle.
    label : str
        Préfixe ajouté aux clés du dictionnaire (utile pour comparaison multi-modèles).

    Returns
    -------
    dict[str, float]
        Dictionnaire des statistiques.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]

    mean_val = float(np.mean(r)) * annualization_factor
    vol_val  = float(np.std(r, ddof=1)) * annualization_factor
    skew_val = float(stats.skew(r))
    kurt_val = float(stats.kurtosis(r, fisher=True))  # kurtosis excédentaire (Fisher)

    jb_stat, jb_pval = stats.jarque_bera(r)

    prefix = f"{label}_" if label else ""
    return {
        f"{prefix}mean":              mean_val,
        f"{prefix}vol":               vol_val,
        f"{prefix}skewness":          skew_val,
        f"{prefix}kurt_excess":       kurt_val,
        f"{prefix}jarque_bera_pval":  float(jb_pval),
        f"{prefix}n_obs":             len(r),
    }


# ---------------------------------------------------------------------------
# Comparaison multi-modèles
# ---------------------------------------------------------------------------

def compare_models(
    returns_dict: dict[str, np.ndarray],
    annualization_factor: float = 1.0,
) -> pd.DataFrame:
    """Tableau comparatif des statistiques de rendement pour plusieurs modèles.

    Parameters
    ----------
    returns_dict : dict[str, array-like]
        Dictionnaire `{nom_modèle: série_rendements}`.
        Exemple : {"baseline": r_base, "adaptive": r_adap}
    annualization_factor : float
        Facteur d'annualisation commun.

    Returns
    -------
    pd.DataFrame
        Une ligne par modèle, colonnes : mean, vol, skewness, kurt_excess,
        jarque_bera_pval, n_obs.
    """
    rows = []
    for name, returns in returns_dict.items():
        stats_d = return_statistics(returns, annualization_factor=annualization_factor)
        rows.append({"model": name, **stats_d})
    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# Utilitaire : extraction du signal γ(t)
# ---------------------------------------------------------------------------

def gamma_statistics(gamma: np.ndarray, label: str = "") -> dict[str, float]:
    """Statistiques descriptives du processus γ(t).

    Parameters
    ----------
    gamma : array-like, shape (n,)
        Série temporelle du paramètre de saturation.
    label : str
        Préfixe des clés.

    Returns
    -------
    dict[str, float]
        mean, std, min, max, median de γ(t).
    """
    g = np.asarray(gamma, dtype=float)
    prefix = f"{label}_" if label else ""
    return {
        f"{prefix}gamma_mean":   float(np.mean(g)),
        f"{prefix}gamma_std":    float(np.std(g, ddof=1)),
        f"{prefix}gamma_min":    float(np.min(g)),
        f"{prefix}gamma_max":    float(np.max(g)),
        f"{prefix}gamma_median": float(np.median(g)),
    }


# ---------------------------------------------------------------------------
# Rapport complet (utilisé en fin de simulation)
# ---------------------------------------------------------------------------

def full_report(
    df: pd.DataFrame,
    baseline_col: str = "baseline_P",
    adaptive_col: str = "adaptive_P",
    baseline_gamma_col: str = "baseline_gamma",
    adaptive_gamma_col: str = "adaptive_gamma",
    annualization_factor: float = 1.0,
) -> pd.DataFrame:
    """Génère un rapport statistique complet depuis le DataFrame de simulation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produit par `simulation.merge_results`, contenant les colonnes
        *_P, *_V, *_M, *_gamma pour les deux modèles.
    baseline_col, adaptive_col : str
        Noms des colonnes de log-prix.
    baseline_gamma_col, adaptive_gamma_col : str
        Noms des colonnes γ(t).
    annualization_factor : float
        Facteur d'annualisation pour les statistiques de rendements.

    Returns
    -------
    pd.DataFrame
        Tableau de comparaison avec les statistiques des rendements et de γ(t).
    """
    r_base = compute_returns(df[baseline_col].to_numpy())
    r_adap = compute_returns(df[adaptive_col].to_numpy())

    stats_base = return_statistics(r_base, annualization_factor=annualization_factor)
    stats_adap = return_statistics(r_adap, annualization_factor=annualization_factor)

    g_stats_base = gamma_statistics(df[baseline_gamma_col].to_numpy())
    g_stats_adap = gamma_statistics(df[adaptive_gamma_col].to_numpy())

    report = pd.DataFrame(
        {
            "baseline": {**stats_base, **g_stats_base},
            "adaptive": {**stats_adap, **g_stats_adap},
        }
    ).T
    return report
