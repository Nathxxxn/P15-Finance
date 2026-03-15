"""
Modèles Agent-Based de Chiarella étendu (Majewski, Ciliberti, Bouchaud, 2020).

Système dynamique en temps continu (intégration Euler-Maruyama) :

    dP_t = κ(V_t - P_t)dt + β·tanh(γ(t)·M_t)dt + σ_N·dW_t^(N)
    dM_t = -α·M_t·dt + α·dP_t
    dV_t = g·dt + σ_V·dW_t^(V)

Variables d'état :
    P_t  : log-prix
    M_t  : signal de tendance (EMA des rendements)
    V_t  : log-valeur fondamentale

Paramètre de saturation γ(t) :
    - Baseline          : γ(t) = γ_0  (ChiarellaModel)
    - Adaptatif linéaire : γ(t) = γ_0·(1 + θ·σ_réalisée(t)/σ_baseline)  (AdaptiveGammaModel)
    - Mispricing         : γ(t) = γ_0·exp(λ·|P_t - V_t|)  (MispricingGammaModel)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Conteneur des paramètres du modèle
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    """Paramètres du modèle de Chiarella étendu.

    Attributes
    ----------
    kappa : float
        Force de rappel des fondamentalistes (κ > 0).
    beta : float
        Poids de la demande des trend-followers (β).
    alpha : float
        Inverse de l'horizon temporel des trend-followers (α > 0).
    gamma0 : float
        Paramètre de saturation de base γ_0.
    sigma_N : float
        Volatilité du bruit de prix (σ_N).
    sigma_V : float
        Volatilité du processus de valeur fondamentale (σ_V).
    g : float
        Drift de la valeur fondamentale.
    dt : float
        Pas de temps pour l'intégration Euler-Maruyama (en années).
    T : float
        Horizon de simulation total (en années).
    """
    kappa: float = 0.1
    beta: float = 1.5
    alpha: float = 0.5
    gamma0: float = 1.0
    sigma_N: float = 0.02
    sigma_V: float = 0.01
    g: float = 0.0
    dt: float = 1 / 252          # pas journalier (environ 252 jours ouvrés/an)
    T: float = 20.0              # 20 ans de simulation


# ---------------------------------------------------------------------------
# Résultat d'une simulation
# ---------------------------------------------------------------------------

class SimulationResult(NamedTuple):
    """Trajectoires simulées par le modèle.

    Attributes
    ----------
    t : np.ndarray, shape (n_steps,)
        Grille temporelle.
    P : np.ndarray, shape (n_steps,) ou (n_paths, n_steps)
        Log-prix.
    M : np.ndarray, même shape que P
        Signal de tendance.
    V : np.ndarray, même shape que P
        Log-valeur fondamentale.
    gamma : np.ndarray, même shape que P
        Valeur de γ(t) à chaque instant.
    """
    t: np.ndarray
    P: np.ndarray
    M: np.ndarray
    V: np.ndarray
    gamma: np.ndarray


# ---------------------------------------------------------------------------
# Classe de base : gamma constant
# ---------------------------------------------------------------------------

class ChiarellaModel:
    """Modèle de Chiarella étendu avec γ constant (baseline).

    Intégration numérique par schéma d'Euler-Maruyama :

        P_{t+dt} = P_t + [κ(V_t - P_t) + β·tanh(γ·M_t)]·dt + σ_N·√dt·ξ_t^(N)
        M_{t+dt} = M_t + α·(dP_t - M_t·dt)
                 = M_t·(1 - α·dt) + α·dP_t
        V_{t+dt} = V_t + g·dt + σ_V·√dt·ξ_t^(V)

    où ξ^(N) et ξ^(V) sont des bruits blancs gaussiens standards indépendants.

    Parameters
    ----------
    params : ModelParams
        Paramètres du modèle.
    seed : int | None
        Graine aléatoire pour la reproductibilité.
    """

    def __init__(self, params: ModelParams | None = None, seed: int | None = None) -> None:
        self.params = params or ModelParams()
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Méthode principale de simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        n_paths: int = 1,
        P0: float = 0.0,
        M0: float = 0.0,
        V0: float = 0.0,
    ) -> SimulationResult:
        """Lance la simulation Monte Carlo sur `n_paths` trajectoires.

        Parameters
        ----------
        n_paths : int
            Nombre de trajectoires indépendantes.
        P0, M0, V0 : float
            Conditions initiales communes à toutes les trajectoires.

        Returns
        -------
        SimulationResult
            Trajectoires simulées.
        """
        p = self.params
        n_steps = int(p.T / p.dt)
        sqrt_dt = np.sqrt(p.dt)

        # Grille temporelle
        t = np.linspace(0.0, p.T, n_steps + 1)

        # Pré-allouer les tableaux (n_paths × n_steps+1)
        shape = (n_paths, n_steps + 1)
        P = np.empty(shape)
        M = np.empty(shape)
        V = np.empty(shape)
        gamma_arr = np.empty(shape)

        # Conditions initiales
        P[:, 0] = P0
        M[:, 0] = M0
        V[:, 0] = V0
        gamma_arr[:, 0] = self._gamma(P, M, V, step=0)

        # Bruits gaussiens pré-générés pour toutes les trajectoires d'un coup
        # shape : (n_paths, n_steps)
        xi_N = self.rng.standard_normal((n_paths, n_steps))
        xi_V = self.rng.standard_normal((n_paths, n_steps))

        # Boucle temporelle (séquentielle car γ peut dépendre du passé)
        for i in range(n_steps):
            gam = self._gamma(P, M, V, step=i)
            gamma_arr[:, i] = gam

            # Incrément de prix
            drift_P = p.kappa * (V[:, i] - P[:, i]) + p.beta * np.tanh(gam * M[:, i])
            dP = drift_P * p.dt + p.sigma_N * sqrt_dt * xi_N[:, i]

            # Mise à jour des états
            P[:, i + 1] = P[:, i] + dP
            # dM_t = -α·M_t·dt + α·dP_t  ⟹  M_{t+dt} = M_t(1 - α·dt) + α·dP
            M[:, i + 1] = M[:, i] * (1.0 - p.alpha * p.dt) + p.alpha * dP
            V[:, i + 1] = V[:, i] + p.g * p.dt + p.sigma_V * sqrt_dt * xi_V[:, i]

        # Dernier γ
        gamma_arr[:, -1] = self._gamma(P, M, V, step=n_steps)

        # Squeeze si une seule trajectoire pour simplifier l'utilisation
        if n_paths == 1:
            return SimulationResult(
                t=t,
                P=P[0],
                M=M[0],
                V=V[0],
                gamma=gamma_arr[0],
            )
        return SimulationResult(t=t, P=P, M=M, V=V, gamma=gamma_arr)

    # ------------------------------------------------------------------
    # Hook γ(t) — à surcharger dans les sous-classes
    # ------------------------------------------------------------------

    def _gamma(
        self,
        P: np.ndarray,
        M: np.ndarray,
        V: np.ndarray,
        step: int,
    ) -> np.ndarray:
        """Retourne γ(t) pour toutes les trajectoires au pas `step`.

        Baseline : γ(t) = γ_0 (constante).

        Parameters
        ----------
        P, M, V : np.ndarray, shape (n_paths, n_steps+1)
            Tableaux d'état courants (seules les colonnes ≤ step sont remplies).
        step : int
            Indice temporel courant.

        Returns
        -------
        np.ndarray, shape (n_paths,)
        """
        n_paths = P.shape[0]
        return np.full(n_paths, self.params.gamma0)


# ---------------------------------------------------------------------------
# Classe enfant : gamma basé sur le mispricing (Option 3 du CLAUDE.md)
# ---------------------------------------------------------------------------

class MispricingGammaModel(ChiarellaModel):
    """Modèle avec γ exponentiel basé sur le mispricing |P_t - V_t|.

    Formulation (Option 3) :

        γ(t) = γ_0 · exp(λ · |P_t - V_t|)

    Interprétation économique : lorsque le prix s'écarte fortement de la valeur
    fondamentale (fort mispricing), γ augmente de façon exponentielle.
    Les trend-followers saturent donc plus vite leur demande, ce qui modélise
    une réticence accrue à « chasser » la tendance quand le marché est déjà
    très éloigné des fondamentaux.

    Parameters
    ----------
    params : ModelParams
        Paramètres hérités.
    lambda_ : float
        Sensibilité exponentielle au mispricing (λ ≥ 0).
        λ = 0 → modèle baseline.
    seed : int | None
        Graine aléatoire.
    """

    def __init__(
        self,
        params: ModelParams | None = None,
        lambda_: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(params=params, seed=seed)
        self.lambda_ = lambda_

    # ------------------------------------------------------------------
    # Hook γ(t) — mispricing exponentiel
    # ------------------------------------------------------------------

    def _gamma(
        self,
        P: np.ndarray,
        M: np.ndarray,
        V: np.ndarray,
        step: int,
    ) -> np.ndarray:
        """γ(t) = γ_0 · exp(λ · |P_t - V_t|).

        Parameters
        ----------
        P, M, V : np.ndarray, shape (n_paths, n_steps+1)
            Tableaux d'état courants (seules les colonnes ≤ step sont remplies).
        step : int
            Indice temporel courant.

        Returns
        -------
        np.ndarray, shape (n_paths,)
        """
        mispricing = np.abs(P[:, step] - V[:, step])  # shape (n_paths,)
        return self.params.gamma0 * np.exp(self.lambda_ * mispricing)


# ---------------------------------------------------------------------------
# Classe enfant : gamma adaptatif linéaire (Option 2 du CLAUDE.md)
# ---------------------------------------------------------------------------

class AdaptiveGammaModel(ChiarellaModel):
    """Modèle avec γ adaptatif linéaire basé sur la volatilité réalisée glissante.

    Formulation (Option 2) :

        γ(t) = γ_0 · (1 + θ · σ_réalisée(t) / σ_baseline)

    où σ_réalisée(t) est l'écart-type des rendements log-prix sur une fenêtre
    glissante de `vol_window` pas de temps, et σ_baseline est une valeur de
    référence (estimée sur les premiers `vol_window` pas ou fournie à la main).

    Interprétation économique : lorsque la volatilité augmente (proxy du VIX),
    γ augmente, ce qui sature plus tôt la demande des trend-followers —
    modélisant une aversion au risque accrue.

    Parameters
    ----------
    params : ModelParams
        Paramètres hérités.
    theta : float
        Sensibilité de γ à la volatilité réalisée (θ ≥ 0).
    vol_window : int
        Taille de la fenêtre glissante (en nombre de pas dt).
        Défaut : 252 (≈ 12 mois avec dt=1/252).
    sigma_baseline : float | None
        Volatilité de référence σ_baseline. Si None, elle est estimée comme
        la moyenne de σ_réalisée sur toute la simulation (calculée lors du
        premier run, puis mise en cache).
    seed : int | None
        Graine aléatoire.
    """

    def __init__(
        self,
        params: ModelParams | None = None,
        theta: float = 1.0,
        vol_window: int = 252,
        sigma_baseline: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(params=params, seed=seed)
        self.theta = theta
        self.vol_window = vol_window
        self._sigma_baseline_fixed = sigma_baseline
        # Cache mis à jour à chaque simulation
        self._sigma_baseline_cache: float | None = sigma_baseline

    # ------------------------------------------------------------------
    # Surcharge de la simulation pour estimer σ_baseline a priori
    # ------------------------------------------------------------------

    def simulate(
        self,
        n_paths: int = 1,
        P0: float = 0.0,
        M0: float = 0.0,
        V0: float = 0.0,
    ) -> SimulationResult:
        """Simulation avec estimation de σ_baseline si non fourni.

        Si `sigma_baseline` n'est pas fixé manuellement, on effectue d'abord
        une passe baseline (γ = γ_0 constant) pour estimer σ_baseline comme
        la moyenne temporelle de la volatilité réalisée glissante, puis on
        relance la simulation adaptative.
        """
        if self._sigma_baseline_fixed is None:
            self._sigma_baseline_cache = self._estimate_sigma_baseline(
                n_paths=n_paths, P0=P0, M0=M0, V0=V0
            )
        return super().simulate(n_paths=n_paths, P0=P0, M0=M0, V0=V0)

    # ------------------------------------------------------------------
    # Hook γ(t) — volatilité réalisée glissante
    # ------------------------------------------------------------------

    def _gamma(
        self,
        P: np.ndarray,
        M: np.ndarray,
        V: np.ndarray,
        step: int,
    ) -> np.ndarray:
        """γ(t) = γ_0 · (1 + θ · σ_réalisée(t) / σ_baseline).

        La volatilité réalisée est calculée sur les `vol_window` derniers
        rendements log-prix disponibles. Avant d'avoir accumulé suffisamment
        d'historique, on utilise tous les rendements disponibles.
        """
        p = self.params
        sigma_base = self._sigma_baseline_cache if self._sigma_baseline_cache is not None else 1.0

        if step < 2:
            # Pas assez d'historique : γ = γ_0
            return np.full(P.shape[0], p.gamma0)

        # Fenêtre de rendements disponibles
        start = max(0, step - self.vol_window)
        # Rendements log-prix : ΔP_{i} = P_{i+1} - P_{i}
        returns = np.diff(P[:, start : step + 1], axis=1)  # shape (n_paths, window)

        # Volatilité réalisée annualisée (annualisation par √(1/dt))
        sigma_real = returns.std(axis=1) / np.sqrt(p.dt)  # shape (n_paths,)

        gamma_t = p.gamma0 * (1.0 + self.theta * sigma_real / sigma_base)
        return gamma_t

    # ------------------------------------------------------------------
    # Estimation de σ_baseline via un run baseline
    # ------------------------------------------------------------------

    def _estimate_sigma_baseline(
        self,
        n_paths: int,
        P0: float,
        M0: float,
        V0: float,
    ) -> float:
        """Estime σ_baseline comme la volatilité réalisée moyenne d'un run baseline.

        On simule le modèle avec γ = γ_0 constant (ChiarellaModel), puis on
        calcule la moyenne sur toutes les trajectoires et tous les instants
        de la volatilité réalisée glissante.

        Returns
        -------
        float
            σ_baseline estimée (> 0).
        """
        baseline_model = ChiarellaModel(params=self.params, seed=None)
        result = baseline_model.simulate(n_paths=n_paths, P0=P0, M0=M0, V0=V0)

        # P shape : (n_paths, n_steps+1) ou (n_steps+1,) si n_paths==1
        P_arr = result.P if result.P.ndim == 2 else result.P[np.newaxis, :]
        returns = np.diff(P_arr, axis=1)                          # rendements
        sigma_real = returns.std(axis=1) / np.sqrt(self.params.dt)  # par trajectoire
        return float(sigma_real.mean())
