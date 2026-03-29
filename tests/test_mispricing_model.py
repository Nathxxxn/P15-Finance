"""
Tests pour MispricingGammaModel (γ(t) = γ_0·exp(λ·|P_t - V_t|)).

Couverture :
    - Limite λ=0 → identique au modèle baseline (γ constant)
    - γ(t) ≥ γ_0 pour tout t (exp positif)
    - γ(t) croît avec le mispricing (propriété monotone)
    - Shapes correctes en mode multi-trajectoires
    - Reproductibilité (seed fixée)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model import ChiarellaModel, MispricingGammaModel, ModelParams, SimulationResult


# ---------------------------------------------------------------------------
# Paramètres légers pour les tests (simulation courte, rapide)
# ---------------------------------------------------------------------------

PARAMS_TEST = ModelParams(
    kappa=0.08,
    beta=0.10,
    alpha=1 / 7,
    gamma0=2.0,
    sigma_N=0.05,
    sigma_V=0.02,
    g=0.0,
    dt=1.0,
    T=50.0,   # 50 pas — rapide
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_mispricing(lambda_: float, seed: int = 42) -> SimulationResult:
    model = MispricingGammaModel(params=PARAMS_TEST, lambda_=lambda_, seed=seed)
    return model.simulate(n_paths=1)


def _simulate_baseline(seed: int = 42) -> SimulationResult:
    model = ChiarellaModel(params=PARAMS_TEST, seed=seed)
    return model.simulate(n_paths=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMispricingGammaModel:

    def test_lambda_zero_equals_gamma0(self):
        """Avec λ=0, γ(t) doit être constant et égal à γ_0."""
        res = _simulate_mispricing(lambda_=0.0)
        assert np.allclose(res.gamma, PARAMS_TEST.gamma0), (
            f"λ=0 devrait donner γ(t)=γ_0={PARAMS_TEST.gamma0} partout, "
            f"obtenu mean={res.gamma.mean():.4f}"
        )

    def test_gamma_at_least_gamma0(self):
        """γ(t) ≥ γ_0 pour tout t, car exp(λ·|·|) ≥ 1 avec λ ≥ 0."""
        for lambda_ in [0.5, 1.0, 2.0]:
            res = _simulate_mispricing(lambda_=lambda_)
            assert np.all(res.gamma >= PARAMS_TEST.gamma0 - 1e-10), (
                f"λ={lambda_} : γ(t) < γ_0 détecté (min={res.gamma.min():.4f})"
            )

    def test_gamma_increases_with_lambda(self):
        """Un λ plus grand doit produire un γ moyen plus élevé (même seed)."""
        res_low  = _simulate_mispricing(lambda_=0.5, seed=0)
        res_high = _simulate_mispricing(lambda_=2.0, seed=0)
        assert res_high.gamma.mean() > res_low.gamma.mean(), (
            "γ_mean devrait croître avec λ"
        )

    def test_simulation_result_shapes(self):
        """Les tableaux retournés doivent avoir la bonne shape pour n_paths=1."""
        res = _simulate_mispricing(lambda_=1.0)
        n_steps = int(PARAMS_TEST.T / PARAMS_TEST.dt)
        expected_len = n_steps + 1

        assert res.t.shape   == (expected_len,), f"t.shape={res.t.shape}"
        assert res.P.shape   == (expected_len,), f"P.shape={res.P.shape}"
        assert res.M.shape   == (expected_len,), f"M.shape={res.M.shape}"
        assert res.V.shape   == (expected_len,), f"V.shape={res.V.shape}"
        assert res.gamma.shape == (expected_len,), f"gamma.shape={res.gamma.shape}"

    def test_simulation_result_shapes_multi_paths(self):
        """Shape correcte pour n_paths > 1."""
        n_paths = 5
        model = MispricingGammaModel(params=PARAMS_TEST, lambda_=1.0, seed=7)
        res = model.simulate(n_paths=n_paths)
        n_steps = int(PARAMS_TEST.T / PARAMS_TEST.dt)
        expected = (n_paths, n_steps + 1)

        assert res.P.shape     == expected, f"P.shape={res.P.shape}"
        assert res.gamma.shape == expected, f"gamma.shape={res.gamma.shape}"

    def test_reproducibility(self):
        """Deux runs avec la même seed donnent des trajectoires identiques."""
        res_a = _simulate_mispricing(lambda_=1.0, seed=123)
        res_b = _simulate_mispricing(lambda_=1.0, seed=123)
        np.testing.assert_array_equal(res_a.P, res_b.P)
        np.testing.assert_array_equal(res_a.gamma, res_b.gamma)

    def test_gamma_formula_spot_check(self):
        """Vérification ponctuelle de la formule γ = γ_0·exp(λ·|P-V|) au pas 1.

        On calcule manuellement le γ attendu à partir des prix et valeurs
        fondamentales issus de la trajectoire, puis on le compare à gamma[1].
        """
        lambda_ = 1.5
        res = _simulate_mispricing(lambda_=lambda_, seed=0)
        # Au pas 1, γ[1] est calculé à partir de P[1] et V[1]
        expected = PARAMS_TEST.gamma0 * np.exp(lambda_ * abs(res.P[1] - res.V[1]))
        assert abs(res.gamma[1] - expected) < 1e-10, (
            f"Formule incorrecte au pas 1 : attendu {expected:.6f}, "
            f"obtenu {res.gamma[1]:.6f}"
        )

    def test_no_nan_in_output(self):
        """Aucune valeur NaN dans les trajectoires simulées."""
        res = _simulate_mispricing(lambda_=1.0, seed=99)
        for name, arr in [("P", res.P), ("M", res.M), ("V", res.V), ("gamma", res.gamma)]:
            assert not np.any(np.isnan(arr)), f"NaN détecté dans {name}"

    def test_lambda_zero_same_dynamics_as_baseline(self):
        """Avec λ=0 et même seed, les trajectoires doivent être identiques au baseline."""
        seed = 55
        res_misp = _simulate_mispricing(lambda_=0.0, seed=seed)
        res_base = _simulate_baseline(seed=seed)
        np.testing.assert_array_almost_equal(res_misp.P, res_base.P, decimal=10)
        np.testing.assert_array_almost_equal(res_misp.V, res_base.V, decimal=10)
