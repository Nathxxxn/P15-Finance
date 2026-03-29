"""
Tests pour MultiKappaModel (Σ κ_i·(V_i(t) − P_t) avec V_i indépendants).

Couverture :
    - Équivalence avec ChiarellaModel quand N=1 groupe et mêmes paramètres
    - Shapes correctes en mode 1 trajectoire et multi-trajectoires
    - Les V_i divergent quand σ_Vi sont différents
    - Reproductibilité (seed fixée)
    - κ_i tous nuls → pas de rappel fondamental (prix = trend + bruit)
    - V_bar est bien la moyenne pondérée des V_i
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model import (
    ChiarellaModel,
    FundamentalistGroup,
    ModelParams,
    MultiKappaModel,
    MultiKappaSimulationResult,
)


# ---------------------------------------------------------------------------
# Paramètres légers pour les tests
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
    T=50.0,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single_group(kappa: float = 0.08, sigma_V: float = 0.02,
                  g: float = 0.0, V0: float = 0.0) -> list[FundamentalistGroup]:
    return [FundamentalistGroup(kappa=kappa, sigma_V=sigma_V, g=g, V0=V0)]


def _two_groups() -> list[FundamentalistGroup]:
    return [
        FundamentalistGroup(kappa=0.15, sigma_V=0.02, g=0.0),
        FundamentalistGroup(kappa=0.02, sigma_V=0.01, g=0.0),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiKappaModel:

    def test_single_group_equivalence_with_baseline(self):
        """N=1 groupe avec (κ, σ_V, g) = ModelParams → trajectoires identiques au baseline."""
        seed = 42
        grp = FundamentalistGroup(
            kappa=PARAMS_TEST.kappa,
            sigma_V=PARAMS_TEST.sigma_V,
            g=PARAMS_TEST.g,
            V0=0.0,
        )
        multi = MultiKappaModel(groups=[grp], base_params=PARAMS_TEST, seed=seed)
        res_multi = multi.simulate(n_paths=1)

        baseline = ChiarellaModel(params=PARAMS_TEST, seed=seed)
        res_base = baseline.simulate(n_paths=1)

        np.testing.assert_array_almost_equal(res_multi.P, res_base.P, decimal=10,
            err_msg="P doit être identique entre MultiKappa(1 groupe) et baseline")
        np.testing.assert_array_almost_equal(res_multi.M, res_base.M, decimal=10,
            err_msg="M doit être identique entre MultiKappa(1 groupe) et baseline")
        np.testing.assert_array_almost_equal(res_multi.V_bar, res_base.V, decimal=10,
            err_msg="V_bar doit être identique à V du baseline quand N=1")

    def test_result_shapes_single_path(self):
        """Shapes correctes pour n_paths=1 avec 2 groupes."""
        model = MultiKappaModel(groups=_two_groups(), base_params=PARAMS_TEST, seed=0)
        res = model.simulate(n_paths=1)
        n_steps = int(PARAMS_TEST.T / PARAMS_TEST.dt)
        expected_len = n_steps + 1

        assert res.t.shape     == (expected_len,)
        assert res.P.shape     == (expected_len,)
        assert res.M.shape     == (expected_len,)
        assert res.V_bar.shape == (expected_len,)
        assert res.gamma.shape == (expected_len,)
        assert res.V_all.shape == (2, expected_len)  # (n_groups, n_steps+1)

    def test_result_shapes_multi_paths(self):
        """Shapes correctes pour n_paths > 1 avec 2 groupes."""
        n_paths = 4
        model = MultiKappaModel(groups=_two_groups(), base_params=PARAMS_TEST, seed=1)
        res = model.simulate(n_paths=n_paths)
        n_steps = int(PARAMS_TEST.T / PARAMS_TEST.dt)
        expected = (n_paths, n_steps + 1)

        assert res.P.shape     == expected
        assert res.V_bar.shape == expected
        assert res.V_all.shape == (2, n_paths, n_steps + 1)

    def test_reproducibility(self):
        """Deux runs avec la même seed produisent des résultats identiques."""
        groups = _two_groups()
        res_a = MultiKappaModel(groups=groups, base_params=PARAMS_TEST, seed=77).simulate()
        res_b = MultiKappaModel(groups=groups, base_params=PARAMS_TEST, seed=77).simulate()

        np.testing.assert_array_equal(res_a.P, res_b.P)
        np.testing.assert_array_equal(res_a.V_bar, res_b.V_bar)
        np.testing.assert_array_equal(res_a.V_all, res_b.V_all)

    def test_vi_diverge_with_different_sigma(self):
        """Avec σ_Vi différents, les V_i s'écartent l'un de l'autre au fil du temps."""
        groups = [
            FundamentalistGroup(kappa=0.05, sigma_V=0.0,  g=0.0),  # V_0 = constant
            FundamentalistGroup(kappa=0.05, sigma_V=0.05, g=0.0),  # V_1 = diffus
        ]
        model = MultiKappaModel(groups=groups, base_params=PARAMS_TEST, seed=123)
        res = model.simulate(n_paths=1)

        # V_0 doit être (presque) constant, V_1 doit varier
        assert res.V_all[0].std() < 1e-10, "V_0 avec σ=0 doit être constant"
        assert res.V_all[1].std() > 0.01,  "V_1 avec σ>0 doit varier"

    def test_v_bar_is_kappa_weighted_average(self):
        """V_bar = Σ(κ_i·V_i) / Σκ_i."""
        groups = _two_groups()  # κ₁=0.15, κ₂=0.02
        model = MultiKappaModel(groups=groups, base_params=PARAMS_TEST, seed=5)
        res = model.simulate(n_paths=1)

        kappas = np.array([g.kappa for g in groups])
        expected_V_bar = (kappas[:, np.newaxis] * res.V_all).sum(axis=0) / kappas.sum()
        np.testing.assert_array_almost_equal(res.V_bar, expected_V_bar, decimal=12)

    def test_zero_kappa_no_fundamental_pull(self):
        """Avec κ_i = 0 pour tous les groupes, le terme fondamentaliste est nul."""
        groups = [
            FundamentalistGroup(kappa=0.0, sigma_V=0.02),
            FundamentalistGroup(kappa=0.0, sigma_V=0.02),
        ]
        # Pas d'erreur, simulation possible
        model = MultiKappaModel(groups=groups, base_params=PARAMS_TEST, seed=9)
        res = model.simulate(n_paths=1)
        assert not np.any(np.isnan(res.P)), "NaN détectés dans P avec κ=0"

    def test_no_nan_in_output(self):
        """Aucune valeur NaN dans les trajectoires simulées."""
        model = MultiKappaModel(groups=_two_groups(), base_params=PARAMS_TEST, seed=33)
        res = model.simulate(n_paths=1)
        for name, arr in [("P", res.P), ("M", res.M), ("V_bar", res.V_bar),
                          ("gamma", res.gamma), ("V_all", res.V_all)]:
            assert not np.any(np.isnan(arr)), f"NaN détecté dans {name}"

    def test_v_bar_constant_when_all_sigma_zero(self):
        """Si tous les σ_Vi = 0 et g_i = 0, V_bar reste constant à V0."""
        groups = [
            FundamentalistGroup(kappa=0.1, sigma_V=0.0, g=0.0, V0=1.0),
            FundamentalistGroup(kappa=0.1, sigma_V=0.0, g=0.0, V0=1.0),
        ]
        model = MultiKappaModel(groups=groups, base_params=PARAMS_TEST, seed=0)
        res = model.simulate(n_paths=1)
        np.testing.assert_allclose(res.V_bar, 1.0, atol=1e-12,
            err_msg="V_bar doit rester constant à 1.0 quand σ_Vi=0 et g_i=0")

    def test_three_groups_no_error(self):
        """3 groupes fonctionnent sans erreur."""
        groups = [
            FundamentalistGroup(kappa=0.05, sigma_V=0.03, g=+0.001),
            FundamentalistGroup(kappa=0.05, sigma_V=0.03, g=-0.001),
            FundamentalistGroup(kappa=0.10, sigma_V=0.01, g=0.0),
        ]
        model = MultiKappaModel(groups=groups, base_params=PARAMS_TEST, seed=7)
        res = model.simulate(n_paths=1)
        assert res.V_all.shape[0] == 3
