# Journal de bord — P15 Finance
 
> **Objectif de ce document** : noter au fil de l'eau les observations importantes, les résultats des simulations, et les raisons des choix de direction pris dans le projet.
 
---
 
## Contexte rapide
 
Modèle agent-based de Chiarella étendu (Majewski, Ciliberti, Bouchaud, 2020). Deux types d'agents coexistent :
- **Fondamentalistes** : ramènent le prix vers la valeur fondamentale (κ)
- **Trend-followers** : suivent la tendance via un signal de momentum (M_t)
 
La dynamique du log-prix est :
 
$$dP_t = \kappa(V_t - P_t)\,dt + \beta\tanh(\gamma(t) M_t)\,dt + \sigma_N\,dW^{(N)}_t$$
 
**Ma tâche dans le groupe** : rendre γ dynamique (dépendant de la volatilité réalisée, proxy VIX) pour modéliser une aversion au risque variable.
 
---
 
## Observations
 
---
 
### [1] Mécanique de la saturation des trend-followers
 
**Paramètre concerné :** γ (gamma, paramètre de saturation)
 
La demande des trend-followers est proportionnelle à `β · tanh(γ · M_t)`. Le **tanh** est une fonction en S qui sature vers ±1 : même si le signal de tendance M_t s'amplifie indéfiniment, la demande des trend-followers plafonne. Cela modélise leurs contraintes réelles (capital limité, aversion aux pertes, risk management).
 
**γ contrôle la vitesse à laquelle on atteint cette saturation** : un γ élevé fait saturer la demande pour des signaux plus faibles.
 
---
 
### [2] Modèle adaptatif linéaire — résultats et interprétation
 
**Formule implémentée (Option 2) :**
$$\gamma(t) = \gamma_0 \cdot \left(1 + \theta \cdot \frac{\sigma_{\text{réal}}(t)}{\sigma_{\text{baseline}}}\right)$$
 
**Paramètres :** γ₀ = 2.0, θ = 1.0, fenêtre vol. réalisée = 12 mois, T = 1 000 mois, seed = 2024.
 
**Résultats (notebook `01_comparaison_gamma`) :**
 
| Métrique | Baseline (γ = 2.0) | Adaptatif (γ variable) | Variation |
|---|---|---|---|
| γ moyen | 2.000 | 3.810 | — |
| γ max | 2.000 | 4.877 | — |
| Vol des rendements (annualisée) | 0.1805 | 0.1808 | ≈ 0 |
| Std(mispricing P−V) | 0.1387 | 0.1497 | **+7.9 %** |
| Max\|mispricing\| | 0.4656 | 0.4966 | **+6.7 %** |
 
**Conclusion : l'hypothèse de départ est infirmée.** Le γ adaptatif *aggrave* légèrement les bulles extrêmes au lieu de les réduire.
 
**Interprétation :** L'intuition initiale ne capturait qu'un seul effet (saturation précoce → moins de demande trend → moins de bulle). Mais un γ élevé rend aussi la courbe tanh **plus raide dans la zone intermédiaire** : pour des signaux M_t modérés, la demande est *plus forte* qu'avec un γ faible, avant de finalement saturer. Or les épisodes de forte volatilité — précisément quand γ monte — correspondent aux phases de formation des bulles, où M_t est encore dans cette zone intermédiaire. Le mécanisme de protection arrive trop tard (une fois M_t très grand, la bulle est déjà formée). L'effet d'amplification sur les signaux modérés domine l'effet de saturation protectrice.
 
**Piste de vérification :** analyser la distribution de γ·M_t au moment des pics de mispricing — si elle tombe systématiquement dans la zone de forte pente du tanh (et non dans la zone saturée), cela confirme ce mécanisme.

---

### [3] Modèle mispricing — résultats et comparaison

**Formule implémentée (Option 3) :**
$$\gamma(t) = \gamma_0 \cdot \exp\!\left(\lambda \cdot |P_t - V_t|\right)$$

**Paramètres :** γ₀ = 2.0, λ = 1.0, T = 1 000 mois, seed = 2024.

#### Comportement de γ(t)

| Métrique de γ | Baseline | Adaptatif (θ=1.0) | Mispricing (λ=1.0) |
|---|---|---|---|
| γ moyen | 2.000 | 3.913 | 2.248 |
| γ écart-type | 0.000 | 0.439 | 0.206 |
| γ min | 2.000 | 2.000 | 2.000 |
| γ max | 2.000 | 5.040 | 3.217 |
| ρ(γ, driver) | — | **0.9996** | **0.9979** |

Le mispricing model produit un γ nettement plus modéré que l'adaptatif (max 3.22 vs 5.04). Les deux modèles dynamiques ont une corrélation quasi-parfaite avec leur driver respectif (ρ ≈ 1.00), ce qui valide l'implémentation.

#### Comparaison complète des trois modèles sur le mispricing P−V

| Métrique du mispricing | Baseline | Adaptatif | Mispricing | Δ adap. | Δ misp. |
|---|---|---|---|---|---|
| Std(P−V) | 0.1387 | 0.1504 | **0.1412** | +8.4 % | +1.8 % |
| Max\|P−V\| | 0.4656 | 0.4984 | **0.4753** | +7.1 % | +2.1 % |
| Skewness | −0.518 | −0.463 | −0.526 | — | — |
| Kurtosis excéd. | 0.471 | 0.414 | **0.509** | — | — |
| q1 % (P−V) | −0.361 | −0.387 | −0.368 | — | — |
| q99 % (P−V) | +0.302 | +0.341 | +0.308 | — | — |

*(Δ = variation relative par rapport au baseline)*

#### Interprétation

**Le modèle mispricing est plus stable que l'adaptatif, mais n'atteint pas non plus l'objectif de réduction des bulles.**

- Son γ(t) reste proche de γ₀ (mean = 2.25 vs 3.91 pour l'adaptatif), car le rappel fondamental κ maintient |P−V| relativement faible en moyenne. Le levier exponentiel ne s'active fortement que lors des épisodes de bulle prononcée — précisément quand on en a besoin, contrairement à l'adaptatif qui monte dès que la vol augmente (même sans bulle structurée).

- La std du mispricing augmente tout de même de +1.8 % par rapport au baseline. Le même mécanisme que pour l'adaptatif est à l'œuvre : un γ plus élevé dans la zone intermédiaire du tanh amplifie temporairement la demande des trend-followers. Mais l'effet est bien plus faible, ce qui confirme que la rétroaction endogène (γ monte *quand la bulle est déjà grande*) est plus pertinente que la rétroaction via la volatilité réalisée.

- La kurtosis excédentaire du mispricing est légèrement supérieure pour le modèle mispricing (0.509 vs 0.471 baseline), signe que les queues sont un peu plus épaisses, mais de façon très marginale.

#### Comparaison des volatilités des rendements

| Modèle | Vol annualisée | Skewness | Kurtosis excéd. |
|---|---|---|---|
| Baseline | 0.1805 | 0.044 | 0.074 |
| Adaptatif | 0.1808 | 0.049 | 0.066 |
| Mispricing | 0.1805 | 0.045 | 0.072 |

Les trois modèles produisent des rendements quasi-identiques en termes de vol (≈ 18 % annualisé) et de distribution — les différences ne sont pas statistiquement significatives sur 1 000 mois. Cela suggère que γ(t) influence principalement la *structure spatiale* du mispricing (amplitude des bulles) plutôt que la volatilité totale des rendements.

#### Bilan des deux modèles dynamiques

| | Adaptatif (σ_réal) | Mispricing (|P−V|) |
|---|---|---|
| Driver | Exogène (vol passée) | Endogène (écart fondamental) |
| Amplitude de γ | Forte (max 5.0) | Modérée (max 3.2) |
| Effet sur bulles | Aggrave (+7–8 %) | Aggrave légèrement (+2 %) |
| Interprétation éco. | Proxy VIX / aversion risque | Mécanisme correctif endogène |
| Pertinence économique | Bonne intuition, effet contre-intuitif | Plus ciblé, moins d'effets de bord |

**Piste suivante :** tenter une formulation *décroissante* γ(t) = γ₀ / (1 + λ·|P−V|) qui freinerait explicitement la demande des trend-followers lors des bulles. → Implémenté dans la section [4].

---

### [4] Modèle mispricing décroissant — résultats et interprétation

**Formule implémentée (Option 4) :**
$$\gamma(t) = \frac{\gamma_0}{1 + \lambda \cdot |P_t - V_t|}$$

**Paramètres :** γ₀ = 2.0, λ = 1.0, T = 1 000 mois, seed = 2024.

#### Comportement de γ(t)

| Métrique de γ | Baseline | Adaptatif | Mispricing↑ | **Décroissant** |
|---|---|---|---|---|
| γ moyen | 2.000 | 3.865 | 2.248 | **1.812** |
| γ écart-type | 0.000 | 0.428 | 0.206 | **0.128** |
| γ min | 2.000 | 2.000 | 2.000 | **1.370** |
| γ max | 2.000 | 4.964 | 3.217 | **2.000** |
| ρ(γ, driver) | — | +0.9996 | +0.9979 | **−0.9953** |

La corrélation négative quasi-parfaite (ρ = −0.9953) avec |P−V| confirme le mécanisme : γ descend bien quand la bulle monte.

#### Comparaison des quatre modèles sur le mispricing P−V

| Modèle | Std(P−V) | Δstd | Max\|P−V\| | Δmax | Skewness | Kurtosis excéd. |
|---|---|---|---|---|---|---|
| Baseline | 0.1387 | — | 0.4656 | — | −0.518 | 0.471 |
| Adaptatif | 0.1501 | **+8.2 %** | 0.4976 | **+6.9 %** | −0.465 | 0.416 |
| Mispricing↑ | 0.1412 | **+1.8 %** | 0.4753 | **+2.1 %** | −0.526 | 0.509 |
| **Décroissant** | **0.1372** | **−1.1 %** | **0.4597** | **−1.3 %** | −0.515 | 0.454 |

**Résultat clé : le modèle décroissant est le seul à réduire les bulles.** Pour la première fois, std(P−V) passe en dessous du baseline (−1.1 %) et le max absolu diminue (−1.3 %).

#### Interprétation économique

La logique est simple et directe : quand le mispricing |P−V| augmente, γ(t) *diminue*, ce qui réduit la pente effective du tanh autour de zéro. La demande des trend-followers β·tanh(γ↓·M_t) est donc **moins forte** pendant les épisodes de bulle — même pour des signaux M_t modérés. Le mécanisme de stabilisation opère exactement là où on en a besoin, sans les effets de bord des formulations croissantes.

Contrairement aux modèles 2 et 3 :
- L'adaptatif monte γ quand la vol monte, ce qui amplifie la demande sur les signaux intermédiaires précisément pendant la *formation* des bulles.
- Le mispricing↑ monte γ quand la bulle est déjà grande, trop tard pour éviter l'amplification initiale.
- Le décroissant réduit γ dès que |P−V| grandit, freinant les trend-followers à chaque étape de la bulle.

#### Comparaison des volatilités des rendements

| Modèle | Vol annualisée | Skewness | Kurtosis excéd. |
|---|---|---|---|
| Baseline | 0.1805 | 0.044 | 0.074 |
| Adaptatif | 0.1808 | 0.048 | 0.066 |
| Mispricing↑ | 0.1805 | 0.045 | 0.072 |
| **Décroissant** | **0.1806** | 0.043 | 0.075 |

La volatilité des rendements reste quasi-identique dans les quatre cas (≈ 18 % annualisé). Le modèle décroissant ne détériore pas les propriétés statistiques des rendements tout en réduisant le mispricing — ce qui est une propriété souhaitable.

#### Bilan général des 4 modèles

| Modèle | Driver | Direction γ | Effet sur bulles | Pertinence |
|---|---|---|---|---|
| Baseline | — | constant | référence | référence |
| Adaptatif | σ_réal (exogène) | ↑ | aggrave +8 % | intuition VIX, effet contre-intuitif |
| Mispricing↑ | \|P−V\| (endogène) | ↑ | aggrave +2 % | ciblé mais effet de bord tanh |
| **Décroissant** | \|P−V\| (endogène) | ↓ | **réduit −1 %** | mécanisme direct et cohérent |

**Piste pour la suite :** calibrer λ plus finement (analyse de sensibilité sur λ ∈ [0.1, 5.0]) et tester sur des simulations Monte Carlo à grande échelle (N = 1 000 trajectoires) pour vérifier la robustesse statistique de la réduction du mispricing.

---

### [5] Analyse de sensibilité sur κ

**Notebook :** `notebooks/02_sensibilite_kappa.ipynb`

**Objectif :** Quantifier l'impact de la force de rappel fondamental κ sur les bulles et la volatilité, toutes choses égales par ailleurs (γ constant, baseline).

**Plage testée :** κ ∈ {0.01, 0.02, 0.05, 0.08, 0.15, 0.30, 0.50}

#### Résultats numériques

| κ | Std(P−V) | Δ Std | Max\|P−V\| | Δ Max | Vol annualisée | Δ Vol |
|---|---|---|---|---|---|---|
| 0.01 | 0.317 | +128.5 % | 0.907 | +94.9 % | 0.177 | −1.9 % |
| 0.02 | 0.256 | +84.3 % | 0.709 | +52.3 % | 0.178 | −1.6 % |
| 0.05 | 0.174 | +25.4 % | 0.547 | +17.4 % | 0.179 | −0.8 % |
| **0.08** | **0.139** | **—** | **0.466** | **—** | **0.181** | **—** |
| 0.15 | 0.103 | −26.0 % | 0.362 | −22.3 % | 0.184 | +2.0 % |
| 0.30 | 0.076 | −45.1 % | 0.266 | −42.8 % | 0.193 | +6.8 % |
| 0.50 | 0.063 | −54.5 % | 0.239 | −48.6 % | 0.206 | +14.2 % |

*(Δ = variation relative par rapport au baseline κ = 0.08)*

#### Interprétation

κ est le levier le plus puissant sur l'amplitude des bulles dans le modèle. La relation Std(P−V) vs κ est **monotone décroissante et convexe** : les gains marginaux sont très importants pour les petits κ (zone κ < 0.10) et se tassent progressivement. Multiplier κ par 6 (de 0.08 à 0.50) réduit les bulles de 55 %, mais au prix d'une augmentation de 14 % de la volatilité des rendements.

Ce **trade-off** s'explique par la mécanique du rappel fondamental : un κ élevé génère des oscillations de correction fréquentes et énergiques vers V, qui se traduisent par plus de bruit à court terme dans les rendements même si les bulles sont contenues.

Fait notable : skewness et kurtosis des rendements sont **quasi-insensibles à κ** sur toute la plage (le test de Jarque-Bera ne rejette pas la normalité dans aucun cas). κ modifie l'amplitude des bulles mais pas la forme de la distribution des rendements.

#### Bilan

Le baseline κ = 0.08 constitue un équilibre raisonnable. Ces résultats motivent l'extension multi-fondamentalistes (section [6]) : si des groupes ont des κ_i hétérogènes avec κ_eff > 0.08, on peut espérer réduire les bulles tout en limitant l'augmentation de volatilité.

---

### [6] Extension multi-fondamentalistes — κ_i et V_i distincts

**Notebook :** `notebooks/03_multi_kappa.ipynb`

**Objectif :** Modéliser l'hétérogénéité des investisseurs fondamentalistes en leur attribuant des forces de rappel κ_i différentes et des processus de valeur fondamentale V_i(t) indépendants.

**Équation modifiée :**
$$dP_t = \left[\sum_i \kappa_i (V_i(t) - P_t)\right] dt + \beta\tanh(\gamma M_t)\,dt + \sigma_N\,dW_t^{(N)}$$
$$dV_i(t) = g_i\,dt + \sigma_{V_i}\,dW_t^{(i)}$$

**Configurations testées :**
- **Baseline** : κ unique = 0.08, un seul processus V
- **Config A** (2 groupes) : fondamentalistes réactifs (κ₁=0.15, σ_V₁=0.02) + investisseurs patients (κ₂=0.02, σ_V₂=0.01) → κ_eff = 0.17
- **Config B** (3 groupes) : optimistes (κ=0.05, g=+0.001) + pessimistes (κ=0.05, g=−0.001) + neutres (κ=0.10, σ_V=0.01) → κ_eff = 0.20

#### Résultats numériques

| Modèle | Std(P−V̄) | Δ Std | Max\|P−V̄\| | Δ Max | Vol annualisée | Δ Vol |
|---|---|---|---|---|---|---|
| Baseline (κ=0.08) | 0.1387 | — | 0.4656 | — | 0.1805 | — |
| Config A (2 groupes) | 0.0962 | **−30.6 %** | 0.3419 | **−26.6 %** | 0.1849 | +2.4 % |
| Config B (3 groupes) | 0.0896 | **−35.4 %** | 0.3159 | **−32.2 %** | 0.1856 | +2.8 % |

#### Interprétation

Les réductions de mispricing sont **substantielles et robustes** : −30 à −35 % sur la std, −27 à −32 % sur les pics maximaux. Ces gains dépassent largement ceux des modèles γ dynamiques.

**L'hétérogénéité apporte un bénéfice au-delà du seul κ_eff.** En comparant avec les résultats du notebook 02 :
- Config A (κ_eff = 0.17) réduit Std(P−V) de 30.6 %, contre seulement 26 % pour un κ uniforme de 0.15.
- Config B (κ_eff = 0.20) réduit Std(P−V) de 35.4 %, supérieur à une interpolation linéaire des résultats homogènes.

Ce surplus de stabilisation provient de la **diversification des processus V_i** : le prix n'est plus attiré vers un seul fondamental volatil mais vers une moyenne consensus V̄ = Σκ_i·V_i / Σκ_i qui absorbe mieux les chocs idiosyncratiques de chaque groupe. La diversification des croyances fondamentales réduit l'amplitude des signaux qui attirent le prix vers des niveaux extrêmes.

**Coût en volatilité très limité :** +2.4 % et +2.8 % seulement, contre +6.8 % et +14 % pour des κ homogènes équivalents. L'hétérogénéité permet donc de "profiter" d'un κ_eff plus élevé sans en payer pleinement le coût en volatilité.

#### Divergence des valeurs fondamentales

La dispersion entre V_i croît avec le temps dans les deux configurations (browniens indépendants). En Config A, le désaccord V₁ − V₂ dérive progressivement. En Config B, le groupe neutre (σ_V=0.01) sert d'ancre qui limite l'excursion maximale du prix malgré le désaccord optimistes/pessimistes. Ces dynamiques sont visibles dans les trajectoires du notebook et représentent un mécanisme réaliste (les investisseurs ont des estimations de valeur intrinsèque différentes et qui divergent dans le temps).

#### Bilan

L'extension multi-fondamentalistes est **la contribution la plus efficace en termes de réduction des bulles** parmi toutes les extensions testées. Elle est également économiquement motivée : les marchés financiers réels sont peuplés d'agents avec des horizons, des croyances et des vitesses de réaction très différents.

**Perspective :** Combiner l'hétérogénéité multi-κ avec le modèle γ décroissant (section [4]) pourrait produire une réduction des bulles encore plus robuste en activant simultanément deux canaux de stabilisation indépendants.

---