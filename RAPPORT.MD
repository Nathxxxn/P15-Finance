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