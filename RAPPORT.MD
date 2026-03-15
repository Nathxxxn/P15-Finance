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