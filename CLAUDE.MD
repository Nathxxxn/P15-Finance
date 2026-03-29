# Projet P15 - Modèles financiers d'agents multi-échelle

## Contexte du Projet
Ce projet implémente et étend le modèle de Chiarella étendu (Majewski, Ciliberti, Bouchaud, 2020). C'est un modèle Agent-Based (HABM) simulant la coexistence des effets de tendance (momentum) et de valeur (mean-reversion) sur les marchés financiers. 

## Focus Actuel (Ma tâche dans le groupe)
Rendre γ dynamique pour modéliser l'aversion au risque variable des investisseurs.

**État d'avancement :**
- ✅ Modèle 1 — Baseline (γ constant) : `ChiarellaModel`
- ✅ Modèle 2 — Adaptatif linéaire (γ ∝ σ_réal) : `AdaptiveGammaModel`
- ✅ Modèle 3 — Mispricing exponentiel (γ ∝ exp(λ|P−V|)) : `MispricingGammaModel`
- ✅ **Modèle 4 — Mispricing décroissant (γ ∝ 1/(1+λ|P−V|)) : `DecreasingMispricingGammaModel`**

**Résultat modèle 4 :** seul modèle à réduire les bulles (std mispricing −1.1 %, max|P−V| −1.3 %) sans dégrader la volatilité des rendements.

## Prochaine Étape (à faire lors de la prochaine session)

**Analyse de sensibilité sur λ pour le modèle décroissant (`DecreasingMispricingGammaModel`) :**

1. **Sweep paramétrique** : simuler λ ∈ {0.1, 0.25, 0.5, 1.0, 2.0, 5.0} sur N=1 000 trajectoires Monte Carlo (n_paths=1000, même seed) et tracer std(P−V) et max|P−V| en fonction de λ — trouver le λ optimal.
2. **Validation statistique** : utiliser un test de Wilcoxon ou bootstrap pour vérifier que la réduction du mispricing est significative (et pas juste du bruit sur 1 trajectoire).
3. **Nouveau notebook** `notebooks/02_sensibilite_lambda.ipynb` dédié à cette analyse.
4. Si les résultats sont robustes, mettre à jour le RAPPORT.MD avec les conclusions et envisager de présenter le modèle 4 comme contribution principale.

## Équations Mathématiques (Modèle de base)
Le système dynamique stochastique en temps continu est défini par :
\[ dP_t = \kappa(V_t - P_t)dt + \beta \tanh(\gamma(t) M_t)dt + \sigma_N dW^{(N)}_t \]
\[ dM_t = -\alpha M_t dt + \alpha dP_t \]
\[ dV_t = g dt + \sigma_V dW^{(V)}_t \]

Où :
- \(P_t\) : Log-prix
- \(V_t\) : Log-valeur fondamentale
- \(M_t\) : Signal de tendance (moyenne mobile exponentielle des rendements)
- \(\kappa\) : Force de rappel des fondamentalistes
- \(\beta\) : Poids de la demande des trend-followers
- \(\alpha\) : Inverse de l'horizon temporel des trend-followers
- \(\gamma(t)\) : Paramètre de saturation (notre focus)

## Formulations pour \(\gamma(t)\)
1. **Modèle constant (Baseline)** : \(\gamma(t) = \gamma_0\)
2. **Modèle adaptatif linéaire** : \(\gamma(t) = \gamma_0 \cdot \left(1 + \theta \cdot \frac{\sigma_{realisee}(t)}{\sigma_{baseline}}\right)\)
3. **Modèle mispricing exponentiel** : \(\gamma(t) = \gamma_0 \cdot \exp(\lambda \cdot |P_t - V_t|)\)
4. **Modèle mispricing décroissant** : \(\gamma(t) = \frac{\gamma_0}{1 + \lambda \cdot |P_t - V_t|}\)

Le modèle 4 est l'extension prioritaire. Avec γ décroissant, un fort mispricing réduit directement la saturation des trend-followers (β·tanh(γ↓·M_t) → demande réduite), ce qui doit freiner la formation des bulles.

## Architecture du Code Souhaitée
- `src/model.py` : Définition des classes de simulation (ex: `ChiarellaModel`, `AdaptiveGammaModel`).
- `src/simulation.py` : Fonctions d'intégration numérique (Euler-Maruyama).
- `src/analysis.py` : Fonctions de calcul de la volatilité glissante, extraction du signal, statistiques de distribution.
- `notebooks/` : Notebooks Jupyter pour les visualisations (comparaison constant vs adaptatif, diagrammes de phase).

## Directives de Codage pour Claude
- **Stack technique** : Python 3.10+, `numpy` (optimisation vectorielle indispensable pour les simulations Monte Carlo), `pandas` (manipulation des séries temporelles), `scipy`, `matplotlib`/`seaborn`.
- **Typage** : Utiliser les type hints stricts (`-> np.ndarray`, `float`, etc.).
- **Modulable** : Le code doit être écrit de manière à pouvoir intégrer facilement plusieurs échelles de temps \(\alpha_i\) plus tard (travail des autres membres du groupe).
- **Commentaires** : Documenter les équations implémentées dans les docstrings.
- Ne pas utiliser de boucles `for` pour les calculs de trajectoires simples si une vectorisation `numpy` est possible (bien que l'intégration Euler-Maruyama séquentielle soit souvent inévitable à cause de \(\gamma(t)\) qui dépend du passé).

## Commandes Fréquentes
- Lancer les tests : `pytest tests/`
- Lancer le script de simulation de base : `python src/simulation.py`

## Directives Git et Gitlab (CRUCIAL)
Je travaille en équipe sur un dépôt Gitlab. Tu dois **absolument** respecter ces règles pour ne pas perturber le travail des autres :

### Gestion des branches
- **NE JAMAIS** push directement sur la branche `main` ou `master`.
- **NE JAMAIS** créer ni pousser de branches commençant par `claude/...` ou avec des noms aléatoires (ex: `claude/frosty-bardeen`, `claude/fervent-napier`). Ces branches polluent le dépôt partagé de l'équipe.
- **TOUJOURS** travailler directement sur une branche au format `nathan/feat/...` ou `nathan/fix/...` dès le début. Ne jamais travailler sur une branche `claude/...` puis la renommer : créer directement la bonne branche.
- Format obligatoire : `nathan/feat/nom-de-la-fonctionnalite` ou `nathan/fix/nom-du-bug`.
- Exemples valides : `nathan/feat/gamma-adaptatif`, `nathan/fix/bug-euler-maruyama`.
- Demande-moi toujours confirmation avant de créer une nouvelle branche.

### Commits
- Fais des petits commits logiques et atomiques (pas de commit géant "mise à jour générale").
- Utilise la convention **Conventional Commits** pour les messages :
  - `feat: [description]` (pour l'ajout d'une fonctionnalité, ex: `feat: ajout de la classe AdaptiveGammaModel`)
  - `fix: [description]` (pour la correction d'un bug)
  - `refactor: [description]` (pour la réorganisation du code sans changement de comportement)
  - `docs: [description]` (pour les commentaires ou docstrings)
- Les messages de commit doivent être en **français**, clairs et concis (moins de 50 caractères pour le titre).

### Push et Synchronisation
- Si je te demande de push, assure-toi d'avoir fait un `git pull --rebase` avant pour éviter les conflits avec le code de mes coéquipiers.

### Fichiers ignorés
- Ne commite jamais de fichiers de données `.csv`, `.pkl` ou dossiers d'environnement virtuel (ex: `venv/`, `.env`).
