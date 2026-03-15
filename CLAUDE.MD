# Projet P15 - Modèles financiers d'agents multi-échelle

## Contexte du Projet
Ce projet implémente et étend le modèle de Chiarella étendu (Majewski, Ciliberti, Bouchaud, 2020). C'est un modèle Agent-Based (HABM) simulant la coexistence des effets de tendance (momentum) et de valeur (mean-reversion) sur les marchés financiers. 

## Focus Actuel (Ma tâche dans le groupe)
L'objectif principal actuel est de modifier le paramètre de saturation des trend-followers, \(\gamma\), pour le rendre **dynamique et dépendant de l'état du marché** (proxy du VIX / volatilité réalisée), afin de modéliser l'aversion au risque variable des investisseurs.

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

## Formulations pour \(\gamma(t)\) à implémenter
Claude, lorsque tu devras coder l'extension, base-toi sur la volatilité réalisée \(\sigma_{realisee}(t)\) sur une fenêtre glissante (ex: 12 mois) :
1. **Modèle constant (Baseline)** : \(\gamma(t) = \gamma_0\)
2. **Modèle adaptatif linéaire** : \(\gamma(t) = \gamma_0 \cdot \left(1 + \theta \cdot \frac{\sigma_{realisee}(t)}{\sigma_{baseline}}\right)\)
3. **Modèle basé sur le mispricing** : \(\gamma(t) = \gamma_0 \cdot \exp(\lambda \cdot |P_t - V_t|)\)

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
- **NE JAMAIS** créer de branches commençant par `claude/...` ou des noms aléatoires.
- Si tu dois créer une branche pour ma fonctionnalité, utilise le format : `nathan/feat/gamma-adaptatif` ou `nathan/fix/nom-du-bug-gamma`.
- Demande-moi toujours confirmation avant de changer de branche ou d'en créer une nouvelle.

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
