# Optimisation-d-achat-de-100-actions-sur-60-jours

S0 = 100 le prix initial de l'action 
S_(n+1) = S_n + sigma*X_(n+1) les prix des jours suivants
où (X_n) sont des loi normales N(0,1) i.i.d.
A_n = 1/n*sum(S_k) k=1..n  (A_0 = S_0) est la moyenne du prix 
but : nous voulons acheter 100 stocks sur 60 jours.
v_n * S_(n+1) désigne le prix qu'on a dépensé pour acheter v_n action
sachant que à la fin on doit avoir somme (v_n) = 100 
chaque soir à partir du 20e jour, on peut sonner une cloche uniquement lorsque 100 actions ont été achetés, et le jeu s'arrête alors.
Nous souhaitons maximiser ceci :100*A_n - (ce que l'on a dépensé)

# 1ère version
LinearPolicy initialise une politique avec des poids aléatoires.
simulate_episode simule un épisode complet, retournant les états, actions et récompenses pour chaque jour.
evaluate_policy évalue et met à jour la politique sur plusieurs épisodes, cherchant le meilleur épisode.

Pour la simulation d'un épisode :
Chaque jour, le prix de l'action est mis à jour et l'état est normalisé.
Une action est choisie en fonction de l'état normalisé.
La cloche est sonnée si les conditions sont remplies et l'épisode s'arrête.

Évaluation et mise à jour de la politique :
Pour chaque épisode, les états, actions et récompenses sont enregistrés.
La politique est mise à jour en utilisant le gradient de politique basé sur les retours.


# 2ème version nn_continuous_(final) :
Ce programme implémente une stratégie d'achat d'actions utilisant des réseaux de neurones et des méthodes de reinforcement learning. Le modèle apprend à maximiser les gains en achetant des actions de manière optimale sur une période de 60 jours.

1. Initialisation des paramètres
Définition des paramètres initiaux du problème, y compris le prix initial de l'action, la volatilité (sigma), le nombre de jours et l'objectif de nombre d'actions à acheter.
2. Simulation des prix
Fonction pour simuler les prix des actions sur les days jours suivants, à partir d'un prix initial et d'une série de variations quotidiennes.
3. Définition du gain
Fonction pour calculer la récompense en tenant compte du coût total (on pourrait rajouter une pénalité pour les grands achats en une seule journée).
4. Définition du modèle de réseau de neurones
Un réseau de neurones avec trois couches cachées, produisant des sorties pour la moyenne et l'écart-type des distributions de nombre d'actions à acheter et de décision de sonner la cloche.
5. Normalisation de l'état
Fonction pour normaliser les variables d'état avant de les passer au modèle.
6. Simulation d'un épisode
Fonction pour simuler un épisode complet, où le modèle prend des décisions d'achat d'actions sur une période de 60 jours.
7. Calcul des retours
Fonction pour calculer les retours cumulés pour chaque étape de l'épisode.
8. Entraînement du modèle
Fonction pour entraîner le modèle sur plusieurs épisodes, en ajustant les poids du réseau de neurones pour maximiser la récompense.
9. Évaluation de la politique
Fonction pour évaluer la politique du modèle sur plusieurs épisodes, en calculant les statistiques moyennes des performances.
10. Initialisation et entraînement du modèle
Initialisation du modèle, application de l'initialisation des poids, entraînement du modèle et affichage des résultats moyens sur plusieurs épisodes.


# Modification effectuée 

-retourner la moyenne des episodes et non plus le dernier épisode, ni le meilleur ce qui est plus robuste pour évaluer la performance de la politique

Achat de fraction d'action
retourner la moyenne des episodes et non plus le meilleur, ou le dernier épisode ce qui est plus robuste pour évaluer la performance de la politique

Normalisation changée (min max)
distribution gaussienne avec torch.distributions.Normal


supprimer la possibilité de vendre + que ce qu'on possède (stock market)

Exportation excel,
possibilitée de short sell
réarrangement du décalage des prix 

constat les log_probs sont parfois plus grande que 0 et donc les prob associées plus grandes que 1... Clamp?
log_prob = log_prob.clamp(max=0)

problème pour le moment, nous n'atteignons pas les 100 actions ciblées à la fin de chaque episode.

modif à faire:

- séparer l'environnement, l'agent avec la simulation pour pouvoir comparer plusieurs agents.

- log prob = densité, tester en appliquant la formule de la loi normale

- introduire de la volatilité stocha dans le simulateur

- pour atteindre le goal : si on a sonné ou bien on a atteint days=60 : a la fin overrider goal - total_stock 
 
sinon : on fait rien
 



 MODIF à faire pour le dimanche 4 AOUT 24:

 Claire séparation de l'environnement, l'agent, et la simulation (refaire les fonctions pour les généraliser si necessaire) presque fini
 Coder les fonctions bernouilli, log_density à la main et vérifier que la moyenne à un gradient X
 Approfondir modèle Heston
 Mise à jour du programme de rachat action : cv state : mean -> prix-mean
 enregistrer un modèle et le charger

Modif Gilles:
Separation de simulate_episode en deux : simulate_episode et execute_step pour mettre execute step dans env
Correction des bugs (rajout des variables manquantes, et suppression des variables en exces, compilation de certaines lignes de code quand ses possibles)
suppression de la verification du gradient de la mean car ca apporte un bug
