---
title: Agritech Answers API
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
---

# Agritech Answers — API

Estimation du rendement d'une parcelle (`POST /predict`) et classement des cultures pour un
pays (`POST /recommend`). Documentation interactive sur `/docs`, état du service sur
`/health`.

Ce Space est **généré automatiquement** : il est poussé par le pipeline du dépôt
[to-haladjian/OC_projet_DS_12](https://github.com/to-haladjian/OC_projet_DS_12) à chaque
succès sur la branche `dev`. Ne pas l'éditer ici, les modifications seraient écrasées au
prochain déploiement.

Deux points à connaître sur les réponses du service :

- **L'estimation de `/predict` ne dépend pas de la culture choisie.** Dans les données
  d'entraînement, le rendement dépend des conditions et des pratiques de la parcelle, pas de
  l'espèce semée. La réponse le signale.
- **Le classement de `/recommend` est un ordre de grandeur à l'échelle d'un pays**, pas
  d'une parcelle, et il parle de rendement attendu — jamais de rentabilité, aucune donnée
  économique n'étant disponible.
