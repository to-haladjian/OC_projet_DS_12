# Étape 1 — Synthèse des choix de préparation et de fusion des données

*Document destiné à alimenter le rapport métier final. Rédigé pour un lecteur non technique.*

---

## 1. Ce que nous avions au départ

Deux jeux de données nous ont été fournis. Ils décrivent tous les deux des rendements
agricoles, mais pas à la même échelle.

| | **Agriculture CropYield** | **CropYield Prediction** |
|---|---|---|
| Unité observée | une parcelle | un pays × une culture × une année |
| Volume | 1 000 000 de lignes | 25 527 lignes (après reconstruction) |
| Contient | sol, irrigation, fertilisation, météo, culture | pays, année, pluie, pesticides, température, culture |
| Ne contient **pas** | **ni pays, ni année** | **ni sol, ni irrigation, ni pratique agricole** |
| Origine | simulée | FAO / Banque mondiale |

Le premier décrit la parcelle telle que l'agriculteur la connaît — c'est exactement ce que
l'utilisateur saisira dans l'application. Le second décrit la réalité agronomique
mondiale observée sur 203 pays entre 1990 et 2013.

## 2. Le problème central : sans fusion, la recommandation n'a aucun sens

C'est le constat le plus important de cette étape, et il conditionne tout le projet.

**Dans Agriculture CropYield, la culture choisie n'a aucune influence sur le rendement.**
Les six cultures donnent le même résultat moyen :

| Culture | Rendement moyen (t/ha) |
|---|---|
| Orge | 4,67 |
| Coton | 4,65 |
| Riz | 4,65 |
| Maïs | 4,64 |
| Soja | 4,64 |
| Blé | 4,62 |

Soit **0,05 t/ha** d'écart entre la meilleure et la moins bonne, un rapport de 1,00. Le
même constat vaut pour la région, le type de sol et la météo.

Nous avons poussé la vérification plus loin, car des moyennes identiques pourraient
masquer un effet conditionnel — une culture médiocre en moyenne mais excellente sous
irrigation, par exemple. **Il n'en est rien :**

- à pluviométrie comparable, les six cultures s'écartent de **0,028 t/ha** au maximum ;
- l'effet de l'engrais varie de +1,494 à +1,514 t/ha selon la culture, celui de
  l'irrigation de +1,189 à +1,209, soit **0,02 t/ha** de variation ;
- aucun plan de l'analyse en composantes principales ne sépare les cultures.

Autrement dit : dans ce jeu, *ce que l'on plante ne change rien*, et pas seulement en
moyenne.

Or la fonction de recommandation demandée consiste précisément à **classer les cultures
entre elles**. Un modèle entraîné sur ce seul jeu produirait un classement **aléatoire**,
qui changerait à chaque ré-entraînement tout en paraissant crédible à l'utilisateur.
C'est le risque métier majeur de ce projet.

**CropYield Prediction, lui, contient cette information.** Les rendements y varient d'un
facteur **12,8** entre cultures (pomme de terre 15,96 t/ha contre sorgho 1,25 t/ha).

> **La fusion des deux jeux n'est donc pas une formalité technique : c'est elle qui rend
> la fonction de recommandation envisageable.**

## 3. Un défaut du fichier fourni, et sa correction

Le sujet fournit un fichier déjà fusionné, `yield_df.csv`. Nous l'avons audité avant de
l'utiliser, et nous avons **choisi de ne pas nous en servir**. Deux raisons, mesurées.

**Premièrement, il ne contient que 101 pays.** Les quatre fichiers sources nomment les
pays de trois façons différentes : forme longue de la FAO (« United States of America »,
« Viet Nam »), noms courants (« United States », « Vietnam ») et style Banque mondiale
(« Venezuela, RB »). Le fichier fourni a été assemblé en rapprochant les noms
**littéralement**, ce qui n'a fonctionné que pour 101 pays sur 270 noms distincts. Les
pays écartés ne sont pas marginaux : **États-Unis, Chine, Russie, Iran, Viêt Nam, Corée
du Sud, Tanzanie, Tchéquie, Turquie** — parmi les premiers producteurs mondiaux.

**Deuxièmement, 54 % de son contenu est redondant.** Le fichier des températures contient
jusqu'à **41 relevés** pour un même couple pays-année. Joint sans regroupement préalable,
il démultiplie les lignes : 3 837 combinaisons se répètent, soit 15 112 lignes
excédentaires. Une analyse menée sur ce fichier accorderait un poids jusqu'à 41 fois
supérieur à certains pays.

**Notre correction.** Nous avons reconstruit le jeu depuis ses quatre fichiers sources,
en rapprochant les pays par leur **code international ISO3** plutôt que par leur nom, et
en regroupant les relevés de température **avant** toute jointure. Chaque jointure est
suivie d'un contrôle automatique qui échoue si le nombre de lignes change.

Résultat : **120 pays** appariés au lieu de 101, tous les grands producteurs présents, et
**25 527 lignes** sans aucun doublon, au lieu de 13 130. Nous avons corrigé la cause du
problème, pas son symptôme.

## 4. La stratégie de fusion retenue

### Pourquoi une jointure classique était impossible

Une fusion ligne à ligne suppose une clé commune. Il n'y en a pas : le premier jeu ignore
le pays et l'année, le second ignore le sol et l'irrigation. **La culture est la seule
dimension partagée.**

Nous avons également écarté l'idée d'empiler les lignes des deux jeux : les unités
d'observation sont incompatibles (une parcelle contre un pays sur une année), et **les
variables climatiques de même nom ne mesurent pas la même chose.**

Pour la pluie, le constat est immédiat : celle du jeu simulé plafonne à 1 000 mm alors que
la médiane du jeu historique dépasse cette valeur. L'une est une pluie de saison culturale,
l'autre une normale annuelle.

Pour la température, le piège est plus discret, puisque l'unité est la même des deux côtés —
des degrés Celsius. Nous avons donc vérifié, et le résultat est net. Le jeu simulé ne
descend **jamais sous 15,0 °C**. Or, dans les données réelles :

- **44 % des pays-années cultivant du blé** sont sous ce seuil en moyenne annuelle ;
- **53 pays sur 137** également, dont la **Russie** (4,0 °C), le **Kazakhstan** (5,2 °C) et
  la **Finlande** (5,5 °C) — tous grands producteurs de blé.

Si la température du simulateur était une moyenne annuelle, il écarterait donc près de la
moitié de la production mondiale de blé. C'est exclu pour un jeu dont nous avons établi que
la grille de modalités est exhaustive : un catalogue construit pour couvrir tous les cas ne
supprimerait pas la moitié de la ceinture céréalière. En revanche, un plancher de 15 °C est
parfaitement logique pour une **température de croissance** : en dessous, une céréale ne
pousse quasiment pas.

**Conclusion : le simulateur décrit une saison culturale, pas une année civile**, et cela
pour ses deux variables climatiques. C'est une correction que nous avons apportée après
coup : nous avions d'abord jugé les deux températures comparables au motif qu'elles
partageaient la même unité. Une unité commune ne garantit pas une grandeur commune.

### La solution : enrichir chaque parcelle par le profil mondial de sa culture

Le jeu simulé reste la **colonne vertébrale** — il porte les leviers que l'agriculteur
maîtrise. Le jeu historique est **résumé en un profil agronomique par culture**, puis
rattaché à chaque parcelle selon la culture plantée.

Six indicateurs sont ainsi ajoutés :

| Indicateur ajouté | Ce qu'il apporte |
|---|---|
| Rendement de référence mondial | **le différentiel entre cultures qui manquait** |
| Dispersion du rendement | l'ampleur du risque selon les contextes |
| Intensité d'usage de pesticides | indicateur indirect du niveau d'intensification |
| Niche thermique de la culture | la température **annuelle** des pays où la culture est réellement cultivée |
| Niveau de pluviométrie habituel | le besoin en eau relatif de la culture |
| Écart à la niche thermique | le seul indicateur croisant les deux sources — mais voir la limite ci-dessous |

Profils obtenus :

| Culture | Rendement de référence (t/ha) | Niche thermique (°C) | Pluviométrie (percentile) |
|---|---|---|---|
| Riz | 3,03 | 22,3 | 54 |
| Blé | 2,31 | 16,3 | 30 |
| Orge* | 2,31 | 16,3 | 30 |
| Maïs | 2,30 | 20,2 | 44 |
| Coton* | 2,20 | 20,9 | 39 |
| Soja | 1,46 | 18,8 | 48 |

\* profils imputés — voir §6.

> **Limite de l'écart à la niche thermique.** Cet indicateur retranche la température de la
> parcelle à celle de la niche de la culture. Les deux ne portant pas sur la même fenêtre —
> saison culturale d'un côté, année civile de l'autre — **l'écart obtenu n'est pas un écart
> agronomique**. Nous le conservons parce qu'il porte l'intention dont la recommandation a
> besoin, mais comme une heuristique assumée, à valider ou écarter lors de la modélisation.

Ces valeurs sont **agronomiquement cohérentes** : le blé et l'orge apparaissent comme les
cultures les plus tempérées et les moins exigeantes en eau, le riz comme la plus chaude et
la plus gourmande. Trois contrôles automatiques vérifient ces ordres à chaque
reconstruction et font échouer la chaîne si l'un d'eux se rompt.

### Un piège méthodologique évité

Notre première approche définissait la « température optimale » d'une culture à partir des
pays-années où elle rend le mieux. Le résultat était aberrant : **12,9 °C pour le blé** et
14,1 °C pour le maïs.

L'explication est instructive : les rendements les plus élevés au monde sont obtenus dans
les pays **développés et tempérés**. Cet indicateur mesurait donc le **niveau de richesse
des pays**, pas l'agronomie. Nous l'avons remplacé par la **température médiane de
culture**, qui rétablit un classement conforme aux connaissances agronomiques.

## 5. Les décisions de nettoyage

| Jeu | Problème constaté | Décision | Justification |
|---|---|---|---|
| Simulé | 231 parcelles à **rendement négatif** | supprimées | physiquement impossible ; toutes sans engrais ni irrigation et 223 sur 231 sous 200 mm de pluie — c'est le bruit du générateur passant sous zéro. 0,023 % des données |
| Simulé | Irrigation et fertilisation en texte | converties en oui/non | typage nécessaire à la modélisation |
| Simulé | Durée de cycle sans lien avec le rendement | **conservée** | son retrait est une décision de modélisation, pas de nettoyage |
| Simulé | Météo indépendante de la pluie et de la température | signalée, conservée | la colonne ne porte aucune information (549,5 / 550,1 / 550,3 mm selon la météo déclarée) |
| Historique | Jusqu'à 41 relevés de température par pays-année | regroupés par moyenne **avant** jointure | supprime la cause des doublons |
| Historique | Trois conventions de nommage des pays | appariement par code ISO3 | récupère 19 pays, dont les plus gros producteurs |
| Historique | Pluie absente certaines années, et 4 années aberrantes | **une valeur par pays — la modale — diffusée sur toutes ses années** | la pluie est une normale climatique constante par pays : cette règle comble les trous sans rien inventer et écarte les aberrations |
| Historique | 11 pays dont la série de pluie est copiée du pays voisin | pluie mise à « inconnu » | toute la série est fausse ; il n'y a rien à retenir |
| Historique | Rendements en hectogrammes/ha | convertis en tonnes/ha | harmonisation des unités |

Une règle a guidé l'ensemble : **l'analyse exploratoire constate, elle ne tranche pas.**
Chaque décision est prise après que son utilité a été établie, et consignée avec son
motif.

### Toutes les valeurs manquantes ne se traitent pas de la même façon

Un pourcentage de valeurs manquantes ne dit pas comment les traiter. Nous avons donc
distingué deux situations, et elles appellent des décisions opposées :

| Variable | Pays sans aucune valeur | Pays avec des trous isolés |
|---|---:|---:|
| Température | 66 | **0** |
| Pesticides | 41 | **1** |
| Pluie | 18 | **181** |

Pour la température et les pesticides, la manquance est **tout ou rien** : aucun pays n'a
de trou partiel. Il n'y a donc rien à reprendre, et écarter ces pays des analyses qui en
dépendent est la seule option qui n'invente aucune valeur.

La pluie suivait le schéma **inverse** : 181 pays avaient une valeur connue, mais pas pour
toutes les années. La cause est identifiée — le fichier source **ne contient aucune ligne
pour 2003**.

Or la pluviométrie y est une normale climatique **constante par pays** : 188 pays sur 192
portent exactement la même valeur toutes les années. Nous avons donc retenu **une valeur
par pays, la plus fréquente, diffusée sur toutes ses années**. Cette règle unique produit
trois effets, sans aucune hypothèse extérieure aux données :

1. **Les trous disparaissent**, 2003 comprise.
2. **Quatre années aberrantes sont écartées** : l'Inde portait 1 940 mm une année contre
   1 083 mm les trente autres — 1 940 étant précisément la valeur de l'Islande, le pays
   alphabétiquement précédent dans le fichier source. Le Paraguay reprenait de même celle
   de la Papouasie-Nouvelle-Guinée. La majorité l'emporte sur l'accident. L'Inde étant le
   premier contributeur du jeu de rendements, cette correction compte.
3. **Onze pays sont écartés de la mesure de pluie** : leur série entière, et pas une seule
   année, est identique à celle du pays alphabétiquement précédent — signature d'un
   remplissage vers le bas dans la source. Parmi eux, cinq portent des données de
   rendement : Angola, Micronésie, Serbie, Monténégro, Vanuatu. Ici la valeur la plus
   fréquente ne répare rien, puisque toute la série est fausse ; leur pluviométrie est donc
   déclarée inconnue, leurs autres variables restant exploitables.

Bilan : **16 296 lignes complètes pour 117 pays**, contre 15 794 pour 119 avant correction.
Nous avons gagné des données réelles et retiré des données fabriquées.

### Le domaine couvert par les données est borné

Pour chaque culture et chaque variable, les quantiles 5 % et 95 % délimitent les 90 %
centraux observés (`data/processed/domaine_apprentissage.csv`). Ces bornes sont
agronomiquement cohérentes — l'igname démarre à 16,2 °C, la pomme de terre descend à
6,2 °C — et serviront à l'application pour **signaler une saisie inhabituelle sans la
refuser** : sortir de ces bornes n'est pas une erreur, c'est une zone où le modèle a peu
appris. L'année 2013, réservée au test final du moteur de recommandation, est exclue de ce
calcul : aucune information issue de cette année ne doit circuler en amont.

## 6. Le traitement de l'orge et du coton

Après harmonisation des libellés (`Rice` → `Rice, paddy`, `Soybean` → `Soybeans`),
**quatre des six cultures** du jeu simulé trouvent leur équivalent dans le jeu historique.
**L'orge et le coton en sont totalement absents.**

Deux options s'offraient à nous : les exclure — mais l'application n'aurait plus
recommandé que quatre cultures et un tiers du jeu aurait été jeté — ou leur attribuer un
profil de substitution documenté. **Nous avons retenu la seconde**, avec une exigence de
transparence :

- **Orge** ← profil du **blé**, transféré intégralement. L'orge est une céréale tempérée
  de calendrier, d'exigences et de niveau de rendement comparables.
- **Coton** ← profil climatique du **sorgho** (culture chaude et semi-aride), mais
  rendement de référence fixé à **2,2 t/ha**, moyenne mondiale FAO du coton-graine. Aucune
  céréale ne peut servir d'approximation au rendement d'une culture textile.

Chaque ligne porte une colonne indiquant si son enrichissement vient des données FAO ou
d'une imputation. Au final, **666 494 lignes** reposent sur des données réelles et
**333 275** sur une imputation explicite.

**Cette imputation vaut pour enrichir le dataset, pas pour classer.** Dans un classement,
l'orge recopierait le rang du blé et le coton celui du sorgho. L'étape 2 retire donc ces deux
cultures du classement de la recommandation ; elles restent disponibles pour l'estimation de
rendement.

## 7. Les variables clés identifiées

L'analyse en composantes principales demandée dans le sujet a été réalisée en deux passes.

**Avec le rendement**, un axe se détache nettement (32,6 % de la variance) et associe
rendement, pluie, engrais et irrigation, tous de même signe : c'est l'axe de la
productivité. Un détail est instructif : le dernier axe ne porte que **0,74 %** de la
variance, ce qui signifie que le rendement est, à très peu près, une **combinaison
linéaire** des autres variables. C'est une mesure directe du mode de génération du jeu.

**Sans le rendement**, aucun axe ne domine : chacun porte environ 20 % de la variance et
**les cinq sont nécessaires** pour atteindre 90 %. Les variables explicatives sont
mutuellement indépendantes, sans redondance à compresser.

Ce n'est pas un échec de l'analyse mais une mesure, et elle a une conséquence pratique :
**il ne faut appliquer aucune réduction de dimension à l'étape suivante**, elle
détruirait de l'information sans rien simplifier.

Une ACP ne pouvant pas hiérarchiser des variables selon leur utilité prédictive — elle ne
regarde jamais le rendement —, le classement vient de mesures directes :

| Variable | Corrélation | Part du rendement expliquée | Effet moyen |
|---|---|---|---|
| **Pluviométrie** | **0,765** | **58 %** | facteur dominant |
| **Fertilisation** | 0,442 | 20 % | **+1,50 t/ha** |
| **Irrigation** | 0,354 | 13 % | **+1,20 t/ha** |
| Température | 0,086 | 1 % | +0,445 t/ha du plus froid au plus chaud |
| Durée du cycle | −0,004 | 0 % | aucun |

La colonne du milieu est celle qu'il faut lire : une corrélation est difficile à
interpréter seule, alors que la part du rendement expliquée se compare directement d'une
variable à l'autre.

**Une précision nécessaire sur la lecture de ces chiffres.** Les corrélations de la
fertilisation et de l'irrigation, 0,44 et 0,35, sont **modérées** — il serait abusif de les
qualifier de fortes. Seule la pluviométrie entretient une relation vraiment forte avec le
rendement.

Mais la corrélation **sous-estime l'importance de ces deux leviers**, parce qu'ils sont des
variables en tout ou rien. L'engrais apporte +1,50 t/ha, soit un écart d'environ **un
écart-type** entre parcelles fertilisées et non fertilisées : c'est un effet important. Son
coefficient de corrélation reste pourtant modéré, car il dépend aussi de la répartition
entre les deux groupes. Pour un levier de ce type, **la mesure à retenir est l'effet en
t/ha**, pas la corrélation.

**Quatre enseignements pour l'agriculteur :**

1. **L'eau est le facteur dominant**, loin devant tous les autres : elle explique à elle
   seule près de 60 % des écarts de rendement.
2. **Fertilisation et irrigation arrivent ensuite, et ce sont deux leviers directement
   actionnables** : +1,50 et +1,20 t/ha. C'est précisément ce qui donne sa valeur à une
   application de conseil — elle porte sur des décisions que l'exploitant peut prendre.
3. **La température a un effet faible mais parfaitement régulier.** Sa corrélation de 0,09
   aurait conduit à l'écarter à tort ; l'analyse par tranches montre une progression sans
   aucune exception. À conserver.
4. **La durée du cycle n'a aucun effet** sur le rendement dans ces données.

### Ces variables clés résistent-elles aux données réelles ?

Le sujet confie au second jeu un rôle précis : **valider** l'analyse du premier. Tous les
chiffres ci-dessus viennent du jeu simulé. Nous avons donc cherché les mêmes relations dans les
données FAO, de deux façons : d'un pays à l'autre, et d'une année sur l'autre au sein d'un même
pays. Même mesure des deux côtés, la corrélation de rang.

| Variable | Jeu simulé | FAO, entre pays (médiane des 10 cultures) | FAO, d'une année sur l'autre |
|---|---|---|---|
| Pluie | **+0,77** | −0,06 | non mesurable (constante par pays) |
| Température | +0,08 | **−0,32** (négative pour 9 cultures sur 10) | −0,02 |
| Intrants (engrais ; proxy FAO : pesticides) | +0,43 | **+0,46** (positive pour les 10) | +0,02 |
| Irrigation | +0,34 | absente du jeu FAO | — |

**La validation n'est que partielle, et c'est un résultat à dire :**

- **La pluie, facteur dominant du jeu simulé, n'est pas liée au rendement dans les données
  réelles.** La raison est identifiée plus bas : la pluie FAO ne mesure pas l'eau que reçoivent
  les cultures, à cause de l'irrigation.
- **La température change de signe** : les pays chauds rendent moins. Mais ce sont aussi, en
  moyenne, des pays moins équipés, et d'une année sur l'autre le lien disparaît. On ne peut en
  conclure ni pour ni contre un effet agronomique.
- **Seule l'intensité en intrants va dans le même sens**, pour les dix cultures — mais
  seulement d'un pays à l'autre. Elle décrit des agricultures intensives, pas l'effet d'un
  levier qu'on actionnerait.

> **Conséquence pour le rapport métier.** « +1,50 t/ha avec engrais, +1,20 t/ha avec
> irrigation, la pluie explique 58 % » décrivent **le jeu simulé**. Ce sont des ordres de
> grandeur plausibles, pas des faits agronomiques mesurés.

### Des proxies pour l'irrigation et la fertilisation

Le jeu FAO ne dit rien des pratiques. Deux de ses variables peuvent en tenir lieu, avec des
limites précises :

- **Fertilisation → pesticides.** Ce sont les deux intrants achetés d'une agriculture
  intensive, et le seul intrant disponible. Mais c'est un tonnage **national**, qui croît avec
  la taille du pays, et non une dose par hectare.
- **Irrigation → pluie très faible.** Sous 300 mm par an, un maïs ou un riz pluvial est
  impossible ; or les rendements de ces pays ne sont pas plus faibles que ceux des pays les plus
  arrosés. L'Égypte, quasiment sans pluie, produit du riz autour de **9 t/ha**, trois fois la
  médiane mondiale. Ces cultures sont irriguées : **une pluie annuelle sous 300 mm signale une
  agriculture irriguée**. Le proxy repère l'irrigation là où elle est certaine ; il ne la mesure
  pas ailleurs.

## 8. Le livrable et ses limites

Le fichier `data/processed/crop_dataset_consolidated.csv` contient **999 769 lignes et 18
colonnes**, sans aucune valeur manquante. Il constitue la source de vérité unique des
étapes suivantes, et un dictionnaire des données l'accompagne.

**Quatre limites doivent être énoncées ouvertement.**

**1. Le jeu simulé n'est pas mesuré.** Distributions uniformes, modalités parfaitement
équilibrées, aucune valeur manquante, quelques rendements négatifs, et une météo tirée
indépendamment de la pluie : ce sont les signatures d'une génération artificielle. Le
modèle apprendra un monde plausible mais synthétique. Les prédictions doivent être
présentées comme des ordres de grandeur, non comme des garanties.

**2. L'enrichissement par culture n'explique pas le rendement simulé.** C'est le point le
plus délicat, et il faut l'énoncer sans détour. Le rendement de référence ne prend que six
valeurs distinctes, une par culture ; or la cible reste le rendement du jeu simulé, qui ne
varie pas selon la culture. Nous avons mesuré la corrélation entre les deux : **−0,0007**,
soit zéro. Un modèle entraîné sur cette cible apprendra donc à ignorer cette colonne.

*Conséquence de conception :* la recommandation ne peut pas reposer sur un modèle entraîné
sur ce jeu. Elle aura son **propre modèle, appris sur les données FAO**, où la culture change
réellement le rendement (étape 2). Une première idée, classer les cultures par un ordre figé
tiré du rendement de référence, a été abandonnée : un ordre figé ne dépend pas du contexte,
alors que l'ordre réel change d'un climat à l'autre.

*Ce que cela implique, à dire clairement :* le dataset consolidé reste le livrable unique
demandé et la source de l'estimation de rendement, mais **ses colonnes d'enrichissement ne
servent à aucun des deux modèles**. La fusion a été utile comme **analyse** : c'est elle qui a
montré que le jeu simulé ne départage pas les cultures, et qui a fixé l'architecture à deux
modèles.

**3. Le classement reste un ordre de grandeur.** Les variables du jeu historique décrivent
le climat moyen d'un pays. Ce qui différencie les cultures tient donc pour une large part
à *l'endroit où elles sont cultivées*, et non à leurs seules qualités agronomiques.

**4. Aucune donnée économique dans les deux jeux.** Ni prix, ni coût, ni marge. L'étape 2
ajoute une source externe et gratuite, les **prix au producteur FAOSTAT**, qui permet de
calculer un **produit brut** par hectare. Les coûts restant inconnus, le service ne pourra pas
désigner la culture la plus **rentable**, seulement celle qui se vend le plus cher à
l'hectare. L'interface doit employer les mots justes.

---

### Points reportés aux étapes suivantes

- Trancher le retrait de la durée de cycle et des variables nominales sans effet.
- Vérifier, par l'importance des variables du modèle, que le rendement de référence n'est
  pas traité comme un simple ré-encodage du nom de la culture.
- Confirmer l'apport de l'écart à la niche thermique, ou l'écarter : il soustrait une
  température de saison culturale à une moyenne annuelle (voir §4), et le jeu historique
  ne le valide pas
  à lui seul, le biais de richesse des pays y brouillant la relation.
- Le CSV consolidé pèse 173 Mo, au-delà de la limite de 100 Mo par fichier de GitHub : à
  résoudre avant la mise en place de l'intégration continue (étape 5).
