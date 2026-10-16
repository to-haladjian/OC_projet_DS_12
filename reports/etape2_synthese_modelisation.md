# Étape 2 — Synthèse de la modélisation

*Document destiné à alimenter le rapport métier final. Rédigé pour un lecteur non technique.*

---

## 1. Deux services, deux modèles, deux jeux de données

L'application doit répondre à deux questions différentes, et l'étape 1 a montré qu'aucun
jeu de données ne permet de répondre aux deux.

| | **`/predict`** — « quel rendement pour ma parcelle ? » | **`/recommend`** — « quelle culture choisir ? » |
|---|---|---|
| Jeu utilisé | Agriculture CropYield (1 000 000 de parcelles simulées) | CropYield Prediction reconstruit (données FAO réelles) |
| Ce qu'il apporte | les leviers de la parcelle : pluie, température, engrais, irrigation | l'écart réel entre cultures, absent du premier jeu |
| Modèle retenu | **régression linéaire** | **forêt d'arbres extrêmement aléatoires** |
| Erreur typique | **0,50 t/ha** | **1,7 t/ha**, pour des rendements de 1 à 16 t/ha |

Ce choix n'est pas une commodité : dans le jeu simulé, la culture n'a **aucun effet** sur le
rendement — toutes les cultures y rendent 4,65 t/ha à 0,03 t/ha près. Classer des cultures à
partir de ce jeu reviendrait à les ordonner au hasard. Il fallait un second modèle, sur les
données réelles.

## 2. `/predict` — un résultat inattendu : il n'y avait rien à optimiser

### Ce qui reste à apprendre a été testé avant de chercher mieux

Plutôt que d'enchaîner les modèles en espérant mieux, nous avons d'abord vérifié **s'il restait
quelque chose à apprendre**. Après une simple régression linéaire, nous avons demandé à un
modèle beaucoup plus souple — capable de repérer des combinaisons et des effets non
linéaires — de prédire **les erreurs** de la régression, à partir de **toutes** les
informations disponibles : culture, région, sol, météo, durée du cycle, profils FAO.

Il n'y parvient pas du tout : sa capacité à prédire ces erreurs est **nulle**. Elles sont du
**bruit pur**, d'écart-type 0,50 t/ha, et le bruit ne se prédit pas. Avec ces données, **aucun
modèle ne peut expliquer plus de 91,3 %** de la variation du rendement — et la régression
linéaire atteint 91,3 %.

*Note de méthode.* Une version précédente déduisait ce maximum directement de l'écart-type des
erreurs linéaires. C'était un raisonnement circulaire : calculé ainsi, le « maximum » est égal,
par construction, au score de la régression elle-même. Le test ci-dessus ne l'est pas.

### Le processus qui a engendré les données a été retrouvé

Les coefficients du modèle ne sont pas seulement prédictifs, ils sont lisibles :

```
rendement = 0,005 × pluie + 0,020 × température + 1,50 × engrais + 1,20 × irrigation + bruit
```

C'est exactement le chiffrage obtenu par l'analyse exploratoire, par une voie totalement
indépendante. Nous avons donc **reconstitué la formule de génération du jeu de données** —
ce qui explique pourquoi rien ne peut faire mieux.

**Ces chiffres décrivent le jeu simulé, pas l'agriculture réelle.** Confrontés aux données FAO
(étape 1), ils ne sont confirmés qu'en partie : les intrants vont dans le même sens, mais la
pluie n'y joue pas le rôle dominant qu'elle a ici. Ce sont des ordres de grandeur plausibles,
pas des effets agronomiques mesurés.

### La campagne d'optimisation a été menée quand même, et n'a rien trouvé

27 configurations testées sur trois familles de modèles, toutes journalisées :
**aucune ne bat la régression linéaire**, sur aucun échantillon de validation. Les modèles
plus complexes — forêts, boosting — font légèrement *moins* bien, car ils approximent par
paliers une relation qui est droite.

C'est un résultat, et il est plus solide qu'une amélioration de troisième décimale : nous
savons *pourquoi* la recherche ne trouve rien.

### Ce que cela donne pour l'utilisateur

| Mesure | Valeur |
|---|---|
| Erreur typique | **0,50 t/ha** |
| Estimations à moins de 0,5 t/ha de la réalité | **68 %** |
| Estimations à moins de 1 t/ha | **95 %** |
| Biais systématique | aucun (+0,0001 t/ha) |

**Les leviers, par ordre d'importance :** la pluie d'abord, puis la **fertilisation
(+1,50 t/ha)** et l'**irrigation (+1,20 t/ha)**. Les deux derniers sont des décisions que
l'exploitant prend lui-même : c'est là que se trouve la valeur d'un outil de conseil.

**Un point d'honnêteté :** l'estimation décrit **les conditions et les pratiques** de la
parcelle, pas la culture semée. Changer la culture ne déplace l'estimation que de 0,004 t/ha,
soit cent fois moins que l'erreur du modèle. L'interface demandera tout de même la culture —
l'utilisateur raisonne ainsi — mais la réponse le précisera.

**Et les pesticides ?** Le sujet les cite parmi les conditions de parcelle, mais le jeu simulé
n'en contient pas : l'estimation ne peut pas en tenir compte, et l'interface ne doit pas les
demander pour ce service.

## 3. `/recommend` — un piège de méthode évité

### Une évaluation naïve aurait menti

Le rendement d'un pays pour une culture donnée change très peu d'une année sur l'autre :
la corrélation avec l'année précédente atteint **0,985**. Une évaluation classique, qui
mélange les lignes au hasard, aurait donc placé des données **quasi identiques** d'un côté et
de l'autre : le modèle n'aurait eu qu'à reconnaître le pays pour retrouver la réponse.

Nous avons donc adopté une **évaluation chronologique** : le modèle apprend sur le passé et
doit prédire une année qu'il n'a jamais vue. L'écart est net — l'erreur mesurée passe de
1,19 à 1,76 t/ha, soit près de **50 % d'erreur supplémentaire**. La première mesure n'était
pas une performance, c'était une fuite.

**L'année 2013 a été mise de côté dès le départ** et n'a servi à aucun choix. Elle n'a été
utilisée qu'une seule fois, à la toute fin.

### Ce que le modèle sait, et ce qu'on a refusé de lui donner

Le code du pays améliore nettement les prédictions — sans surprise, il résume à lui seul le
niveau de développement agricole, les variétés et la mécanisation. Mais c'est une information
qu'un agriculteur ne peut pas « saisir », et qui empêcherait de servir un pays absent du jeu.

Nous lui donnons donc la **position géographique** du pays plutôt que son identité : deux
pays voisins se ressemblent alors, au lieu d'être deux catégories étrangères.

Autre précaution : les conditions de l'année à venir sont **inconnues** au moment de
recommander. Le modèle travaille donc sur la moyenne des **trois années précédentes**. On y
perd très peu, et le service devient réaliste.

### Un modèle nettement meilleur que le linéaire, cette fois

Contrairement à `/predict`, la relation n'a ici rien de linéaire : l'effet de la température
dépend de la culture, celui de la géographie n'est pas additif. Les modèles à arbres font
**3,5 fois moins d'erreur** qu'une régression, et la forêt d'arbres extrêmement aléatoires
l'emporte sur toutes les années de validation.

### Les réglages du modèle ont été cherchés, pas hérités

Nous avons mené pour ce service la même recherche de réglages que pour `/predict` — et pour
une meilleure raison : ici, rien n'interdisait a priori de gagner. **36 configurations** ont
été testées sur trois familles de modèles, toutes en évaluation chronologique.

Le verdict est net, et il va à l'encontre de ce qu'on espérait : **la recherche ne livre
rien d'utilisable.**

| Configuration | Erreur | Années améliorées | Taille du fichier |
|---|---|---|---|
| Réglage initial | 1,4514 t/ha | — | **39 Mo** |
| Meilleure, même taille | 1,4502 t/ha | 2 sur 5 | 50,4 Mo |
| Meilleure, toutes tailles | 1,4458 t/ha | 3 sur 5 | 100,7 Mo |

Trois raisons de conserver le réglage initial :

1. **Le gain est négligeable** : 0,0056 t/ha au mieux, quand l'erreur varie de 0,147 t/ha
   d'une année à l'autre. Le gain est **26 fois plus petit** que cette variation : il n'est
   pas distinguable du hasard de l'échantillonnage.
2. **Il ne tient pas d'une année sur l'autre** : 2 ou 3 années améliorées sur 5. Un gain qui
   ne se reproduit pas sur une majorité d'années n'est pas un gain.
3. **Les deux configurations « meilleures » dépassent la limite de taille** du fichier
   servi. La cause est mécanique : elles laissent les arbres pousser plus profond, ce qui
   alourdit le modèle de 29 % à 158 %.

Le réglage initial, légèrement plus contraint, produit des arbres plus courts — d'où à la
fois sa taille réduite **et** sa performance équivalente. Contrainte de livraison et
performance vont ici dans le même sens.

Ce qui change : **le réglage servi est désormais un choix mesuré, et non un réglage repris
ailleurs sans vérification.** C'est toute la différence entre « nous avons utilisé ces
valeurs » et « nous avons vérifié que ces valeurs sont les bonnes ».

### La qualité du classement, mesurée pour elle-même

Un modèle peut être précis en moyenne et proposer un ordre absurde. Or l'utilisateur ne voit
pas un rendement : il voit un **ordre**. Nous avons donc mesuré le classement directement — et
pas seulement celui qui sert à choisir le modèle, mais **celui que l'utilisateur verra**.

| La meilleure culture est proposée en 1re position | Modèle | Ordre figé (référence) |
|---|---|---|
| 10 cultures, en tonnes brutes (choix du modèle) | 93 % | 74 % |
| 10 cultures, en matière sèche | 82 % | 36 % |
| **4 cultures de l'application, en matière sèche** | **92 %** | 65 % |
| **4 cultures de l'application, en produit brut (USD/ha)** | **93 %** | 65 % |

*Mesures en validation chronologique (2008-2012), sur les couples pays × culture observés.*

Deux leçons :

- **Le premier chiffre surestimait ce que voit l'utilisateur.** En matière sèche, les écarts
  entre cultures se resserrent et l'ordre devient plus difficile : sur dix cultures, la
  première place passe de 93 % à 82 %. Sur les quatre cultures réellement servies, elle remonte
  à 92-93 %.
- **Se tromper coûte peu.** Quand le modèle ne place pas la meilleure culture en tête, il place
  une culture proche : suivre son classement coûte en moyenne **1,3 %** du produit brut de la
  meilleure culture, contre **11,8 %** pour un ordre figé. C'est la « précision économique »
  demandée par le sujet.

Le modèle apporte donc bien un classement qui **dépend du contexte**, et non une table figée.

## 4. Le défaut que nous avons corrigé : « pomme de terre » partout

Voici le problème le plus important de cette étape, et il n'est pas statistique mais
agronomique.

Classé en tonnes par hectare, le service place **presque toujours un tubercule en tête** :
la pomme de terre rend 16 t/ha quand le blé en rend 2,4. Un service qui répond « pomme de
terre » à 117 pays, quelles que soient les conditions saisies, n'est pas un service de
recommandation.

**La cause est simple : l'écart est surtout de l'eau.** Une pomme de terre est à 79 % d'eau,
un grain de blé à 13 %. Comparer leurs tonnages revient à comparer des choses différentes.

Nous avons donc ajouté deux corrections :

1. **Le classement est rendu en matière sèche**, ce qui rend les cultures comparables. La
   première place s'ouvre alors à nettement plus de cultures, et les céréales y apparaissent.
2. **Chaque culture porte son groupe** — céréales, tubercules, oléoprotéagineux — pour que
   l'interface puisse ne jamais opposer directement une pomme de terre et un blé.

Les deux classements, brut et en matière sèche, sont affichés : l'utilisateur voit le tonnage
qu'il récoltera *et* la comparaison honnête entre cultures.

### Un troisième garde-fou

Le modèle peut recommander une culture que le pays **n'a jamais produite**. Rien dans la
validation ne mesure ce cas : toutes les données d'évaluation portent sur des couples
pays × culture déjà observés. Ces cultures sont donc **signalées**, et nous recommandons de
présenter les **deux ou trois premières** cultures plutôt que la seule première place.

### L'orge et le coton retirés du classement

Absentes des données FAO, l'orge et le coton avaient reçu à l'étape 1 le profil d'une culture
« donneuse » (blé et sorgho) pour enrichir le dataset. Dans un classement, cet emprunt ne tient
pas : l'orge serait **toujours ex æquo avec le blé**, dont elle recopierait la prédiction, et le
coton porterait le rendement du sorgho — ce que l'étape 1 elle-même excluait. Elles sont donc
**retirées du classement**, et restent disponibles pour l'estimation de rendement. Le
classement porte sur les **quatre** cultures de l'application présentes dans les données FAO :
maïs, blé, riz, soja.

## 5. La question de la rentabilité

Le sujet demande « la culture la plus rentable ». Aucun des deux jeux ne contient de prix. Nous
avons ajouté une troisième source, gratuite et officielle : les **prix payés aux producteurs**
publiés par la FAO (FAOSTAT), par pays, culture et année. Cela permet de calculer un **produit
brut** : ce que la récolte rapporte à l'hectare, à la sortie de la ferme.

- Le prix retenu pour un pays est la médiane de ses cinq dernières années connues ; à défaut,
  la médiane mondiale, et l'interface le signale. Les valeurs manifestement erronées (plus de
  dix fois la médiane mondiale, comme le maïs mauritanien des années 1990, saisi en monnaie
  locale) sont écartées.
- **Ce n'est pas une marge.** Les coûts de production — semences, engrais, main-d'œuvre,
  irrigation — restent inconnus. Le service dit « produit brut », jamais « bénéfice ».

**Ce que nous avons appris :**

1. **Sur les quatre cultures de l'application, le classement en produit brut est plausible.**
   En France : maïs, puis riz, blé et soja.
2. **Sur dix cultures, il ne l'est pas.** L'igname, dont le prix mondial est le plus élevé,
   arriverait en tête dans 84 pays sur 117 — dont 68 qui n'en ont jamais cultivé. Deux
   incertitudes se multiplient : un rendement extrapolé et un prix d'emprunt. Le produit brut
   n'est donc calculé que sur les quatre cultures de l'application.
3. **Le critère change la réponse** : la culture classée première n'est pas la même en
   matière sèche et en produit brut dans la grande majorité des pays. L'interface doit dire
   sur quel critère elle classe.
4. **Le classement économique est sensible aux prix.** Quand les prix varient de ±20 %, la
   première culture reste la même dans la plupart des cas, mais pour **un pays sur deux** elle
   change au moins une fois sur dix. **Les prix doivent être affichés et modifiables** :
   l'agriculteur connaît son prix de vente mieux que la médiane FAO.

## 6. Traçabilité et reproductibilité


Toutes les évaluations sont enregistrées dans un journal d'expérimentations (MLflow), avec
pour chacune le modèle, les variables, le protocole et les résultats. Le classement des
modèles se lit directement dans ce journal.

- **Graine aléatoire fixée** partout : deux exécutions donnent des résultats identiques.
- **Le jeu de test n'a servi à aucun choix**, pour aucun des deux services.
- Chaque modèle est accompagné de ses métadonnées : variables attendues, bornes du domaine
  d'apprentissage, scores de référence, et avertissements à afficher.
- Un **contrôle automatique** vérifie que les coefficients de `/predict` retrouvent bien le
  générateur, et qu'aucune variable historique de `/recommend` ne contient d'information du
  futur. Ces contrôles font échouer l'entraînement en cas de régression silencieuse.

## 7. Ce qu'il faut dire, et ne pas taire

**Sur `/predict` :**

1. Le jeu est **simulé**, pas mesuré. Les relations apprises ne sont pas des preuves
   agronomiques, et les estimations sont des ordres de grandeur. Confrontée aux données FAO,
   la hiérarchie des leviers n'est confirmée qu'en partie (étape 1).
2. Le jeu n'a ni pays ni année : une estimation ne peut être rattachée à aucun lieu ni à
   aucune saison.
3. L'estimation **ne dépend pas de la culture**.

**Sur `/recommend` :**

4. Les conditions sont **nationales** : la pluie est une constante par pays, les pesticides
   un tonnage national qui dépend de la taille du pays. Le classement est un ordre de
   grandeur à l'échelle d'un pays, **pas d'une parcelle**.
5. Chaque pays ayant une position unique, le modèle **peut** s'en servir pour le reconnaître.
6. La recommandation d'une culture **nouvelle pour un pays** n'est pas mesurée par la
   validation.
7. **117 pays** servis, données s'arrêtant en **2013** : les autres pays ne peuvent pas être
   servis, et une campagne récente demanderait des données à jour.
8. **Le produit brut n'est pas une rentabilité.** Les prix viennent d'une source externe
   (FAOSTAT), souvent de la médiane mondiale faute de prix national, et les coûts de
   production sont inconnus. Le service peut désigner la culture qui se vend le plus cher à
   l'hectare, pas la plus rentable.
9. **L'orge et le coton ne sont pas classés** : absents des données FAO, leur rang n'aurait été
   qu'une copie.

**Sur les deux :**

10. Les colonnes d'enrichissement ajoutées au dataset consolidé à l'étape 1 **ne sont
   utilisées par aucun des deux modèles** : `/predict` n'en tire rien, et `/recommend`
   travaille directement sur les données FAO. C'est la conséquence logique de la stratégie de
   fusion retenue, et elle était annoncée à l'étape 1.

---

### Pour l'étape 3 (API)

Les deux modèles sont sauvegardés avec leurs métadonnées. Les bornes du domaine
d'apprentissage, produites à l'étape 1, servent à **signaler une saisie inhabituelle sans la
refuser** : sortir de ces bornes n'est pas une erreur, c'est une zone où le modèle a peu
appris. L'API de l'étape 3 applique cette règle.

Le classement servi en découle directement :

- il porte sur les **quatre cultures** de l'application présentes dans les données FAO ;
- il affiche le rendement brut, la matière sèche **et** le produit brut, et dit sur quel
  critère il classe ;
- les **prix sont affichés et modifiables**, avec leur provenance (pays ou médiane mondiale) ;
- il met en avant les **deux ou trois premières** cultures et signale celles que le pays ne
  cultive pas.
