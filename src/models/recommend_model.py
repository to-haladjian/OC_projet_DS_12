"""`/recommend` model: ranking of crops by expected yield.

This is where the step 2 modelling work lives. Unlike `/predict`, this service relies
on a dataset where **the crop actually predicts yield**: the FAO dataset rebuilt in
step 1.

Three design choices shape this module.

1. **Temporal validation, never random.** The yield of a country x crop pair is
   correlated at 0.985 with that of the previous year. A random split would put
   near-identical rows on both sides and overestimate performance by a factor close
   to 5. We train on the past and predict the following year.
2. **No country identifier.** The model receives the country's *position*, not its
   code: two neighbouring countries then look alike, instead of being two unrelated
   levels.
3. **A ranking in raw tonnes makes no sense across crops.** Potato yields 16 t/ha
   against 2.4 t/ha for wheat, but the gap is mostly water. The ranking is therefore
   given in **dry matter**, and per agronomic group. When producer prices are
   available (`src.data.prix`), the **gross product in USD/ha** is added: it answers
   the brief's economic question, within the limits stated in that module.

Runnable as a script to retrain and save the served model.
"""

from __future__ import annotations

import json
import lzma
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.data import prix as prix_fao
from src.data.crop_mapping import CROP_TO_FAO_ITEM
from src.models import evaluation, features, historique, protocole

#: Validation years: each is predicted by a model trained on everything before it.
ANNEES_VALIDATION = [2008, 2009, 2010, 2011, 2012]

#: Chosen representation: the country's past conditions + geography + year.
VARIABLES = features.RECOMMEND_JEUX["historique"]


def charge() -> pd.DataFrame:
    """Read the rebuilt FAO dataset, add the derived variables, and check for leakage.

    Rows without history — the first year of each country — and those with a missing
    variable fall out of scope: a model cannot learn from an incomplete row, and there
    is nothing to impute (see step 1).
    """
    brut = pd.read_csv(config.CROP_YIELD_PREDICTION_CSV)
    prepare = historique.prepare(brut)
    historique.controle_absence_de_fuite(prepare)
    return prepare.dropna(subset=VARIABLES + [features.CIBLE_RECOMMEND]).reset_index(drop=True)


#: Variables with no natural order: to one-hot encode if they are in the chosen set.
#: `iso3` only appears in the ablation sets of notebook 09, which measure its contribution
#: before replacing it with the geographic position.
CATEGORIELLES = ("crop", "iso3")


def pipeline(modele, variables: list[str] | None = None) -> Pipeline:
    """One-hot encode the categorical variables, leave the rest as is.

    `variables` specifies the set used; by default, only `crop` is encoded — the case
    of the served model. The encoding is **inside** the pipeline, hence relearned on
    each validation fold without seeing the evaluated fold.
    """
    presentes = [v for v in (variables or ["crop"]) if v in CATEGORIELLES]
    preparation = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), presentes)],
        remainder="passthrough",
    )
    return Pipeline([("preparation", preparation), ("modele", modele)])


#: Settings of the served model, **validated by search** (notebook 09, section 5) and not
#: borrowed from elsewhere. 36 configurations sampled over three families in temporal
#: validation:
#:
#: * the best one found gains 0.0056 t/ha, i.e. **26 times less** than the variation from
#:   one validation year to the next (standard deviation 0.147 t/ha) — indistinguishable
#:   from noise;
#: * it only wins on 3 folds out of 5, below the chosen criterion;
#: * and above all it exceeds the 50 MB limit, because `min_samples_split=2` lets the trees
#:   grow deeper: 100.7 MB at 300 trees, 50.4 MB at 150.
#:
#: `min_samples_split=3` acts as a light regularisation that **also shortens the trees**:
#: hence both a smaller size and an equivalent performance. The delivery constraint and
#: performance point the same way here.
REGLAGES = {
    "max_features": 0.9,
    "min_samples_split": 3,
}


def modele_retenu(n_arbres: int = 150) -> Pipeline:
    """The served model: ExtraTrees with the :data:`REGLAGES` settings.

    150 trees is the largest model that fits under the 50 MB limit; RMSE plateaus from
    150 anyway (notebook 10).

    `n_jobs=1` on purpose: the machine only has 4 GB free, and a forest parallelised
    over 12 cores duplicates the data that many times.
    """
    return pipeline(
        ExtraTreesRegressor(
            n_estimators=n_arbres,
            random_state=config.SEED,
            n_jobs=1,
            **REGLAGES,
        )
    )


@dataclass
class Resultat:
    """What a full training run produces.

    `classement` holds one set of ranking metrics per view (see :data:`VUES`).
    """

    modele: Pipeline
    scores: dict[str, float]
    classement: dict[str, dict[str, float]]
    n_apprentissage: int
    n_test: int


def entraine(df: pd.DataFrame | None = None, n_arbres: int = 150, verbeux: bool = True) -> Resultat:
    """Train on the years before 2013, evaluate once on 2013."""
    df = charge() if df is None else df
    X_app, X_test, y_app, y_test = protocole.decoupage_temporel(
        df, VARIABLES, features.CIBLE_RECOMMEND
    )

    modele = modele_retenu(n_arbres).fit(X_app, y_app)
    predictions = modele.predict(X_test)

    scores = evaluation.metriques(y_test, predictions, "test")
    test = df[df["year"] == config.ANNEE_TEST]
    prix = prix_fao.charge() if config.PRIX_PRODUCTEUR_CSV.exists() else None
    classement = qualite_par_vue(test, pd.Series(predictions, index=test.index), prix)

    if verbeux:
        print(f"apprentissage : {len(X_app):,} lignes ({df.year.min()}-{config.ANNEE_TEST - 1})")
        print(f"test          : {len(X_test):,} lignes ({config.ANNEE_TEST})")
        print()
        print(f"RMSE {scores['test_rmse']:.4f} t/ha | MAE {scores['test_mae']:.4f} "
              f"| R2 {scores['test_r2']:.4f}")
        print()
        print(tableau_vues(classement).round(3).to_string())

    return Resultat(modele, scores, classement, len(X_app), len(X_test))


# --------------------------------------------------------------------------- #
# Ranking quality, measured on what the user actually sees
# --------------------------------------------------------------------------- #

#: Ranking views. The raw view is the one notebook 09 measured; the others are those
#: the service can serve. Measuring only the raw view would grade a ranking nobody
#: sees: the served order is in dry matter, or in gross product.
VUES = {
    "brut": "10 cultures FAO, rendement brut (t/ha)",
    "matiere_seche": "10 cultures FAO, matière sèche (t/ha)",
    "matiere_seche_produit": "4 cultures de l'application, matière sèche (t/ha)",
    "produit_brut": "10 cultures FAO, produit brut (USD/ha)",
    "produit_brut_produit": "4 cultures de l'application, produit brut (USD/ha)",
}


def qualite_par_vue(
    observe: pd.DataFrame, predit: pd.Series, prix: pd.DataFrame | None = None,
) -> dict[str, dict[str, float]]:
    """Ranking metrics for each view of :data:`VUES`.

    The same conversion is applied to observed and predicted yields: dry matter share,
    or reference price. Prices are those known at the row's year (no later price).
    The gross product views are skipped when `prix` is None.

    The `regret` of a view is in that view's unit: t/ha of dry matter, or USD/ha.
    """
    cible = features.CIBLE_RECOMMEND
    ms = observe["crop"].map(features.MATIERE_SECHE)
    produit = observe["crop"].isin(PRODUIT_VERS_FAO.values()).to_numpy()

    conversions = {"brut": 1.0, "matiere_seche": ms}
    if prix is not None:
        conversions["produit_brut"] = prix_fao.prix_par_ligne(prix, observe)

    resultats = {}
    for vue, facteur in conversions.items():
        converti = observe.assign(**{cible: observe[cible] * facteur})
        predit_converti = predit * facteur
        resultats[vue] = evaluation.qualite_classement(converti, predit_converti)
        if f"{vue}_produit" in VUES:
            resultats[f"{vue}_produit"] = evaluation.qualite_classement(
                converti[produit], predit_converti[produit]
            )
    return resultats


def tableau_vues(qualite: dict[str, dict[str, float]]) -> pd.DataFrame:
    """The per-view metrics as a table, one row per view."""
    table = pd.DataFrame.from_dict(qualite, orient="index")
    table.index = [VUES[v] for v in table.index]
    return table


def metriques_mlflow(qualite: dict[str, dict[str, float]]) -> dict[str, float]:
    """Flatten the per-view metrics into MLflow metric names."""
    return {
        f"classement_{vue}_{nom}": valeur
        for vue, mesures in qualite.items()
        for nom, valeur in mesures.items()
    }


# --------------------------------------------------------------------------- #
# Served ranking
# --------------------------------------------------------------------------- #

#: Reverse mapping: product crop -> FAO label, for the 4 shared crops.
PRODUIT_VERS_FAO = {k: v for k, v in CROP_TO_FAO_ITEM.items() if v}

#: Product crops missing from the FAO dataset are **left out of the ranking**. Step 1
#: gave them a donor crop to enrich the consolidated dataset, but in a ranking a donor
#: does not hold: barley would always tie with wheat, whose prediction it copies, and
#: cotton would carry the yield of sorghum, which step 1 itself rules out as a proxy.
#: Both remain available through `/predict`.
CULTURES_EXCLUES = {
    culture: "absente du jeu FAO : son classement copierait celui de sa culture donneuse"
    for culture, item in CROP_TO_FAO_ITEM.items()
    if item is None
}


def contexte_pays(df: pd.DataFrame, iso3: str, annee: int | None = None) -> dict:
    """Latest known conditions of a country, to prefill in the interface."""
    du_pays = df[df["iso3"] == iso3]
    if du_pays.empty:
        raise KeyError(
            f"{iso3} n'est pas un pays servi : aucune ligne exploitable (voir les pays "
            "retires faute de temperature, de pesticides ou de pluviometrie fiable)"
        )
    annee = int(du_pays["year"].max()) if annee is None else annee
    lignes = du_pays[du_pays["year"] == annee]
    if lignes.empty:
        raise KeyError(f"aucune donnee pour {iso3} en {annee}")
    premiere = lignes.iloc[0]
    return {
        "iso3": iso3,
        "year": int(annee),
        "rain_mm": float(premiere["rain_mm"]),
        "temp_hist": float(premiere["temp_hist"]),
        "log_pest_hist": float(premiere["log_pest_hist"]),
        "lat_abs": float(premiere["lat_abs"]),
        "geo_x": float(premiere["geo_x"]),
        "geo_y": float(premiere["geo_y"]),
        "geo_z": float(premiere["geo_z"]),
    }


#: Columns the ranking can be sorted by.
CRITERES = ("matiere_seche_t_ha", "produit_brut_usd_ha", "rendement_brut_t_ha")


def predit_cultures(modele: Pipeline, contexte: dict, cultures: dict[str, str]) -> pd.DataFrame:
    """Predicted yield of each crop in a context, raw and in dry matter, with its group.

    `cultures` maps the displayed name to the FAO label the model knows. No price, no
    sort: the two building blocks the service assembles (see :func:`classement`).
    """
    lignes = pd.DataFrame(
        [{**contexte, "crop": fao} for fao in cultures.values()]
    )[["crop"] + [v for v in VARIABLES if v != "crop"]]
    predictions = modele.predict(lignes)

    table = pd.DataFrame(
        {
            "culture": list(cultures),
            "culture_fao": list(cultures.values()),
            "rendement_brut_t_ha": predictions,
        }
    )
    table["matiere_seche_t_ha"] = [
        features.en_matiere_seche(r, fao)
        for r, fao in zip(table["rendement_brut_t_ha"], table["culture_fao"])
    ]
    table["groupe"] = table["culture_fao"].map(features.GROUPE_CULTURE)
    return table


def ajoute_produit_brut(table: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Add price, price source and gross product, from an already resolved price table.

    `reference` is indexed by FAO label, with `prix_usd_t` and `prix_source` — as
    returned by :func:`src.data.prix.prix_reference`, or built by the API from the
    user's own prices.
    """
    manquantes = sorted(set(table["culture_fao"]) - set(reference.index))
    if manquantes:
        raise KeyError(f"aucun prix pour : {manquantes}")
    table = table.copy()
    table["prix_usd_t"] = table["culture_fao"].map(reference["prix_usd_t"])
    table["prix_source"] = table["culture_fao"].map(reference["prix_source"])
    table["produit_brut_usd_ha"] = table["rendement_brut_t_ha"] * table["prix_usd_t"]
    return table


def classement(
    modele: Pipeline, contexte: dict, cultures_observees: set[str] | None = None,
    cultures_produit: bool = True, prix: pd.DataFrame | None = None,
    critere: str = "matiere_seche_t_ha",
) -> pd.DataFrame:
    """Rank crops by expected yield, raw, in dry matter and, if prices are given, in
    gross product.

    Four precautions, which change what the user reads:

    * **dry matter** — a ranking in raw tonnes always puts a tuber first, most of the
      gap being water;
    * **agronomic group** — a potato is not pitted directly against wheat;
    * **`observee`** — a crop never grown in the country is flagged: validation barely
      measures this case, and the model extrapolates there;
    * **`prix_source`** — the gross product uses the country's own price when it
      publishes one, the world median otherwise.

    `cultures_produit=True` restricts to the application's crops present in the FAO
    dataset; barley and cotton are left out (see :data:`CULTURES_EXCLUES`).
    `critere` picks the sort column, among :data:`CRITERES`.
    """
    if critere not in CRITERES:
        raise ValueError(f"critere inconnu : {critere!r}, attendu parmi {CRITERES}")
    if critere == "produit_brut_usd_ha" and prix is None:
        raise ValueError("le classement en produit brut demande les prix")

    if cultures_produit:
        demandes = dict(PRODUIT_VERS_FAO)
    else:
        demandes = {c: c for c in features.RECOMMEND_CULTURES}

    table = predit_cultures(modele, contexte, demandes)
    if prix is not None:
        reference = prix_fao.prix_reference(
            prix, contexte["iso3"], contexte["year"], table["culture_fao"]
        )
        table = ajoute_produit_brut(table, reference)
    if cultures_observees is not None:
        table["observee"] = table["culture_fao"].isin(cultures_observees)
    return table.sort_values(critere, ascending=False).reset_index(drop=True)


def cultures_du_pays(df: pd.DataFrame, iso3: str) -> set[str]:
    """Crops actually observed in the country, to flag extrapolations."""
    return set(df.loc[df["iso3"] == iso3, "crop"])


# --------------------------------------------------------------------------- #
# Size of the served model
# --------------------------------------------------------------------------- #

def taille_compressee(modele: Pipeline) -> float:
    """Model size in MB, lzma-compressed."""
    return len(lzma.compress(pickle.dumps(modele), preset=6)) / 1024**2


def sauvegarde(resultat: Resultat, df: pd.DataFrame) -> None:
    """Save the lzma-compressed model, and its metadata.

    `lzma` rather than joblib's `zlib`: a forest compresses much better, which keeps
    the file under GitHub's 50 MB per-file limit.
    """
    config.MODELES.mkdir(parents=True, exist_ok=True)
    with lzma.open(config.MODELE_RECOMMEND, "wb", preset=6) as f:
        pickle.dump(resultat.modele, f)

    metadonnees = {
        "service": "recommend",
        "modele": "ExtraTreesRegressor",
        "n_estimators": resultat.modele.named_steps["modele"].n_estimators,
        "reglages": REGLAGES,
        "reglages_valides_par": (
            "recherche de 36 configurations sur 3 familles en validation temporelle "
            "(notebook 09) : le meilleur gain, 0,0056 t/ha, est 26 fois plus petit que la "
            "variation entre annees de validation, ne tient que sur 3 plis sur 5, et "
            "depasse la limite de 50 Mo"
        ),
        "variables": VARIABLES,
        "cible": features.CIBLE_RECOMMEND,
        "unite": "t/ha",
        "seed": config.SEED,
        "protocole": {
            "type": "temporel",
            "annees_validation": ANNEES_VALIDATION,
            "annee_test": config.ANNEE_TEST,
        },
        "n_apprentissage": resultat.n_apprentissage,
        "n_test": resultat.n_test,
        "scores": resultat.scores,
        "qualite_classement": {
            vue: {"description": VUES[vue], **mesures}
            for vue, mesures in resultat.classement.items()
        },
        "cultures_fao": features.RECOMMEND_CULTURES,
        "cultures_produit_classees": PRODUIT_VERS_FAO,
        "cultures_exclues_du_classement": CULTURES_EXCLUES,
        "prix": {
            "source": config.SOURCE_PRIX_URL,
            "element": "Producer Price (USD/tonne)",
            "regle": (
                f"mediane des {prix_fao.FENETRE_ANNEES} dernieres annees connues du pays, "
                "sinon mediane mondiale ; jamais un prix posterieur a l'annee du contexte"
            ),
        },
        "pays_servis": sorted(df["iso3"].unique()),
        "avertissements": [
            "Le classement est un ordre de grandeur a l'echelle du contexte climatique d'un "
            "pays, pas une estimation de parcelle.",
            "Le produit brut (rendement x prix producteur FAOSTAT) n'est pas une marge : les "
            "couts de production sont inconnus. Il ne designe pas la culture la plus "
            "rentable, seulement celle qui se vend le plus cher par hectare.",
            "Une culture jamais observee dans le pays est signalee : la validation ne mesure "
            "presque pas ce cas.",
        ],
    }
    chemin = config.MODELES / "recommend_model_metadata.json"
    chemin.write_text(json.dumps(metadonnees, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    df = charge()
    resultat = entraine(df)
    sauvegarde(resultat, df)
    taille = config.MODELE_RECOMMEND.stat().st_size / 1024**2
    print(f"\necrit {config.MODELE_RECOMMEND.relative_to(config.RACINE)} ({taille:.1f} Mo)")
    print("ecrit models/recommend_model_metadata.json")
    if taille > 50:
        raise AssertionError(f"modele servi de {taille:.1f} Mo : au-dela de la limite de 50 Mo")


if __name__ == "__main__":
    main()
