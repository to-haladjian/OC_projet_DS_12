"""Agritech Answers — Streamlit interface (step 4).

The interface holds **no ML logic**: it builds forms from what the API describes, sends
what the user typed, and displays what the API answers. All calls go through
`api_client.py`.

Run locally (API started first):  streamlit run app/app.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import api_client as api

st.set_page_config(page_title="Agritech Answers", page_icon="🌾", layout="wide")

#: Single-series bar colour (reference palette, slot 1).
COULEUR_BARRES = "#2a78d6"

#: Vega expression formatting an axis tick the French way: "1 304", "2,5".
ETIQUETTE_AXE_FR = (
    "replace(replace(format(datum.value, ',~f'), regexp(',', 'g'), '\u202f'), '.', ',')"
)


def nombre_fr(valeur: float, decimales: int = 2) -> str:
    """`1304.2` -> `1 304`, `9.228` -> `9,23`: French separators, narrow no-break space."""
    texte = f"{valeur:,.{decimales}f}"
    return texte.replace(",", "\u202f").replace(".", ",")


CRITERES = {
    "rendement": ("Rendement prédit", "rendement_t_ha", "t/ha"),
    "matiere_seche": ("Matière sèche", "matiere_seche_t_ha", "t/ha"),
    "produit_brut": ("Produit brut", "produit_brut_usd_ha", "USD/ha"),
}


# --------------------------------------------------------------------------- #
# Cached reads: the API describes the forms, the interface does not hard-code them.
# The API address is part of the cache key: an answer from one API must never be
# served while pointing at another.
# --------------------------------------------------------------------------- #

@st.cache_data(ttl=600, show_spinner=False)
def lit_contexte_predict(url: str) -> dict:
    return api.contexte_predict()


@st.cache_data(ttl=600, show_spinner=False)
def lit_pays(url: str) -> list[dict]:
    return api.pays()


@st.cache_data(ttl=600, show_spinner=False)
def lit_contexte_pays(url: str, iso3: str) -> dict:
    return api.contexte_pays(iso3)


def arrete_sur_erreur(erreur: api.ErreurAPI) -> None:
    st.error(erreur.message, icon="🚫")
    st.stop()


def affiche_avertissements(avertissements: list[str]) -> None:
    for message in avertissements:
        st.warning(message, icon="⚠️")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def barre_laterale() -> str:
    with st.sidebar:
        st.title("🌾 Agritech Answers")
        mode = st.radio("Que voulez-vous faire ?", ["Prédiction", "Recommandation"],
                        captions=["Estimer le rendement d'une parcelle",
                                  "Classer les cultures pour un pays"])
        st.divider()
        try:
            etat = api.sante()
            st.caption(f"API connectée · v{etat['version_api']}")
            st.caption(f"Modèles : {etat['modeles']['predict']} · {etat['modeles']['recommend']}")
        except api.ErreurAPI as erreur:
            st.caption(f"API indisponible — {erreur.message}")

        with st.expander("Ce que l'outil sait, et ne sait pas"):
            st.markdown(
                "- **Prédiction** : appris sur des parcelles **simulées**. Les effets "
                "(engrais, irrigation, pluie) sont des ordres de grandeur, et la culture "
                "choisie ne change pas l'estimation.\n"
                "- **Recommandation** : appris sur des données **FAO réelles**, mais à "
                "l'échelle d'un **pays** (données jusqu'en 2013), pas d'une parcelle.\n"
                "- **Produit brut** = rendement × prix de vente. Ce n'est **pas une "
                "marge** : les coûts de production sont inconnus.\n"
                "- L'orge et le coton, absents des données FAO, ne sont pas classés."
            )
    return mode


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

def page_prediction() -> None:
    st.header("Estimer le rendement d'une parcelle")
    try:
        contexte = lit_contexte_predict(api.url_api())
    except api.ErreurAPI as erreur:
        arrete_sur_erreur(erreur)

    physiques, domaine = contexte["bornes_physiques"], contexte["domaine_entrainement"]
    cultures = contexte["cultures"]

    with st.form("prediction"):
        culture = st.selectbox("Culture", list(cultures), format_func=cultures.get)
        gauche, droite = st.columns(2)
        pluie = gauche.slider(
            "Pluie de la saison (mm)", min_value=float(physiques["pluie_mm"]["min"]),
            max_value=float(physiques["pluie_mm"]["max"]), value=600.0, step=10.0,
            help=f"Le modèle a appris entre {domaine['pluie_mm']['min']:.0f} et "
                 f"{domaine['pluie_mm']['max']:.0f} mm : au-delà, l'estimation est extrapolée.",
        )
        temperature = droite.slider(
            "Température moyenne (°C)", min_value=float(physiques["temperature_c"]["min"]),
            max_value=float(physiques["temperature_c"]["max"]), value=25.0, step=0.5,
            help=f"Le modèle a appris entre {domaine['temperature_c']['min']:.0f} et "
                 f"{domaine['temperature_c']['max']:.0f} °C.",
        )
        engrais = gauche.toggle("Engrais utilisé", value=True)
        irrigation = droite.toggle("Parcelle irriguée", value=False)
        envoye = st.form_submit_button("Estimer le rendement", type="primary")

    if not envoye:
        st.caption("Renseignez la parcelle, puis lancez l'estimation.")
        return
    try:
        with st.spinner("Estimation…"):
            reponse = api.predire(culture, pluie, temperature, engrais, irrigation)
    except api.ErreurAPI as erreur:
        arrete_sur_erreur(erreur)

    bas, haut = reponse["intervalle_95_t_ha"]
    st.metric(f"Rendement estimé — {reponse['libelle']}", f"{nombre_fr(reponse['rendement_t_ha'])} t/ha")
    st.caption(f"95 % des parcelles comparables : entre **{nombre_fr(bas)}** et "
               f"**{nombre_fr(haut)} t/ha**.")
    culture_sans_effet, *autres = reponse["avertissements"]
    st.info(culture_sans_effet, icon="ℹ️")
    affiche_avertissements(autres)


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #

def graphique_classement(classement: pd.DataFrame, critere: str) -> alt.Chart:
    """Horizontal bars, sorted, values printed at the bar end."""
    titre, colonne, unite = CRITERES[critere]
    donnees = classement.assign(
        etiquette=classement.apply(
            lambda l: l["libelle"] + ("" if l["cultivee_dans_le_pays"] else " (jamais cultivée ici)"),
            axis=1,
        ),
        valeur_texte=classement[colonne].map(lambda v: nombre_fr(v, 0 if unite == "USD/ha" else 2)),
        rendement_texte=classement["rendement_t_ha"].map(lambda v: f"{nombre_fr(v)} t/ha"),
        seche_texte=classement["matiere_seche_t_ha"].map(lambda v: f"{nombre_fr(v)} t/ha"),
        produit_texte=classement["produit_brut_usd_ha"].map(lambda v: f"{nombre_fr(v, 0)} USD/ha"),
    )
    ordre = list(donnees["etiquette"])
    base = alt.Chart(donnees).encode(
        y=alt.Y("etiquette:N", sort=ordre, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X(f"{colonne}:Q", title=f"{titre} ({unite})",
                axis=alt.Axis(grid=True, gridWidth=1, tickCount=5, labelExpr=ETIQUETTE_AXE_FR)),
        tooltip=[
            alt.Tooltip("libelle:N", title="Culture"),
            alt.Tooltip("rendement_texte:N", title="Rendement"),
            alt.Tooltip("seche_texte:N", title="Matière sèche"),
            alt.Tooltip("produit_texte:N", title="Produit brut"),
            alt.Tooltip("prix_source:N", title="Prix"),
        ],
    )
    barres = base.mark_bar(size=22, cornerRadiusEnd=4, color=COULEUR_BARRES).encode(
        opacity=alt.condition("datum.cultivee_dans_le_pays", alt.value(1.0), alt.value(0.4)),
    )
    valeurs = base.mark_text(align="left", dx=6, fontSize=13).encode(text="valeur_texte:N")
    return (barres + valeurs).properties(height=alt.Step(44))


def formulaire_recommandation(defauts: dict, pays_iso3: str) -> tuple[dict, str, bool]:
    """The form, prefilled with the country's values. Widget keys carry the country, so
    choosing another country resets them."""
    cle = lambda nom: f"{nom}_{pays_iso3}"  # noqa: E731
    with st.form(f"recommandation_{pays_iso3}"):
        gauche, droite = st.columns(2)
        temperature = gauche.slider(
            "Température moyenne annuelle (°C)", min_value=-30.0, max_value=60.0,
            value=float(defauts["temperature_c"]), step=0.01, key=cle("temperature"),
            help="Moyenne des dernières années connues du pays.",
        )
        pluie = droite.slider(
            "Pluie annuelle normale (mm)", min_value=0.0, max_value=5000.0,
            value=float(defauts["pluie_mm"]), step=1.0, key=cle("pluie"),
        )
        pesticides = gauche.number_input(
            "Pesticides utilisés dans le pays (t/an)", min_value=0.0, max_value=5_000_000.0,
            value=float(defauts["pesticides_t"]), step=100.0, key=cle("pesticides"),
            help="Tonnage national : c'est la seule mesure d'intensité agricole disponible.",
        )
        critere = droite.radio(
            "Classer selon", list(CRITERES), format_func=lambda c: CRITERES[c][0],
            horizontal=True, key=cle("critere"),
        )
        with st.expander("Prix de vente (USD/t) — modifiables"):
            colonnes = st.columns(len(defauts["prix"]))
            prix = {}
            for colonne, ligne in zip(colonnes, defauts["prix"]):
                prix[ligne["culture"]] = colonne.number_input(
                    ligne["culture"], min_value=1.0, max_value=100_000.0,
                    value=float(ligne["prix_usd_t"]), step=5.0, key=cle(f"prix_{ligne['culture']}"),
                    help=f"Référence FAOSTAT : {ligne['prix_source']}.",
                )
        envoye = st.form_submit_button("Classer les cultures", type="primary")
    saisie = {"temperature_c": temperature, "pluie_mm": pluie, "pesticides_t": pesticides,
              "prix_usd_t": prix}
    return saisie, critere, envoye


def page_recommandation() -> None:
    st.header("Quelle culture privilégier ?")
    try:
        liste_pays = lit_pays(api.url_api())
    except api.ErreurAPI as erreur:
        arrete_sur_erreur(erreur)

    noms = {p["iso3"]: p["nom"] for p in liste_pays}
    iso3 = st.selectbox("Pays", list(noms), format_func=noms.get,
                        index=list(noms).index("FRA") if "FRA" in noms else 0)
    try:
        defauts = lit_contexte_pays(api.url_api(), iso3)
    except api.ErreurAPI as erreur:
        arrete_sur_erreur(erreur)
    st.caption(f"Valeurs préremplies : conditions de {defauts['nom']} ({defauts['annee']}). "
               "Modifiez-les pour décrire votre situation.")

    saisie, critere, envoye = formulaire_recommandation(defauts, iso3)
    if not envoye:
        return
    try:
        with st.spinner("Classement…"):
            reponse = api.recommander(iso3, critere, api.champs_modifies(saisie, defauts))
    except api.ErreurAPI as erreur:
        arrete_sur_erreur(erreur)

    classement = pd.DataFrame(reponse["classement"])
    titre, colonne, unite = CRITERES[critere]
    tete = classement["libelle"].head(2).tolist()
    st.subheader(f"À privilégier : {tete[0]}, puis {tete[1]}")
    st.caption(
        f"Classement selon **{titre.lower()}**. Les deux ou trois premières cultures sont à "
        "considérer ensemble : l'écart entre elles est souvent inférieur à l'incertitude."
    )

    st.altair_chart(graphique_classement(classement, critere), width="stretch")

    tableau = classement[["rang", "libelle", "rendement_t_ha", "matiere_seche_t_ha", "prix_usd_t",
                          "prix_source", "produit_brut_usd_ha", "cultivee_dans_le_pays"]].round(
        {"rendement_t_ha": 2, "matiere_seche_t_ha": 2, "prix_usd_t": 0, "produit_brut_usd_ha": 0})
    st.dataframe(
        tableau,
        hide_index=True, width="stretch",
        column_config={
            "rang": st.column_config.NumberColumn("Rang", format="%d"),
            "libelle": "Culture",
            "rendement_t_ha": st.column_config.NumberColumn("Rendement (t/ha)", format="localized"),
            "matiere_seche_t_ha": st.column_config.NumberColumn("Matière sèche (t/ha)", format="localized"),
            "prix_usd_t": st.column_config.NumberColumn("Prix (USD/t)", format="localized"),
            "prix_source": "Origine du prix",
            "produit_brut_usd_ha": st.column_config.NumberColumn("Produit brut (USD/ha)", format="localized"),
            "cultivee_dans_le_pays": st.column_config.CheckboxColumn("Cultivée dans le pays"),
        },
    )

    non_classees = ", ".join(c["libelle"] for c in reponse["cultures_non_classees"])
    st.caption(f"Non classées : {non_classees} — absentes des données FAO, leur rang "
               "ne serait qu'une copie de celui d'une autre culture.")
    saisies = reponse["contexte"]["valeurs_saisies"]
    if saisies:
        st.caption("Valeurs modifiées par vous : " + ", ".join(saisies) + ".")
    affiche_avertissements(reponse["avertissements"])


# --------------------------------------------------------------------------- #

if barre_laterale() == "Prédiction":
    page_prediction()
else:
    page_recommandation()
