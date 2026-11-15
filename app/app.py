"""Agritech Answers — Streamlit interface.

The interface holds **no ML logic**: it builds forms from what the API describes, sends
what the user typed, and displays what the API answers. All calls go through
`api_client.py`.

Both screens start the same way — a country, then its conditions, prefilled and
editable — because both ask the same model: one crop's yield, or all crops ranked.

Run locally (API started first):  streamlit run app/app.py  (or `make app`)
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import api_client as api

st.set_page_config(page_title="Agritech Answers", page_icon="🌾", layout="wide")

#: Single-series bar colour (reference palette, slot 1).
BAR_COLOUR = "#2a78d6"

#: Vega expression formatting an axis tick the French way: "1 304", "2,5".
AXIS_LABEL_FR = (
    "replace(replace(format(datum.value, ',~f'), regexp(',', 'g'), ' '), '.', ',')"
)

#: Displayed names of the conditions, to tell the user which ones they changed.
CONDITION_LABELS = {
    "temperature_c": "température", "rain_mm": "pluie",
    "pesticides_t": "pesticides", "price_usd_t": "prix",
}


def number_fr(value: float, decimals: int = 2) -> str:
    """`1304.2` -> `1 304`, `9.228` -> `9,23`: French separators, narrow no-break space."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", " ").replace(".", ",")


CRITERIA = {
    "yield": ("Rendement prédit", "yield_t_ha", "t/ha"),
    "dry_matter": ("Matière sèche", "dry_matter_t_ha", "t/ha"),
    "gross_product": ("Produit brut", "gross_product_usd_ha", "USD/ha"),
}


# --------------------------------------------------------------------------- #
# Cached reads: the API describes the forms, the interface does not hard-code them.
# The API address is part of the cache key: an answer from one API must never be
# served while pointing at another.
# --------------------------------------------------------------------------- #

@st.cache_data(ttl=600, show_spinner=False)
def read_predict_context(url: str) -> dict:
    return api.predict_context()


@st.cache_data(ttl=600, show_spinner=False)
def read_countries(url: str) -> list[dict]:
    return api.countries()


@st.cache_data(ttl=600, show_spinner=False)
def read_country_context(url: str, iso3: str) -> dict:
    return api.country_context(iso3)


def stop_on_error(error: api.APIError) -> None:
    st.error(error.message, icon="🚫")
    st.stop()


def show_warnings(warnings: list[str]) -> None:
    for message in warnings:
        st.warning(message, icon="⚠️")


def show_entered(entered_values: list[str]) -> None:
    if entered_values:
        names = ", ".join(CONDITION_LABELS.get(v, v) for v in entered_values)
        st.caption(f"Valeurs modifiées par vous : {names}.")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def render_sidebar() -> str:
    with st.sidebar:
        st.title("🌾 Agritech Answers")
        mode = st.radio("Que voulez-vous faire ?", ["Prédiction", "Recommandation"],
                        captions=["Estimer le rendement d'une culture",
                                  "Classer les cultures pour vos conditions"])
        st.divider()
        try:
            state = api.health()
            st.caption(f"API connectée · v{state['api_version']}")
            st.caption(f"Modèle de rendement : {state['models']['yield']}")
        except api.APIError as error:
            st.caption(f"API indisponible — {error.message}")

        with st.expander("Ce que l'outil sait, et ne sait pas"):
            st.markdown(
                "- **Un seul modèle** estime le rendement, pour la prédiction comme pour la "
                "recommandation : les deux écrans donnent le même chiffre.\n"
                "- Il a appris sur des données **FAO réelles**, à l'échelle d'un **pays** "
                "(jusqu'en 2013), pas d'une parcelle : il donne un ordre de grandeur.\n"
                "- L'effet de l'**engrais** et de l'**irrigation** vient de parcelles "
                "**simulées** : il est affiché à part, à titre indicatif.\n"
                "- **Produit brut** = rendement × prix de vente. Ce n'est **pas une "
                "marge** : les coûts de production sont inconnus.\n"
                "- L'orge et le coton, absents des données FAO, ne sont ni estimés ni classés."
            )
    return mode


# --------------------------------------------------------------------------- #
# Shared by both screens: the country and its conditions
# --------------------------------------------------------------------------- #

def choose_country() -> tuple[str, dict]:
    """Country selector, and the country's default values to prefill the form."""
    try:
        country_list = read_countries(api.api_base_url())
    except api.APIError as error:
        stop_on_error(error)

    names = {p["iso3"]: p["name"] for p in country_list}
    iso3 = st.selectbox("Pays", list(names), format_func=names.get,
                        index=list(names).index("FRA") if "FRA" in names else 0)
    try:
        defaults = read_country_context(api.api_base_url(), iso3)
    except api.APIError as error:
        stop_on_error(error)
    st.caption(f"Valeurs préremplies : conditions de {defaults['name']} ({defaults['year']}). "
               "Modifiez-les pour décrire votre situation.")
    return iso3, defaults


def condition_inputs(defaults: dict, key) -> dict:
    """The three conditions, prefilled. `key` builds widget keys that carry the screen
    and the country, so choosing another country resets them."""
    left, right = st.columns(2)
    temperature = left.slider(
        "Température moyenne annuelle (°C)", min_value=-30.0, max_value=60.0,
        value=float(defaults["temperature_c"]), step=0.01, key=key("temperature"),
        help="Moyenne des dernières années connues du pays.",
    )
    rain = right.slider(
        "Pluie annuelle normale (mm)", min_value=0.0, max_value=5000.0,
        value=float(defaults["rain_mm"]), step=1.0, key=key("rain"),
    )
    pesticides = left.number_input(
        "Pesticides utilisés dans le pays (t/an)", min_value=0.0, max_value=5_000_000.0,
        value=float(defaults["pesticides_t"]), step=100.0, key=key("pesticides"),
        help="Tonnage national : c'est la seule mesure d'intensité agricole disponible.",
    )
    return {"temperature_c": temperature, "rain_mm": rain, "pesticides_t": pesticides}


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

def page_prediction() -> None:
    st.header("Estimer le rendement d'une culture")
    try:
        context = read_predict_context(api.api_base_url())
    except api.APIError as error:
        stop_on_error(error)
    crops = context["crops"]

    iso3, defaults = choose_country()
    key = lambda name: f"prediction_{name}_{iso3}"  # noqa: E731
    with st.form(f"prediction_{iso3}"):
        crop = st.selectbox("Culture", list(crops), format_func=crops.get, key=key("crop"))
        entered = condition_inputs(defaults, key)
        submitted = st.form_submit_button("Estimer le rendement", type="primary")

    unavailable = ", ".join(c["label"] for c in context["unavailable_crops"])
    if not submitted:
        st.caption(f"Non disponibles : {unavailable} — absentes des données FAO.")
        return
    try:
        with st.spinner("Estimation…"):
            response = api.predict(iso3, crop, api.changed_fields(entered, defaults))
    except api.APIError as error:
        stop_on_error(error)

    low, high = response["interval_95_t_ha"]
    st.metric(f"Rendement estimé — {response['label']}, {response['context']['country']}",
              f"{number_fr(response['yield_t_ha'])} t/ha")
    st.caption(f"Entre **{number_fr(low)}** et **{number_fr(high)} t/ha** dans 95 % des cas "
               "observés en validation. "
               f"En matière sèche : {number_fr(response['dry_matter_t_ha'])} t/ha.")
    show_entered(response["context"]["entered_values"])

    effects = " · ".join(
        f"{e['label'].lower()} **+{number_fr(e['effect_t_ha'])} t/ha**"
        for e in response["practice_effects"]
    )
    st.info(
        f"Effet des pratiques, mesuré sur des parcelles simulées : {effects}. "
        "Ordre de grandeur indicatif, non additionné à l'estimation ci-dessus.",
        icon="ℹ️",
    )
    show_warnings(response["warnings"])


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #

def ranking_chart(ranking: pd.DataFrame, criterion: str) -> alt.Chart:
    """Horizontal bars, sorted, values printed at the bar end."""
    title, column, unit = CRITERIA[criterion]
    frame = ranking.assign(
        label_text=ranking.apply(
            lambda row: row["label"] + ("" if row["grown_in_country"] else " (jamais cultivée ici)"),
            axis=1,
        ),
        value_text=ranking[column].map(lambda v: number_fr(v, 0 if unit == "USD/ha" else 2)),
        yield_text=ranking["yield_t_ha"].map(lambda v: f"{number_fr(v)} t/ha"),
        dry_text=ranking["dry_matter_t_ha"].map(lambda v: f"{number_fr(v)} t/ha"),
        product_text=ranking["gross_product_usd_ha"].map(lambda v: f"{number_fr(v, 0)} USD/ha"),
    )
    order = list(frame["label_text"])
    base = alt.Chart(frame).encode(
        y=alt.Y("label_text:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X(f"{column}:Q", title=f"{title} ({unit})",
                axis=alt.Axis(grid=True, gridWidth=1, tickCount=5, labelExpr=AXIS_LABEL_FR)),
        tooltip=[
            alt.Tooltip("label:N", title="Culture"),
            alt.Tooltip("yield_text:N", title="Rendement"),
            alt.Tooltip("dry_text:N", title="Matière sèche"),
            alt.Tooltip("product_text:N", title="Produit brut"),
            alt.Tooltip("price_source:N", title="Prix"),
        ],
    )
    bars = base.mark_bar(size=22, cornerRadiusEnd=4, color=BAR_COLOUR).encode(
        opacity=alt.condition("datum.grown_in_country", alt.value(1.0), alt.value(0.4)),
    )
    values = base.mark_text(align="left", dx=6, fontSize=13).encode(text="value_text:N")
    return (bars + values).properties(height=alt.Step(44))


def recommendation_form(defaults: dict, iso3: str) -> tuple[dict, str, bool]:
    """The form, prefilled with the country's values."""
    key = lambda name: f"recommendation_{name}_{iso3}"  # noqa: E731
    with st.form(f"recommendation_{iso3}"):
        entered = condition_inputs(defaults, key)
        criterion = st.radio(
            "Classer selon", list(CRITERIA), format_func=lambda c: CRITERIA[c][0],
            horizontal=True, key=key("criterion"),
        )
        with st.expander("Prix de vente (USD/t) — modifiables"):
            columns = st.columns(len(defaults["prices"]))
            prices = {}
            for column, row in zip(columns, defaults["prices"]):
                prices[row["crop"]] = column.number_input(
                    row["crop"], min_value=1.0, max_value=100_000.0,
                    value=float(row["price_usd_t"]), step=5.0, key=key(f"price_{row['crop']}"),
                    help=f"Référence FAOSTAT : {row['price_source']}.",
                )
        submitted = st.form_submit_button("Classer les cultures", type="primary")
    return {**entered, "price_usd_t": prices}, criterion, submitted


def page_recommendation() -> None:
    st.header("Quelle culture privilégier ?")
    iso3, defaults = choose_country()

    entered, criterion, submitted = recommendation_form(defaults, iso3)
    if not submitted:
        return
    try:
        with st.spinner("Classement…"):
            response = api.recommend(iso3, criterion, api.changed_fields(entered, defaults))
    except api.APIError as error:
        stop_on_error(error)

    ranking = pd.DataFrame(response["ranking"])
    title, column, unit = CRITERIA[criterion]
    leaders = ranking["label"].head(2).tolist()
    st.subheader(f"À privilégier : {leaders[0]}, puis {leaders[1]}")
    st.caption(
        f"Classement selon **{title.lower()}**. Les deux ou trois premières cultures sont à "
        "considérer ensemble : l'écart entre elles est souvent inférieur à l'incertitude."
    )

    st.altair_chart(ranking_chart(ranking, criterion), width="stretch")

    display_table = ranking[["rank", "label", "yield_t_ha", "dry_matter_t_ha", "price_usd_t",
                             "price_source", "gross_product_usd_ha", "grown_in_country"]].round(
        {"yield_t_ha": 2, "dry_matter_t_ha": 2, "price_usd_t": 0, "gross_product_usd_ha": 0})
    st.dataframe(
        display_table,
        hide_index=True, width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn("Rang", format="%d"),
            "label": "Culture",
            "yield_t_ha": st.column_config.NumberColumn("Rendement (t/ha)", format="localized"),
            "dry_matter_t_ha": st.column_config.NumberColumn("Matière sèche (t/ha)", format="localized"),
            "price_usd_t": st.column_config.NumberColumn("Prix (USD/t)", format="localized"),
            "price_source": "Origine du prix",
            "gross_product_usd_ha": st.column_config.NumberColumn("Produit brut (USD/ha)", format="localized"),
            "grown_in_country": st.column_config.CheckboxColumn("Cultivée dans le pays"),
        },
    )

    unranked = ", ".join(c["label"] for c in response["unranked_crops"])
    st.caption(f"Non classées : {unranked} — absentes des données FAO, leur rendement ne "
               "peut être ni estimé ni classé.")
    show_entered(response["context"]["entered_values"])
    show_warnings(response["warnings"])


# --------------------------------------------------------------------------- #

if render_sidebar() == "Prédiction":
    page_prediction()
else:
    page_recommendation()
