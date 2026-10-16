# Dictionnaire des donnees — `crop_dataset_consolidated.csv`

Genere par `src/data/build_dataset.py`. Origine : **A** = Agriculture CropYield,
**B** = CropYield Prediction reconstruit, **derivee** = calculee lors de la fusion.

| Colonne | Type | Unite | Origine | Description |
|---|---|---|---|---|
| `Region` | category | - | A | Zone abstraite (North/South/East/West), pas un pays |
| `Soil_Type` | category | - | A | Texture du sol : Sandy, Loam, Chalky, Silt, Clay, Peaty |
| `Crop` | category | - | A | Culture de la parcelle (6 especes) |
| `Rainfall_mm` | float64 | mm | A | Pluie de saison culturale (100-1000) |
| `Temperature_Celsius` | float64 | degC | A | Temperature moyenne de la parcelle sur la saison culturale (15-40) |
| `Fertilizer_Used` | bool | - | A | Fertilisation appliquee |
| `Irrigation_Used` | bool | - | A | Parcelle irriguee |
| `Weather_Condition` | category | - | A | Sunny, Rainy ou Cloudy — independante de la pluie et de la temperature mesurees |
| `Days_to_Harvest` | int16 | jours | A | Duree du cycle (60-150) ; sans lien avec le rendement |
| `Yield_tons_per_hectare` | float64 | t/ha | A | CIBLE — rendement observe |
| `crop_ref_yield_t_ha` | float64 | t/ha | B | Rendement FAO median de la culture dans le monde |
| `crop_ref_yield_p25` | float64 | t/ha | B | 1er quartile de la dispersion FAO |
| `crop_ref_yield_p75` | float64 | t/ha | B | 3e quartile de la dispersion FAO |
| `crop_pesticides_intensity` | float64 | tonnes | B | Pesticides medians des pays-annees cultivant l'espece ; proxy d'intensification |
| `crop_temp_niche` | float64 | degC | B | Temperature mediane ANNUELLE des pays-annees cultivant l'espece (niche thermique) |
| `crop_rain_pctl` | float64 | percentile | B | Pluie mediane de la culture, en percentile de la distribution de B (jamais en mm) |
| `crop_ref_source` | category | - | derivee | 'FAO' ou 'imputed' (orge, coton) |
| `temp_dev_from_niche` | float64 | degC | derivee | abs(Temperature_Celsius - crop_temp_niche) ; croise A et B, mais soustrait une
      temperature de saison culturale a une moyenne annuelle : heuristique, pas un
      ecart agronomique mesure |
