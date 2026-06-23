import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Prognoza deszczu w Australii",
    page_icon="🌦️",
)


@st.cache_resource
def load_artifacts():
    return joblib.load("models/aussie_rain.joblib")


artifacts = load_artifacts()

model = artifacts["model"]
input_cols = artifacts["input_cols"]
numeric_cols = artifacts["numeric_cols"]
categorical_cols = artifacts["categorical_cols"]
encoded_cols = artifacts["encoded_cols"]
imputer = artifacts["imputer"]
if not hasattr(imputer, "_fill_dtype"):
    imputer._fill_dtype = imputer.statistics_.dtype

scaler = artifacts["scaler"]
encoder = artifacts["encoder"]

st.title("🌦️ Prognoza deszczu na jutro")

st.markdown("""
Ta aplikacja wykorzystuje model **Random Forest**
do przewidywania, czy jutro będzie padać w Australii.
""")

with st.expander("ℹ️ Objaśnienie danych wejściowych"):
    st.markdown("""
- **Location** — miasto lub stacja meteorologiczna w Australii.
- **MinTemp / MaxTemp** — minimalna i maksymalna temperatura dzienna (°C).
- **Rainfall** — ilość opadów w milimetrach.
- **WindGustDir** — kierunek najsilniejszego podmuchu wiatru.
- **WindDir9am / WindDir3pm** — kierunek wiatru o 9:00 i 15:00.
- **Humidity9am / Humidity3pm** — wilgotność względna (%).
- **Pressure9am / Pressure3pm** — ciśnienie atmosferyczne (hPa).
- **RainToday** — czy dzisiaj padał deszcz.
""")

with st.expander("📖 Instrukcja obsługi krok po kroku"):
    st.markdown("""
### Jak korzystać z aplikacji

1. **Wybierz lokalizację** z listy dostępnych australijskich miast lub stacji meteorologicznych.

2. **Wprowadź dzisiejsze dane pogodowe**, w tym:
   - temperaturę minimalną i maksymalną;
   - ilość opadów;
   - kierunek i prędkość wiatru;
   - wilgotność;
   - ciśnienie atmosferyczne;
   - pozostałe wymagane parametry.

3. **Określ, czy dzisiaj padał deszcz** w polu **"Czy dzisiaj padał deszcz?"**:
   - **Tak** — jeśli dzisiaj wystąpiły opady;
   - **Nie** — jeśli dzisiaj nie było opadów.

4. **Sprawdź poprawność wprowadzonych danych.** Im dokładniejsze dane, tym bardziej wiarygodna prognoza modelu.

5. **Kliknij przycisk "Przewidź".**

6. **Sprawdź wynik**, który zawiera:
   - prognozę, czy jutro będzie padać (**Tak** lub **Nie**);
   - prawdopodobieństwo opadów w procentach.

> ⚠️ Uwaga: Prognoza jest generowana przez model uczenia maszynowego na podstawie historycznych danych meteorologicznych i ma charakter wyłącznie informacyjny. Nie jest oficjalną prognozą pogody.
""")

locations = [
    "Albury", "BadgerysCreek", "Cobar", "CoffsHarbour", "Moree",
    "Newcastle", "NorahHead", "NorfolkIsland", "Penrith", "Richmond",
    "Sydney", "SydneyAirport", "WaggaWagga", "Canberra", "Melbourne",
    "MelbourneAirport", "Hobart", "Brisbane", "Cairns", "GoldCoast",
    "Townsville", "Adelaide", "Perth", "PerthAirport", "Darwin",
    "AliceSprings", "MountGambier", "Ballarat", "Bendigo", "Launceston"
]

wind_dirs = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
]

user_input = {}

st.subheader("Wprowadź dane pogodowe")

user_input["Location"] = st.selectbox("📍 Lokalizacja", locations)

user_input["MinTemp"] = st.number_input("🌡️ Temperatura minimalna (°C)", min_value=-50.0, value=15.0)
user_input["MaxTemp"] = st.number_input("☀️ Temperatura maksymalna (°C)", min_value=-50.0, value=25.0)
user_input["Rainfall"] = st.number_input("🌧️ Opady deszczu (mm)", min_value=0.0, value=0.0)
user_input["Evaporation"] = st.number_input("Parowanie", min_value=0.0, value=5.0)
user_input["Sunshine"] = st.number_input("Nasłonecznienie", min_value=0.0, value=8.0)

user_input["WindGustDir"] = st.selectbox("Kierunek najsilniejszego podmuchu wiatru", wind_dirs)
user_input["WindGustSpeed"] = st.number_input("Prędkość najsilniejszego podmuchu wiatru (km/h)", min_value=0.0, value=35.0)

user_input["WindDir9am"] = st.selectbox("Kierunek wiatru o 9:00", wind_dirs)
user_input["WindDir3pm"] = st.selectbox("Kierunek wiatru o 15:00", wind_dirs)

user_input["WindSpeed9am"] = st.number_input("🍃 Prędkość wiatru o 9:00 (km/h)", min_value=0.0, value=15.0)
user_input["WindSpeed3pm"] = st.number_input("🍃 Prędkość wiatru o 15:00 (km/h)", min_value=0.0, value=20.0)

user_input["Humidity9am"] = st.number_input("💧 Wilgotność o 9:00 (%)", min_value=0.0, max_value=100.0, value=60.0)
user_input["Humidity3pm"] = st.number_input("💧 Wilgotność o 15:00 (%)", min_value=0.0, max_value=100.0, value=50.0)

user_input["Pressure9am"] = st.number_input("📈 Ciśnienie atmosferyczne o 9:00 (hPa)", min_value=0.0, value=1015.0)
user_input["Pressure3pm"] = st.number_input("📈 Ciśnienie atmosferyczne o 15:00 (hPa)", min_value=0.0, value=1013.0)

user_input["Cloud9am"] = st.number_input("Zachmurzenie o 9:00", min_value=0.0, max_value=9.0, value=4.0)
user_input["Cloud3pm"] = st.number_input("Zachmurzenie o 15:00", min_value=0.0, max_value=9.0, value=4.0)

user_input["Temp9am"] = st.number_input("Temperatura o 9:00 (°C)", min_value=-50.0, value=18.0)
user_input["Temp3pm"] = st.number_input("Temperatura o 15:00 (°C)", min_value=-50.0, value=23.0)

rain_today_label = st.selectbox("☔ Czy dzisiaj padał deszcz?", ["Nie", "Tak"])
user_input["RainToday"] = "Yes" if rain_today_label == "Tak" else "No"


def preprocess_input(user_input):
    input_df = pd.DataFrame([user_input])
    input_df = input_df[input_cols]

    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    encoded = encoder.transform(input_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=input_df.index)

    final_df = pd.concat(
        [input_df.drop(columns=categorical_cols), encoded_df],
        axis=1
    )

    return final_df


if st.button("Przewidź"):

    # =========================
    # Walidacja danych wejściowych
    # =========================

    if user_input["MinTemp"] >= user_input["MaxTemp"]:
        st.error("❌ Temperatura minimalna musi być niższa niż temperatura maksymalna.")
        st.stop()

    if not (user_input["MinTemp"] <= user_input["Temp9am"] <= user_input["MaxTemp"]):
        st.error("❌ Temperatura o 9:00 musi mieścić się między temperaturą minimalną a maksymalną.")
        st.stop()

    if not (user_input["MinTemp"] <= user_input["Temp3pm"] <= user_input["MaxTemp"]):
        st.error("❌ Temperatura o 15:00 musi mieścić się między temperaturą minimalną a maksymalną.")
        st.stop()

    if user_input["RainToday"] == "No" and user_input["Rainfall"] > 0:
        st.error("❌ Wybrano, że dzisiaj nie padało, ale ilość opadów jest większa niż 0 mm.")
        st.stop()

    if user_input["WindSpeed9am"] > user_input["WindGustSpeed"]:
        st.error("❌ Prędkość wiatru o 9:00 nie może przekraczać maksymalnej prędkości podmuchu.")
        st.stop()

    if user_input["WindSpeed3pm"] > user_input["WindGustSpeed"]:
        st.error("❌ Prędkość wiatru o 15:00 nie może przekraczać maksymalnej prędkości podmuchu.")
        st.stop()

    # =========================
    # Predykcja
    # =========================

    final_input = preprocess_input(user_input)

    prediction = model.predict(final_input)[0]
    probabilities = model.predict_proba(final_input)[0]

    yes_probability = probabilities[
        list(model.classes_).index("Yes")
    ]

    if prediction == "Yes":
        st.success("🌧️ Prognoza: jutro będzie padać")
    else:
        st.success("☀️ Prognoza: jutro nie będzie padać")

    st.info(f"Prawdopodobieństwo deszczu jutro: {yes_probability:.2%}")
