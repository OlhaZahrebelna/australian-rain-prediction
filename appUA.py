import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Прогноз дощу в Австралії",
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

st.title("🌦️ Прогноз дощу на завтра")

st.markdown("""
Цей застосунок використовує модель **Random Forest**
для прогнозування того, чи буде дощ завтра в Австралії.
""")

with st.expander("ℹ️ Пояснення щодо введення даних"):
    st.markdown("""
- **Location** — місто або метеостанція в Австралії.
- **MinTemp / MaxTemp** — мінімальна та максимальна температура за день (°C).
- **Rainfall** — кількість опадів у міліметрах.
- **WindGustDir** — напрямок найсильнішого пориву вітру.
- **WindDir9am / WindDir3pm** — напрямок вітру о 9:00 та 15:00.
- **Humidity9am / Humidity3pm** — відносна вологість (%).
- **Pressure9am / Pressure3pm** — атмосферний тиск (гПа).
- **RainToday** — чи спостерігався дощ сьогодні.
""")

    
with st.expander("📖 Покрокова інструкція користування"):
    st.markdown("""
### Як скористатися застосунком

1. **Оберіть місто (Location)** зі списку доступних австралійських міст або метеостанцій.

2. **Введіть погодні показники за поточний день**, зокрема:
   - мінімальну та максимальну температуру;
   - кількість опадів;
   - напрямок і швидкість вітру;
   - вологість повітря;
   - атмосферний тиск;
   - інші необхідні параметри.

3. **Вкажіть, чи був сьогодні дощ** у полі **"RainToday"**:
   - **Yes** — якщо сьогодні були опади;
   - **No** — якщо дощу не було.

4. **Перевірте правильність введених даних.** Чим точніші показники, тим надійнішим буде прогноз моделі.

5. **Натисніть кнопку "Спрогнозувати".**

6. **Ознайомтеся з результатом**, який містить:
   - прогноз щодо наявності дощу завтра (**Yes** або **No**);
   - ймовірність прогнозу у відсотках.

> ⚠️ Зверніть увагу: прогноз генерується моделлю машинного навчання на основі історичних метеорологічних даних і має рекомендаційний характер. Він не є офіційним прогнозом погоди.
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

st.subheader("Введіть погодні дані")

user_input["Location"] = st.selectbox("📍 Місто", locations)

user_input["MinTemp"] = st.number_input("🌡️ Мінімальна температура (°C)", min_value=-50.0, value=15.0)
user_input["MaxTemp"] = st.number_input("☀️ Максимальна температура (°C)", min_value=-50.0, value=25.0)
user_input["Rainfall"] = st.number_input("🌧️ Кількість опадів (мм)", min_value=0.0, value=0.0)
user_input["Evaporation"] = st.number_input("Evaporation", min_value=0.0, value=5.0)
user_input["Sunshine"] = st.number_input("Sunshine", min_value=0.0, value=8.0)

user_input["WindGustDir"] = st.selectbox("Напрямок найсильнішого пориву вітру", wind_dirs)
user_input["WindGustSpeed"] = st.number_input("Швидкість найсильнішого пориву вітру (км/год)", min_value=0.0, value=35.0)

user_input["WindDir9am"] = st.selectbox("Напрямок вітру о 9:00", wind_dirs)
user_input["WindDir3pm"] = st.selectbox("Напрямок вітру о 15:00", wind_dirs)

user_input["WindSpeed9am"] = st.number_input("🍃 Швидкість вітру о 9:00 (км/год)", min_value=0.0, value=15.0)
user_input["WindSpeed3pm"] = st.number_input("🍃 Швидкість вітру о 15:00 (км/год)", min_value=0.0, value=20.0)

user_input["Humidity9am"] = st.number_input("💧 Вологість о 9:00 (%)", min_value=0.0, max_value=100.0, value=60.0)
user_input["Humidity3pm"] = st.number_input("💧 Вологість о 15:00 (%)", min_value=0.0, max_value=100.0, value=50.0)

user_input["Pressure9am"] = st.number_input("📈 Атмосферний тиск о 9:00 (гПа)", min_value=0.0, value=1015.0)
user_input["Pressure3pm"] = st.number_input("📈 Атмосферний тиск о 15:00 (гПа)", min_value=0.0, value=1013.0)

user_input["Cloud9am"] = st.number_input("Хмарність о 9:00", min_value=0.0, max_value=9.0, value=4.0)
user_input["Cloud3pm"] = st.number_input("Хмарність о 15:00", min_value=0.0, max_value=9.0, value=4.0)

user_input["Temp9am"] = st.number_input("Температура о 9:00 (°C)", min_value=-50.0, value=18.0)
user_input["Temp3pm"] = st.number_input("Температура о 15:00 (°C)", min_value=-50.0, value=23.0)

user_input["RainToday"] = st.selectbox("☔ Чи був дощ сьогодні?", ["No", "Yes"])

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


    # Якщо всі критичні перевірки пройдені
    final_input = preprocess_input(user_input)

    prediction = model.predict(final_input)[0]
    probabilities = model.predict_proba(final_input)[0]

    yes_probability = probabilities[list(model.classes_).index("Yes")]

    if prediction == "Yes":
        st.success("🌧️ Прогноз: завтра буде дощ")
    else:
        st.success("☀️ Прогноз: завтра дощу не буде")

    st.info(f"Ймовірність дощу завтра: {yes_probability:.2%}")
    
if st.button("Спрогнозувати"):

    # =========================
    # Перевірка введених даних
    # =========================

    # Мінімальна температура повинна бути меншою за максимальну
    if user_input["MinTemp"] >= user_input["MaxTemp"]:
        st.error("❌ Мінімальна температура повинна бути меншою за максимальну.")
        st.stop()

    # Температура о 9:00 повинна бути в межах MinTemp і MaxTemp
    if not (user_input["MinTemp"] <= user_input["Temp9am"] <= user_input["MaxTemp"]):
        st.error("❌ Температура о 9:00 повинна бути між мінімальною та максимальною.")
        st.stop()

    # Температура о 15:00 повинна бути в межах MinTemp і MaxTemp
    if not (user_input["MinTemp"] <= user_input["Temp3pm"] <= user_input["MaxTemp"]):
        st.error("❌ Температура о 15:00 повинна бути між мінімальною та максимальною.")
        st.stop()

    # Якщо дощу не було, то опади мають бути 0
    if user_input["RainToday"] == "No" and user_input["Rainfall"] > 0:
        st.error("❌ Ви вказали, що дощу не було, але кількість опадів більша за 0 мм.")
        st.stop()

    # Швидкість вітру не може бути більшою за максимальний порив
    if user_input["WindSpeed9am"] > user_input["WindGustSpeed"]:
        st.error("❌ Швидкість вітру о 9:00 не може перевищувати максимальний порив.")
        st.stop()

    if user_input["WindSpeed3pm"] > user_input["WindGustSpeed"]:
        st.error("❌ Швидкість вітру о 15:00 не може перевищувати максимальний порив.")
        st.stop()

    # =========================
    # Прогноз
    # =========================

    final_input = preprocess_input(user_input)

    prediction = model.predict(final_input)[0]
    probabilities = model.predict_proba(final_input)[0]

    yes_probability = probabilities[
        list(model.classes_).index("Yes")
    ]

    if prediction == "Yes":
        st.success("🌧️ Прогноз: завтра буде дощ")
    else:
        st.success("☀️ Прогноз: завтра дощу не буде")

    st.info(f"Ймовірність дощу завтра: {yes_probability:.2%}")