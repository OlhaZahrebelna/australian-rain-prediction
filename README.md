# 🌦️ Australian Rain Prediction

A machine learning web application that predicts whether it will rain tomorrow in Australia using historical weather data and a Random Forest classifier.

# 🌐 Available Languages

The application is available in three language versions:

* 🇬🇧 **English**
* 🇺🇦 **Українська (Ukrainian)**
* 🇵🇱 **Polski (Polish)**

All three versions provide the same functionality and machine learning model, differing only in the user interface language. Users can enter weather conditions and receive a prediction of whether it will rain tomorrow along with the estimated probability.

## Project Structure

```text
.
├── appEN.py                  # English interface
├── appUA.py                  # Ukrainian interface
├── appPL.py                  # Polish interface
├── weather_data_processing.py
├── models/
│   └── aussie_rain.joblib
├── data/
│   └── weatherAUS.csv
├── notebooks/
├── requirements.txt
└── README.md
```
