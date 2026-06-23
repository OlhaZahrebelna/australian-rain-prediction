import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Any


def drop_na_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drop rows with NA values in the specified columns."""
    return df.dropna(subset=columns).copy()


def split_data_by_year(df: pd.DataFrame, year_col: str) -> Dict[str, pd.DataFrame]:
    """Split the dataframe into train, validation and test sets by year."""
    year = pd.to_datetime(df[year_col]).dt.year

    train_df = df[year < 2015].copy()
    val_df = df[year == 2015].copy()
    test_df = df[year > 2015].copy()

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def create_inputs_targets(
    df_dict: Dict[str, pd.DataFrame],
    input_cols: list,
    target_col: str,
) -> Dict[str, Any]:
    """Create inputs and targets for train, validation and test sets."""
    data = {}

    for split in df_dict:
        data[f"{split}_inputs"] = df_dict[split][input_cols].copy()
        data[f"{split}_targets"] = df_dict[split][target_col].copy()

    return data


def impute_missing_values(data: Dict[str, Any], numeric_cols: list):
    """Impute missing numeric values using the mean from the training set."""
    if not numeric_cols:
        return None

    imputer = SimpleImputer(strategy="mean")
    imputer.fit(data["train_inputs"][numeric_cols])

    for split in ["train", "val", "test"]:
        data[f"{split}_inputs"].loc[:, numeric_cols] = imputer.transform(
            data[f"{split}_inputs"][numeric_cols]
        )

    return imputer


def scale_numeric_features(data: Dict[str, Any], numeric_cols: list):
    """Scale numeric features using MinMaxScaler fitted on the training set."""
    if not numeric_cols:
        return None

    scaler = MinMaxScaler()
    scaler.fit(data["train_inputs"][numeric_cols])

    for split in ["train", "val", "test"]:
        data[f"{split}_inputs"].loc[:, numeric_cols] = scaler.transform(
            data[f"{split}_inputs"][numeric_cols]
        )

    return scaler


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Create OneHotEncoder compatible with both old and new scikit-learn versions.

    scikit-learn < 1.2 uses sparse=False.
    scikit-learn >= 1.2 uses sparse_output=False.
    """
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


def encode_categorical_features(data: Dict[str, Any], categorical_cols: list):
    """One-hot encode categorical features."""
    if not categorical_cols:
        return None, []

    encoder = make_one_hot_encoder()
    encoder.fit(data["train_inputs"][categorical_cols])

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    for split in ["train", "val", "test"]:
        inputs = data[f"{split}_inputs"]
        encoded = encoder.transform(inputs[categorical_cols])

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoded_cols,
            index=inputs.index,
        )

        data[f"{split}_inputs"] = pd.concat(
            [inputs.drop(columns=categorical_cols), encoded_df],
            axis=1,
        )

    return encoder, encoded_cols


def preprocess_data(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Preprocess the raw weather dataframe.

    Returns train/validation/test datasets and all preprocessing objects needed
    later for Streamlit inference: input columns, numeric columns, categorical
    columns, imputer, scaler, encoder and encoded column names.
    """
    raw_df = drop_na_values(raw_df, ["RainToday", "RainTomorrow"])

    split_dfs = split_data_by_year(raw_df, "Date")

    input_cols = list(raw_df.columns)[1:-1]
    target_col = "RainTomorrow"

    data = create_inputs_targets(split_dfs, input_cols, target_col)

    numeric_cols = data["train_inputs"].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data["train_inputs"].select_dtypes(include="object").columns.tolist()

    imputer = impute_missing_values(data, numeric_cols)
    scaler = scale_numeric_features(data, numeric_cols)
    encoder, encoded_cols = encode_categorical_features(data, categorical_cols)

    # Metadata and fitted preprocessing objects needed for Streamlit deployment
    data["input_cols"] = input_cols
    data["target_col"] = target_col
    data["numeric_cols"] = numeric_cols
    data["categorical_cols"] = categorical_cols
    data["encoded_cols"] = encoded_cols
    data["imputer"] = imputer
    data["scaler"] = scaler
    data["encoder"] = encoder

    # Keys used in your notebook
    data["train_X"] = data["train_inputs"]
    data["train_y"] = data["train_targets"]
    data["val_X"] = data["val_inputs"]
    data["val_y"] = data["val_targets"]
    data["test_X"] = data["test_inputs"]
    data["test_y"] = data["test_targets"]

    return data
