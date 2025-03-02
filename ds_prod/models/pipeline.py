from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# TODO: split the functionalities between differetn modules
def get_preprocessor(
        num_columns: list[str], cat_columns: list[str]
) -> tuple[str, ColumnTransformer]:
    transformer = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]),
            num_columns
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            cat_columns,
        ),
    ])
    return "preprocessor", transformer


def get_model(model_name: str):
    match model_name:
        case "logistic_regression":
            model = LogisticRegression()
        case "random_forest":
            model = RandomForestClassifier()
        case _:
            raise ValueError("Incorrect model name!")
    return "model", model


def get_pipeline(steps: list[tuple[str, Any]]):
    return Pipeline(steps)