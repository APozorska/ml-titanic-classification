import joblib

from sklearn.model_selection import train_test_split

from ds_prod.config import load_task_config
from ds_prod.logger import get_logger
from ds_prod.data.loader import load_csv
from ds_prod.models.pipeline import get_preprocessor, get_model, get_pipeline

logger = get_logger("Training Pipeline", log_level="DEBUG")


def main(config_path: str):
    # Load configuration
    config = load_task_config(config_path)
    logger.debug(f"Config: {config}")
    logger.info("Configuration loaded")

    # Load data
    data_path = config["data"]["path"]
    data = load_csv(data_path)
    num_cols = config["data"]["features"]["num"]
    cat_cols = config["data"]["features"]["cat"]
    target_col = config["data"]["target_col"]

    # Split feaures and target
    logger.debug(f"Num features: {num_cols}; cat features: {cat_cols}")
    X = data[num_cols + cat_cols]
    y = data[target_col]
    logger.info("Data loaded correctly!")

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.3, random_state=config["model"]["random_state"])
    logger.debug(f"Len of train data: {len(X_train)}, test_data: {len(X_test)}")

    # Create pipeline
    preprocessor = get_preprocessor(num_cols, cat_cols)
    model = get_model(config["model"]["name"])
    pipeline = get_pipeline([preprocessor, model])
    logger.info("Pipeline loaded successfully")

    # Train model
    # TODO: Perform hyperparameter optimazation
    pipeline.fit(X_train, y_train)
    logger.info("Model training completed")

    # Evaluate
    accuracy = pipeline.score(X_test, y_test)
    logger.info(f"Model accuracy: {accuracy:.4f}")

    # Save model
    model_path = config["model"]["output_path"]
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved in {model_path}")


if __name__ == '__main__':
    from ds_prod.settings import CFG_PATH

    main(CFG_PATH)
