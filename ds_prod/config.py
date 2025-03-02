import yaml
from pathlib import Path

# TODO: Rewrite with Pydantic
def load_task_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open() as f:
        config = yaml.safe_load(f)

    return config

