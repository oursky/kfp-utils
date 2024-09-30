import os
from typing import Any, Dict

import yaml

from .k8s import load_k8s_model

PIPELINE_NAME = os.getenv('PIPELINE_NAME')
TASK_IMAGE = os.getenv('TASK_IMAGE')
TRAINER_TASK_IMAGE = os.getenv('TRAINER_TASK_IMAGE')
IMAGE_TAG = os.getenv('IMAGE_TAG', 'latest')

DEFAULT_MAX_PIPELINE_CACHE_STALENESS = os.getenv(
    'DEFAULT_MAX_PIPELINE_CACHE_STALENESS'
)

TASK_IMAGE_WITH_TAG = f'{TASK_IMAGE}:{IMAGE_TAG}'
TRAINER_TASK_IMAGE_WITH_TAG = f'{TRAINER_TASK_IMAGE}:{IMAGE_TAG}'
DEFAULT_TUNNING_TASK_SERVICE_ACCOUNT = os.getenv(
    'DEFAULT_TUNNING_TASK_SERVICE_ACCOUNT',
    'pipeline-runner',
)


def load_default() -> Dict:
    default_path = os.getenv('DEFAULT_YAML')
    if not default_path:
        return {}
    with open(default_path) as fp:
        settings = yaml.safe_load(fp.read())
        return settings


DEFAULT_SETTINGS = load_default()


def get_default_settings(path: str, model_cls=None) -> Any:
    cur = DEFAULT_SETTINGS
    for part in path.split('.'):
        cur = cur.get(part)
        if cur is None:
            break

    if cur is None:
        return None

    if model_cls is not None:
        if isinstance(cur, list):
            return [load_k8s_model(model_cls, x) for x in cur]

        return load_k8s_model(model_cls, cur)

    if isinstance(cur, dict):
        return {**cur}
    elif isinstance(cur, list):
        return [*cur]
    else:
        return cur
