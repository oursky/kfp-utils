from typing import Dict, List, Optional

from kfp_utils.pipeline.config import get_default_settings
from kubernetes.client.models import (
    V1Affinity,
    V1EnvFromSource,
    V1EnvVar,
    V1Toleration,
    V1Volume,
    V1VolumeMount,
)

from .config import TRAINER_TASK_IMAGE_WITH_TAG
from .task import Task


class TrainerTask(Task):
    image: str = TRAINER_TASK_IMAGE_WITH_TAG

    affinity: Optional[V1Affinity] = (
        get_default_settings('trainerTask.affinity')
        or get_default_settings('task.affinity')
        or None
    )

    node_selectors: Dict[str, str] = (
        get_default_settings('trainerTask.nodeSelector')
        or get_default_settings('task.nodeSelector')
        or dict()
    )

    resource_limits: Dict[str, str] = (
        get_default_settings('trainerTask.resources.limits')
        or get_default_settings('task.resources.limits')
        or dict()
    )

    resource_requests: Dict[str, str] = (
        get_default_settings('trainerTask.resources.requests')
        or get_default_settings('task.resources.requests')
        or dict()
    )

    tolerations: List[V1Toleration] = (
        get_default_settings('trainerTask.tolerations', V1Toleration)
        or get_default_settings('task.tolerations', V1Toleration)
        or list()
    )
    volumes: List[V1Volume] = (
        get_default_settings('trainerTask.volumes', V1Volume)
        or get_default_settings('task.volumes', V1Volume)
        or list()
    )
    volume_mounts: List[V1VolumeMount] = (
        get_default_settings(
            'trainerTask.volumeMounts',
            V1VolumeMount,
        )
        or get_default_settings(
            'task.volumeMounts',
            V1VolumeMount,
        )
        or list()
    )
    env: List[V1EnvVar] = (
        get_default_settings('trainerTask.env', V1EnvVar)
        or get_default_settings('task.env', V1EnvVar)
        or list()
    )
    env_from: List[V1EnvFromSource] = (
        get_default_settings('trainerTask.envFrom', V1EnvFromSource)
        or get_default_settings('task.envFrom', V1EnvFromSource)
        or list()
    )
