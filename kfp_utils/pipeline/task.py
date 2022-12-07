import functools
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import yaml
from kfp.dsl import ContainerOp
from kfp_utils.pipeline.config import (
    DEFAULT_MAX_PIPELINE_CACHE_STALENESS,
    PIPELINE_NAME,
    TASK_IMAGE_WITH_TAG,
    get_default_settings,
)
from kubernetes.client.models import (
    V1Affinity,
    V1EnvFromSource,
    V1EnvVar,
    V1Toleration,
    V1Volume,
    V1VolumeMount,
)

import kfp


@dataclass
class Argument:
    name: str
    type: type

    command_arg_type: str
    description: Optional[str] = None

    type_naming = {
        str: 'String',
        int: 'Integer',
        float: 'Float',
        bool: 'Bool',
    }

    def to_component_args(self) -> Dict:
        return {
            'name': self.name,
            'type': self.type_naming.get(self.type, self.type.__name__),
            **({'description': self.description} if self.description else {}),
        }

    def to_command_args(self, lut: Optional[Dict[str, str]] = None) -> List:
        return [
            f'--{self.name}',
            {self.command_arg_type: self.name}
            if lut is None
            else str(lut[self.name]),
        ]


@dataclass
class Flag:
    name: str

    def to_command_arg(self) -> str:
        return f'--{self.name}'


@dataclass
class Input(Argument):
    command_arg_type: str = 'inputValue'
    default: Optional[Any] = None

    def to_component_args(self) -> Dict:
        def_dict = super().to_component_args()
        if self.default is not None:
            def_dict['default'] = str(self.default)

        return def_dict

    def to_command_args(self, lut: Optional[Dict[str, str]] = None) -> List:
        if lut is None or self.name in lut:
            return super().to_command_args(lut=lut)

        output = super().to_command_args()
        output[1] = self.default_str or str(lut[self.name])
        return output

    @property
    def default_str(self) -> Optional[str]:
        if self.default is not None:
            return str(self.default)

        return None


SPECIAL_OUTPUTS = [
    'mlpipeline_ui_metadata',
    'mlpipeline_metrics',
]


@dataclass
class Output(Argument):
    command_arg_type: str = 'outputPath'

    def to_component_args(self) -> Dict:
        output = super().to_component_args()
        if self.name in SPECIAL_OUTPUTS:
            output['name'] = self.name.replace('_', '-')
        return output

    def to_command_args(self, *args, **kwargs) -> List:
        output = super().to_command_args(*args, **kwargs)
        if self.name in SPECIAL_OUTPUTS:
            output[1][self.command_arg_type] = self.name.replace('_', '-')
        return output


class TaskMeta(type):
    auto_naming_classes = []

    def __init__(cls, name, bases, attr):
        super(TaskMeta, cls).__init__(name, bases, attr)

        TaskMeta.auto_fill_name(cls)

    @staticmethod
    def auto_fill_name(cls):
        if cls.name is None and cls.__name__ not in [
            'Task',
            'TrainerTask',
            'HyperParameterTuningTask',
        ]:
            cls.name = re.sub(
                r'([a-z0-9])([A-Z])',
                r'\1-\2',
                re.sub(r'Op$', '', cls.__name__),
            ).lower()

        if cls.command_name is None:
            cls.command_name = cls.name


@dataclass
class RetryArgs:
    # ref: https://github.com/kubeflow/pipelines/blob/d5bc8ddd6250d90b38cff5759e856f73e71e7d03/sdk/python/kfp/dsl/_container_op.py#L1016 #noqa
    num_retries: int
    policy: Optional[str] = None
    backoff_duration: Optional[str] = None
    backoff_factor: Optional[float] = None
    backoff_max_duration: Optional[str] = None


class Task(metaclass=TaskMeta):

    image: str = TASK_IMAGE_WITH_TAG
    working_dir: str = f'/tasks/{PIPELINE_NAME}'

    name: Optional[str] = None
    description: Optional[str] = None
    command_name: Optional[str] = None

    inputs: List[Input] = list()
    outputs: List[Output] = list()
    flags: List[Flag] = list()

    max_cache_staleness: Optional[str] = DEFAULT_MAX_PIPELINE_CACHE_STALENESS
    image_pull_policy: Optional[str] = None

    affinity: Optional[V1Affinity] = (
        get_default_settings('task.affinity', V1Affinity) or None
    )
    node_selectors: Dict[str, str] = (
        get_default_settings('task.nodeSelector') or dict()
    )
    resource_limits: Dict[str, str] = (
        get_default_settings('task.resources.limits') or dict()
    )
    resource_requests: Dict[str, str] = (
        get_default_settings('task.resources.requests') or dict()
    )

    tolerations: List[V1Toleration] = (
        get_default_settings('task.tolerations', V1Toleration) or list()
    )
    volumes: List[V1Volume] = (
        get_default_settings('task.volumes', V1Volume) or list()
    )
    volume_mounts: List[V1VolumeMount] = (
        get_default_settings(
            'task.volumeMounts',
            V1VolumeMount,
        )
        or list()
    )
    env: List[V1EnvVar] = get_default_settings('task.env', V1EnvVar) or list()
    env_from: List[V1EnvFromSource] = (
        get_default_settings('task.envFrom', V1EnvFromSource) or list()
    )

    retry: Optional[RetryArgs] = None

    def __new__(cls, *args, **kwargs) -> ContainerOp:
        return cls._to_op_factory()(*args, **kwargs)

    @classmethod
    def get_command_name(cls) -> str:
        return f'./{cls.command_name}.py'

    @classmethod
    def get_command(cls, lut: Optional[Dict[str, str]] = None) -> List[Any]:
        return (
            [cls.get_command_name()]
            + sum([x.to_command_args(lut=lut) for x in cls.inputs], [])
            + sum([x.to_command_args(lut=lut) for x in cls.outputs], [])
            + [x.to_command_arg() for x in cls.flags]
        )

    @classmethod
    def _to_op_factory(cls) -> Callable[..., ContainerOp]:
        kfp_component = {
            'name': cls.name,
            **({'description': cls.description} if cls.description else {}),
            **(
                {'inputs': [x.to_component_args() for x in cls.inputs]}
                if len(cls.inputs)
                else {}
            ),
            **(
                {'outputs': [x.to_component_args() for x in cls.outputs]}
                if len(cls.outputs)
                else {}
            ),
            'implementation': {
                'container': {
                    'image': cls.image,
                    'command': cls.get_command(),
                }
            },
        }

        text = yaml.dump(kfp_component)
        op_factory = kfp.components.load_component_from_text(text)

        @functools.wraps(op_factory)
        def wrapped_op_factory(*args, **kwargs):
            op = op_factory(*args, **kwargs)
            op._container.working_dir = cls.working_dir
            cls._inject_settings_to_container_op(op)
            cls._set_max_cache_staleness_to_container_op(op)
            cls._set_image_pull_policy_to_container_op(op)
            cls._set_retry_to_container_op(op)

            return op

        return wrapped_op_factory

    @classmethod
    def _inject_settings_to_container_op(cls, op: ContainerOp):
        injections = [
            (cls.affinity, op.add_affinity),
            (cls.node_selectors, op.add_node_selector_constraint),
            (cls.resource_limits, op.add_resource_limit),
            (cls.resource_requests, op.add_resource_request),
            (cls.tolerations, op.add_toleration),
            (cls.volumes, op.add_volume),
            (cls.volume_mounts, op.add_volume_mount),
            (cls.env, op.add_env_variable),
            (cls.env_from, op.add_env_from),
        ]

        for attr, injector in injections:

            if isinstance(attr, dict):
                for k, v in attr.items():
                    injector(k, v)
            elif isinstance(attr, list):
                for x in attr:
                    injector(x)
            elif isinstance(attr, V1Affinity):
                injector(attr)
            elif attr is not None:
                raise Exception(f'Unexpected attribute type: {type(attr)}')

    @classmethod
    def _set_max_cache_staleness_to_container_op(cls, op: ContainerOp):
        if cls.max_cache_staleness is not None:
            caching_strategy = op.execution_options.caching_strategy
            caching_strategy.max_cache_staleness = cls.max_cache_staleness

    @classmethod
    def _set_image_pull_policy_to_container_op(cls, op: ContainerOp):
        if cls.image_pull_policy is not None:
            op.set_image_pull_policy(cls.image_pull_policy)

    @classmethod
    def _set_retry_to_container_op(cls, op: ContainerOp):
        if cls.retry is not None:
            op.set_retry(
                num_retries=cls.retry.num_retries,
                policy=cls.retry.policy,
                backoff_duration=cls.retry.backoff_duration,
                backoff_factor=cls.retry.backoff_factor,
                backoff_max_duration=cls.retry.backoff_max_duration,
            )
