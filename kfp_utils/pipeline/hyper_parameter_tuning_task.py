import random
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from kfp.dsl import ResourceOp

from .k8s import dump_k8s_model
from .ops import ResourceOpWithCustomDelete
from .optimization_algorithm import TPE, OptimizationAlgorothm
from .task import Argument, Input
from .trainer_task import TrainerTask
from .yaml_representer import FloatString, QuotedString


class MetricFilePath(str):
    pass


class MissingMetricFilePath(Exception):
    pass


class MissingOptimizingMetric(Exception):
    pass


HPT_TRIAL_NAME = '${trialParameters.trialName}'


def force_number(s: Any) -> Any:
    if isinstance(s, float) or isinstance(s, int):
        return s

    return f'+{s}'


@dataclass
class ParameterRange:
    min: Union[int, float]
    max: Union[int, float]
    step: Optional[Union[int, float]] = None

    def to_feasible_space(self) -> Dict:
        output = {
            'min': FloatString(self.min),
            'max': FloatString(self.max),
        }

        if self.step is not None:
            output['step'] = self.step

        return output


@dataclass
class _HyperParameterBase:
    range: Union[List[str], ParameterRange]


@dataclass
class HyperParameter(Argument, _HyperParameterBase):

    command_arg_type: str = ''

    type_naming = {
        str: 'categorical',
        int: 'int',
        float: 'double',
    }

    def to_parameter(self) -> Dict:
        return {
            'name': self.name,
            'parameterType': (
                'categorical'
                if isinstance(self.range, list)
                else self.type_naming[self.type]
            ),
            'feasibleSpace': (
                {'list': [str(x) for x in self.range]}
                if isinstance(self.range, list)
                else self.range.to_feasible_space()
            ),
        }

    def to_trial_parameter(self) -> Dict:
        output = {'name': self.name, 'reference': self.name}

        if self.description is not None:
            output['description'] = self.description

        return output

    def to_command_args(self, *args, **kwargs) -> List:
        output = super().to_command_args()
        output[1] = f'${{trialParameters.{self.name}}}'
        return output


@dataclass
class Metric:
    name: str


@dataclass
class OptimizingMetric(Metric):
    goal: float
    is_maximize: bool = True


class HyperParameterTuningTask(TrainerTask):
    algorithm: OptimizationAlgorothm = TPE()
    metrics: List[Metric] = list()

    parallel_trial_count: int = 3
    max_trial_count: Optional[int] = None
    max_failed_trial_count: Optional[int] = None

    metric_pattern: str = QuotedString('"([^"]+)"\\s*:\\s*"([^"]+)"')

    def __new__(cls, experiment_suffix: str, **kwargs) -> ResourceOp:
        cls.experiment_suffix = experiment_suffix
        return cls._to_resource_op(**kwargs)

    @classmethod
    def get_hyperparameters(cls) -> List[HyperParameter]:
        return [x for x in cls.inputs if isinstance(x, HyperParameter)]

    @classmethod
    def get_template_variables(cls) -> List[Dict[str, str]]:
        return [
            {
                'name': 'trialName',
                'reference': '${trialSpec.Name}',
            }
        ]

    @classmethod
    def get_metric_file_path(cls) -> Input:
        metric_file_path = next(
            (x for x in cls.inputs if x.type == MetricFilePath), None
        )
        if metric_file_path is None:
            raise MissingMetricFilePath

        return metric_file_path

    @classmethod
    def get_optimizing_metric(cls) -> OptimizingMetric:
        metric = next(
            (x for x in cls.metrics if isinstance(x, OptimizingMetric)), None
        )
        if metric is None:
            raise MissingOptimizingMetric

        return metric

    @classmethod
    def get_metric_collector_spec(cls, kwargs: Dict) -> Dict:
        return {
            'collector': {'kind': 'File'},
            'source': {
                'filter': {'metricsFormat': [cls.metric_pattern]},
                'fileSystemPath': {
                    'kind': 'File',
                    'path': cls.get_metric_file_path().to_command_args(kwargs)[
                        1
                    ],
                },
            },
        }

    @classmethod
    def get_objective_spec(cls) -> Dict:
        optimizing_metric = cls.get_optimizing_metric()
        return {
            'type': (
                'maximize' if optimizing_metric.is_maximize else 'minimize'
            ),
            'goal': optimizing_metric.goal,
            'objectiveMetricName': QuotedString(optimizing_metric.name),
            **cls._inject_settings_to_manifest(
                'additionalMetricNames',
                [
                    QuotedString(x.name)
                    for x in cls.metrics
                    if not isinstance(x, OptimizingMetric)
                ],
            ),
        }

    @classmethod
    def _to_resource_op(cls, **kwargs) -> ResourceOp:
        max_failed_trial_count = kwargs.get(
            'max_failed_trial_count', cls.max_failed_trial_count
        )

        return ResourceOpWithCustomDelete(
            name=cls.name,
            k8s_resource=cls._to_resouece_manifest(**kwargs),
            action='apply',
            success_condition='status.trialsSucceeded>0,status.completionTime',
            failure_condition=f'status.trialsFailed>{max_failed_trial_count}',  # noqa
        )

    @classmethod
    def get_experiment_name(cls) -> str:
        return f'{cls.name}-{cls.experiment_suffix}'

    @classmethod
    def _to_resouece_manifest(cls, **kwargs) -> Dict:

        container_name = f'{cls.name}-trial'

        return {
            'apiVersion': 'kubeflow.org/v1beta1',
            'kind': 'Experiment',
            'metadata': {
                'name': cls.get_experiment_name(),
            },
            'spec': {
                **cls.algorithm.to_algorithm_spec(),
                **cls._inject_settings_to_manifest(
                    'parallelTrialCount',
                    force_number(
                        kwargs.pop(
                            'parallel_trial_count', cls.parallel_trial_count
                        )
                    ),
                ),
                **cls._inject_settings_to_manifest(
                    'maxTrialCount',
                    force_number(
                        kwargs.pop('max_trial_count', cls.max_trial_count)
                    ),
                ),
                **cls._inject_settings_to_manifest(
                    'maxFailedTrialCount',
                    force_number(
                        kwargs.pop(
                            'max_failed_trial_count',
                            cls.max_failed_trial_count,
                        )
                    ),
                ),
                'metricsCollectorSpec': cls.get_metric_collector_spec(kwargs),
                'objective': cls.get_objective_spec(),
                'parameters': [
                    hp.to_parameter() for hp in cls.get_hyperparameters()
                ],
                'trialTemplate': {
                    'primaryContainerName': container_name,
                    'trialParameters': (
                        [
                            hp.to_trial_parameter()
                            for hp in cls.get_hyperparameters()
                        ]
                        + cls.get_template_variables()
                    ),
                    'trialSpec': cls._to_job_manifest(container_name, kwargs),
                },
            },
        }

    @classmethod
    def _to_job_manifest(cls, container_name: str, kwargs: Dict) -> Dict:
        return {
            'apiVersion': 'batch/v1',
            'kind': 'Job',
            'spec': {
                'template': {
                    'spec': {
                        'containers': [
                            {
                                'name': container_name,
                                'image': cls.image,
                                'workingDir': cls.working_dir,
                                'command': cls.get_command(lut=kwargs),
                                **cls._inject_settings_to_manifest(
                                    'env',
                                    cls.env,
                                ),
                                **cls._inject_settings_to_manifest(
                                    'envFrom',
                                    cls.env_from,
                                ),
                                **cls._inject_settings_to_manifest(
                                    'volumeMounts',
                                    cls.volume_mounts,
                                ),
                                'resources': {
                                    **cls._inject_settings_to_manifest(  # noqa
                                        'requests',
                                        cls.resource_requests,
                                    ),
                                    **cls._inject_settings_to_manifest(  # noqa
                                        'limits',
                                        cls.resource_limits,
                                    ),
                                },
                            }
                        ],
                        'restartPolicy': 'Never',
                        **cls._inject_settings_to_manifest(
                            'nodeSelector', cls.node_selectors
                        ),
                        **cls._inject_settings_to_manifest(
                            'tolerations', cls.tolerations
                        ),
                        **cls._inject_settings_to_manifest(
                            'volumes', cls.volumes
                        ),
                    }
                }
            },
        }

    @classmethod
    def _inject_settings_to_manifest(
        cls, fields, data: Union[List, Dict]
    ) -> Dict:
        if data:
            return {fields: dump_k8s_model(data)}

        return {}
