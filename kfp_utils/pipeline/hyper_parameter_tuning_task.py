from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from kfp.dsl import ResourceOp

from .k8s import dump_k8s_model
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
            'parameterType': self.type_naming[self.type],
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

    metric_pattern: str = QuotedString('"([^"]+)"\\\\s*:\\\\s*"([^"]+)"')

    def __new__(cls, **kwargs) -> ResourceOp:
        return cls._to_resource_op(**kwargs)

    @classmethod
    def get_hyperparameters(cls) -> List[HyperParameter]:
        return [x for x in cls.inputs if isinstance(x, HyperParameter)]

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
                'optimizing_metric',
                [
                    QuotedString(x.name)
                    for x in cls.metrics
                    if not isinstance(x, OptimizingMetric)
                ],
            ),
        }

    @classmethod
    def _to_resource_op(cls, **kwargs) -> ResourceOp:
        return ResourceOp(
            name=cls.name,
            k8s_resource=cls._to_resouece_manifest(**kwargs),
            action='create',
            success_condition='status.trialsSucceeded>1,status.completionTime',
            # failure_condition=f'status.trialsFailed>{cls.max_failed_trial_count}',
        )

    @classmethod
    def _to_resouece_manifest(cls, **kwargs) -> Dict:

        container_name = f'{cls.name}-trial'

        return {
            'apiVersion': 'kubeflow.org/v1beta1',
            'kind': 'Experiment',
            'metadata': {
                'name': f'{cls.name}-experiment',
            },
            'spec': {
                **cls.algorithm.to_algorithm_spec(),
                **cls._inject_settings_to_manifest(
                    'parallelTrialCount',
                    kwargs.pop(
                        'parallel_trial_count', cls.parallel_trial_count
                    ),
                ),
                **cls._inject_settings_to_manifest(
                    'maxTrialCount',
                    kwargs.pop('max_trial_count', cls.max_trial_count),
                ),
                **cls._inject_settings_to_manifest(
                    'maxFailedTrialCount',
                    kwargs.pop(
                        'max_failed_trial_count', cls.max_failed_trial_count
                    ),
                ),
                'metricsCollectorSpec': cls.get_metric_collector_spec(kwargs),
                'objective': cls.get_objective_spec(),
                'parameters': [
                    hp.to_parameter() for hp in cls.get_hyperparameters()
                ],
                'trialTemplate': {
                    'primaryContainerName': container_name,
                    'trialParameters': [
                        hp.to_trial_parameter()
                        for hp in cls.get_hyperparameters()
                    ],
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
