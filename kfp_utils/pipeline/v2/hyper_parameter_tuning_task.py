import yaml
import re
import functools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable

from kfp.dsl import PipelineTask
from kfp.kubernetes import add_toleration, add_node_selector
from kubernetes.client.models import V1Affinity, V1Toleration

from .add_k8s_attribute import add_affinity
from .config import (
    DEFAULT_TUNNING_TASK_SERVICE_ACCOUNT,
    get_default_settings,
)
from .k8s import dump_k8s_model
from .ops import ResourceOp
from .optimization_algorithm import TPE, OptimizationAlgorothm
from .task import Argument, Input
from .trainer_task import TrainerTask
from .yaml_representer import FloatString, QuotedString
from .op_factory import TemplateFillOpFactory, TemplateFillArg


class MetricFilePath(str):
    pass


class MissingMetricFilePath(Exception):
    pass


class MissingOptimizingMetric(Exception):
    pass


HPT_TRIAL_NAME = '${trialParameters.trialName}'
MANIFEST_PLACEHOLDER_PATTERN = r'{{channel\:task=(?P<channel>[^;]*);name=(?P<name>[^;]+);type=(?P<type>[^;]+);}}'
inverse_type_naming = {
    value: key
    for key, value in Argument.type_naming.items()
}


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
            output['step'] = str(self.step)

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


@dataclass
class ResourceMonitorSetting:
    affinity: Optional[V1Affinity] = field(
        default_factory=lambda: (
            get_default_settings('hptResourceMonitor.affinity', V1Affinity)
            or None
        )
    )
    node_selectors: Dict[str, str] = field(
        default_factory=lambda: (
            get_default_settings('hptResourceMonitor.nodeSelector') or dict()
        )
    )
    tolerations: List[V1Toleration] = field(
        default_factory=lambda: (
            get_default_settings('task.tolerations', V1Toleration) or list()
        )
    )


class HyperParameterTuningTask(TrainerTask):

    resource_monitor_setting = ResourceMonitorSetting()

    algorithm: OptimizationAlgorothm = TPE()
    metrics: List[Metric] = list()

    parallel_trial_count: int = 3
    max_trial_count: Optional[int] = None
    max_failed_trial_count: Optional[int] = None

    metric_pattern: str = QuotedString('"([^"]+)"\\s*:\\s*"([^"]+)"')
    
    service_account_name: str = DEFAULT_TUNNING_TASK_SERVICE_ACCOUNT

    def __new__(
        cls, 
        experiment_name: str, 
        **kwargs
    ) -> ResourceOp.ResourceOpPipelineTask:
        cls.experiment_name = experiment_name
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
    def _add_resource_op_affinity(
        cls, 
        op: ResourceOp.ResourceOpPipelineTask
    ) -> ResourceOp.ResourceOpPipelineTask:
        if cls.affinity is not None:
            op = add_affinity(
                task=op,
                affinity=cls.resource_monitor_setting.affinity
            )
        return op
    
    @classmethod
    def _add_resource_op_node_selectors(
        cls, 
        op: ResourceOp.ResourceOpPipelineTask
    ) -> ResourceOp.ResourceOpPipelineTask:
        for key, value in cls.resource_monitor_setting.node_selectors.items():
            op = add_node_selector(
                task=op, 
                label_key=key, 
                label_value=value
            )
        return op
    
    @classmethod
    def _add_resource_op_tolerations(
        cls, 
        op: ResourceOp.ResourceOpPipelineTask
    ) -> ResourceOp.ResourceOpPipelineTask:
        for toleration in cls.resource_monitor_setting.tolerations:
            op = add_toleration(
                task=op,
                key=toleration.key,
                operator=toleration.operator,
                value=toleration.value,
                effect=toleration.effect,
                toleration_seconds=toleration.toleration_seconds
            )
        return op

    @classmethod
    def _inject_settings_to_resource_op(
        cls, 
        op: ResourceOp.ResourceOpPipelineTask
    ) -> ResourceOp.ResourceOpPipelineTask:
        injections = [
            cls._add_resource_op_affinity,
            cls._add_resource_op_node_selectors,
            cls._add_resource_op_tolerations,
        ]
        for injection in injections:
            op = injection(op)
        return op

    @classmethod
    def _to_resource_op(cls, **kwargs) -> ResourceOp.ResourceOpPipelineTask:
        max_failed_trial_count = kwargs.get(
            'max_failed_trial_count', cls.max_failed_trial_count
        )
        manifest_op = cls._to_manifest_op_factory(**kwargs)()
        failure_condition_op = TemplateFillOpFactory(
            template='status.trialsFailed>{max_failed_trial_count}',
            fn_name='fill_failure_condition',
            fn_args=[TemplateFillArg('max_failed_trial_count', int)]
        )(max_failed_trial_count=max_failed_trial_count)
        resource_op = ResourceOp(
            action='apply',
            manifest=manifest_op.output,
            success_condition='status.trialsSucceeded>0,status.completionTime',
            failure_condition=failure_condition_op.output,
            resource_name=cls.experiment_name,
            resource_kind='Experiment',
        ).set_display_name(cls.name)

        cls._inject_settings_to_resource_op(resource_op)

        return resource_op

    @classmethod
    def _to_resource_manifest(cls, **kwargs) -> Dict:

        container_name = f'{cls.name}-trial'

        return {
            'apiVersion': 'kubeflow.org/v1beta1',
            'kind': 'Experiment',
            'metadata': {
                'name': cls.experiment_name,
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
                        'serviceAccountName': cls.service_account_name,
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

    @classmethod
    def _extract_template_and_args_from_placeholder_text(
        cls, 
        placeholder_text: str
    ) -> tuple[str, list[TemplateFillArg]]:
        template = placeholder_text
        placeholders = list(re.finditer(MANIFEST_PLACEHOLDER_PATTERN, placeholder_text))
        fn_args = []
        for match in placeholders:
            placeholder_parts = match.groupdict()
            placeholder_name, placeholder_type = placeholder_parts['name'], placeholder_parts['type']
            fn_args.append(
                TemplateFillArg(
                    placeholder_name, 
                    inverse_type_naming[placeholder_type]
                )
            )
            template = template.replace(match.group(), f"{{{placeholder_name}}}")
        # wrap all k8s template variables with additional curl brackets
        #   so that they are ignored in the composed function f-string replacement
        template = re.sub(r"\${([\w\. ]+)}", r"${{\1}}", template)
        return template, fn_args

    @classmethod
    # kubeflow driver occasionally fails to substitute all pipeline variables ${{inputs.xxx}} in the manifest template
    #   if the manifest template input is crafted directly as a yaml string in template generation time
    # 
    # workaround: create an intermediate PythonFunctionOp that returns a substituted manifest template
    # potential related issue: https://github.com/kubeflow/pipelines/issues/8328
    # FIXME: create a minimal reproducible sample and report in KFP github
    def _to_manifest_op_factory(cls, **kwargs) -> Callable[..., PipelineTask]:
        manifest_dict = cls._to_resource_manifest(**kwargs)
        # replace metadata name with f-string placeholder for substitution in the composed function
        placeholder_name = 'experiment_name'
        manifest_dict['metadata']['name'] = f'{{{placeholder_name}}}'
        template, fn_args = cls._extract_template_and_args_from_placeholder_text(yaml.dump(manifest_dict))
        fn_args.append(TemplateFillArg(placeholder_name, str))
        manifest_op_factory = TemplateFillOpFactory(
            template=template,
            fn_name=f'create_manifest',
            fn_args=fn_args,
        )
        @functools.wraps(manifest_op_factory)
        def wrapped_manifest_op_factory(*args, **additional_kwargs):
            op_kwargs = kwargs | additional_kwargs
            op = manifest_op_factory(
                experiment_name=cls.experiment_name,
                *args,
                **op_kwargs,
            )
            op.set_caching_options(False)
            return op
        return wrapped_manifest_op_factory
