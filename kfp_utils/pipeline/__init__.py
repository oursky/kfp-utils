from importlib.metadata import version

def resolve_package_version_to_tuple(package_name: str) -> tuple[int]:
    package_version = version(package_name)
    return tuple([ 
        int(version_part) 
        for version_part in package_version.split('.')
    ])

kfp_version = resolve_package_version_to_tuple('kfp')

if kfp_version < (2, 0):
    # import KFP-V1 codes
    from .hyper_parameter_tuning_task import (
        HyperParameter,
        HyperParameterTuningTask,
        Metric,
        MetricFilePath,
        OptimizingMetric,
        ParameterRange,
    )
    from .task import Flag, Input, Output, RetryArgs, Task
    from .trainer_task import TrainerTask
    from .config import PIPELINE_NAME
    from .op_group import OpGroup
    from .decorators import task_with_env_from_secret
else:
    # import KFP-v2 codes
    from .v2 import *

__all__ = [
    'Flag',
    'Input',
    'Output',
    'Task',
    'HyperParameter',
    'HyperParameterTuningTask',
    'Metric',
    'MetricFilePath',
    'OptimizingMetric',
    'TrainerTask',
    'ParameterRange',
    'RetryArgs',
    'PIPELINE_NAME',
    'OpGroup',
    'task_with_env_from_secret',
]
