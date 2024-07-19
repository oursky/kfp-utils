from typing import Optional

from kfp.dsl import PipelineTask
from kfp.kubernetes import (
    use_config_map_as_env,
    use_secret_as_env,
    use_field_path_as_env,
)
from kubernetes.client.models import (
    V1EnvVar,
    V1EnvVarSource,
)

def add_env_vars(
    task: PipelineTask,
    env_vars: list[V1EnvVar]
) -> PipelineTask:
        for env_var in env_vars:
            value_from: Optional[V1EnvVarSource] = env_var.value_from
            if value_from is None:
                task.set_env_variable(
                    name=env_var.name,
                    value=env_var.value,
                )
            elif value_from.config_map_key_ref is not None:
                key = value_from.config_map_key_ref.key
                config_map_name = value_from.config_map_key_ref.name,
                task = use_config_map_as_env(
                    task=task,
                    config_map_name=config_map_name,
                    config_map_key_to_env={key: env_var.name},
                )
            elif value_from.secret_key_ref is not None:
                key = value_from.secret_key_ref.key
                secret_name = value_from.secret_key_ref.name
                task = use_secret_as_env(
                    task=task,
                    secret_name=secret_name,
                    secret_key_to_env={key: env_var.name}
                )
            elif value_from.field_ref is not None:
                task = use_field_path_as_env(
                    task=task,
                    env_name=env_var.name,
                    field_path=value_from.field_ref.field_path,
                )
            else:
                raise ValueError(f'Unsupported env type for env var {env_var.name}')
        return task
