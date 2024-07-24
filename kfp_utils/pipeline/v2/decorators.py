from kubernetes.client.models import (
    V1EnvVar,
    V1EnvVarSource,
    V1SecretKeySelector,
)


def task_with_env(name: str, value: str):
    def decorator(cls):
        cls.env = [
            *cls.env,
            V1EnvVar(
                name=name,
                value=value,
            ),
        ]
        return cls

    return decorator


def task_with_env_from_secret(
    name: str, secret_name: str, secret_key: str = None
):
    def decorator(cls):
        cls.env = [
            *cls.env,
            V1EnvVar(
                name=name,
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name=secret_name,
                        key=(secret_key or name),
                    )
                ),
            ),
        ]
        return cls

    return decorator
