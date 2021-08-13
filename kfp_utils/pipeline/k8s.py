import re
from typing import Any, Dict

import kubernetes.client.models  # noqa W0611

K8S_MODEL_MODULE_PREFIX = 'kubernetes.client.models'


def get_k8s_model_cls(short_name):
    return eval(f'{K8S_MODEL_MODULE_PREFIX}.{short_name}')


def load_k8s_model(model_cls, data) -> Any:
    if data is None:
        return None

    def get_value(snake_case: str, camel_case: str) -> Any:
        value_type = model_cls.openapi_types[snake_case]
        value = data.get(camel_case)

        if value_type.startswith('V'):
            return load_k8s_model(
                get_k8s_model_cls(value_type),
                value,
            )
        elif value_type.startswith('list[V') and value:
            sub_type = re.match(r'list\[(.*)\]', value_type).group(1)
            sub_model_cls = get_k8s_model_cls(sub_type)
            return [load_k8s_model(sub_model_cls, x) for x in value]
        elif value_type.startswith('dict(') and value:
            sub_type = re.match(r'dict\([^,]*, (.*)\)', value_type).group(1)
            if sub_type[0] == 'V':
                sub_model_cls = get_k8s_model_cls(sub_type)
                return {
                    k: load_k8s_model(sub_model_cls, v)
                    for k, v in value.items()
                }

        return value

    data = {
        snake_case: get_value(snake_case, camel_case)
        for snake_case, camel_case in model_cls.attribute_map.items()
    }

    return model_cls(**data)


def dump_k8s_model(obj: Any) -> Dict:
    if isinstance(obj, list):
        return [dump_k8s_model(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: dump_k8s_model(v) for k, v in obj.items()}
    elif not obj.__class__.__module__.startswith('K8S_MODEL_MODULE_PREFIX'):
        return obj

    return {
        camel_case: dump_k8s_model(getattr(obj, snake_case))
        for snake_case, camel_case in obj.attribute_map.items()
    }
