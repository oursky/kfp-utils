import functools
import os.path
from typing import Callable, List, Optional

import kfp
from kfp.dsl import ContainerOp, ResourceOp

from .component_definitions.generate_random_string import (
    generate_random_string,
)


class PythonFunctionOp:
    function: Callable

    def __new__(cls, *args, **kwargs) -> ContainerOp:
        return cls._to_op_factory()(*args, **kwargs)

    @classmethod
    def _to_op_factory(cls) -> Callable[..., ContainerOp]:
        op_factory = kfp.components.create_component_from_func(cls.function)

        @functools.wraps(op_factory)
        def wrapped_op_factory(*args, **kwargs):
            op = op_factory(*args, **kwargs)
            caching_strategy = op.execution_options.caching_strategy
            caching_strategy.max_cache_staleness = 'P0D'

            return op

        return wrapped_op_factory


class RawContainerOp:
    component_defintion_path: str

    def __new__(cls, *args, **kwargs) -> ContainerOp:
        return cls._to_op_factory()(*args, **kwargs)

    @classmethod
    def _to_op_factory(cls) -> Callable[..., ContainerOp]:
        op_factory = kfp.components.load_component_from_text(
            cls._load_component_definition()
        )

        @functools.wraps(op_factory)
        def wrapped_op_factory(*args, **kwargs):
            op = op_factory(*args, **kwargs)
            caching_strategy = op.execution_options.caching_strategy
            caching_strategy.max_cache_staleness = 'P0D'

            return op

        return wrapped_op_factory

    @classmethod
    def _load_component_definition(cls) -> str:
        with open(cls.component_defintion_path, 'r') as fp:
            return fp.read()


class GenerateRandomStringOp(PythonFunctionOp):
    function = generate_random_string


class GetBestTrialOp(RawContainerOp):
    component_defintion_path = os.path.join(
        os.path.dirname(__file__),
        'component_definitions',
        'yaml',
        'get_best_trial.yaml',
    )


class DeleteK8sResourceOp(RawContainerOp):
    component_defintion_path = os.path.join(
        os.path.dirname(__file__),
        'component_definitions',
        'yaml',
        'delete_k8s_resource.yaml',
    )


class ResourceOpWithCustomDelete(ResourceOp):
    def delete(self, flags: Optional[List[str]] = None):
        """Returns a ResourceOp which deletes the resource."""
        if self.resource.action == "delete":
            raise ValueError("This operation is already a resource deletion.")

        if isinstance(self.k8s_resource, dict):
            kind = self.k8s_resource["kind"]
        else:
            kind = self.k8s_resource.kind

        return DeleteK8sResourceOp(
            name=self.outputs["name"],
            kind=kind,
            flags=' '.join(flags or ["--wait=false"]),
        )
