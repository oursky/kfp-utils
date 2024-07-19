import os.path
import functools
from typing import Callable

from kfp.components import load_component_from_text, YamlComponent
from kfp.dsl import PipelineTask, component

from .component_definitions.generate_random_string import (
    generate_random_string,
)

class PythonFunctionOp:
    function: Callable
    base_image: str = "python:3.10-slim-bookworm"

    def __new__(cls, *args, **kwargs) -> PipelineTask:
        return cls._to_op_factory()(*args, **kwargs)

    @classmethod
    def _to_op_factory(cls) -> Callable[..., PipelineTask]:
        op_factory = component(
            cls.function,
            base_image=cls.base_image
        )

        @functools.wraps(op_factory)
        def wrapped_op_factory(*args, **kwargs):
            op = op_factory(*args, **kwargs)
            op.set_caching_options(False)

            return op

        return wrapped_op_factory


class RawContainerOp:
    component_defintion_path: str

    cache_enabled: bool = False

    def __new__(cls, *args, **kwargs) -> PipelineTask:
        return cls._to_op_factory()(*args, **kwargs)

    @classmethod
    def _to_op_factory(cls) -> Callable[..., PipelineTask]:
        op_factory = load_component_from_text(
            cls._load_component_definition()
        )
        return cls._get_wrapped_op_factory(op_factory)
    
    @classmethod
    def _get_wrapped_op_factory(
        cls, 
        op_factory: YamlComponent
    ) -> Callable[..., PipelineTask]:
        @functools.wraps(op_factory)
        def wrapped_op_factory(*args, **kwargs):
            op = op_factory(*args, **kwargs)
            op = op.set_caching_options(cls.cache_enabled)
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
