import os.path
import functools
from typing import Callable, List, Optional, Dict, Any

from kfp.components import load_component_from_text, YamlComponent
from kfp.dsl import PipelineTask, pipeline_context, component
from kfp.dsl.structures import ComponentSpec
from kubernetes.client.models import (
    V1Volume,
    V1EphemeralVolumeSource,
    V1PersistentVolumeClaimTemplate,
    V1PersistentVolumeClaimSpec,
    V1VolumeMount,
)
from .component_definitions.generate_random_string import (
    generate_random_string,
)
from .add_k8s_attribute import add_volumes_and_mounts

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


# ResourceOp is deprecated in KFP-V2
# as a workaround, the following image is used to replicate the previous ResourceOp behavior
# 
# ref: https://github.com/kubeflow/kfp-tekton/blob/0b894195443ca225652720a3342517dfaa304c95/tekton-catalog/kubectl-wrapper/deploy/kubectl-deploy.yaml
class ResourceOp(RawContainerOp):
    component_defintion_path = os.path.join(
        os.path.dirname(__file__),
        'component_definitions',
        'yaml',
        'resource_op.yaml',
    )

    RESOURCE_OP_OUTPUT_VOLUME_NAME = 'resource-op-output'
    # to align with the hardcoded output directory path of the image
    RESOURCE_OP_OUTPUT_VOLUME_MOUNT_PATH = '/tekton/results'

    class ResourceOpPipelineTask(PipelineTask):
        def __init__(
            self, 
            component_spec: ComponentSpec, 
            args: Dict[str, Any], 
            execute_locally: bool = False,
            *,
            resource_name: str = '',
            resource_kind: str = '',
        ) -> None:
            if not resource_name or not resource_kind:
                # FIXME: infer the resource name and kind from the input manifest
                raise ValueError('resource_name and resource_kind must take non-empty value')
            super().__init__(component_spec, args, execute_locally)
            self._resource_name = resource_name
            self._resource_kind = resource_kind
        
        def delete(self, flags: Optional[List[str]] = None):
            """Returns a ResourceOp which deletes the resource."""

            return DeleteK8sResourceOp(
                name=self._resource_name,
                kind=self._resource_kind,
                flags=' '.join(flags or ["--wait=false"]),
            )

    @staticmethod
    def extract_task_inputs_from_yaml_component(
        yaml_component: YamlComponent,
        *args, 
        **kwargs,
    ) -> dict[str, Any]:
        task_inputs = {}
        if args:
            raise TypeError(
                'Components must be instantiated using keyword arguments. Positional '
                f'parameters are not allowed (found {len(args)} such parameters for '
                f'component "{yaml_component.name}").')
        for k, v in kwargs.items():
            if k not in yaml_component._component_inputs:
                raise TypeError(
                    f'{yaml_component.name}() got an unexpected keyword argument "{k}".')
            task_inputs[k] = v

        # Skip optional inputs and arguments typed as PipelineTaskFinalStatus.
        missing_arguments = [
            arg 
            for arg in yaml_component.required_inputs 
            if arg not in kwargs
        ]
        if missing_arguments:
            argument_or_arguments = 'argument' if len(
                missing_arguments) == 1 else 'arguments'
            arguments = ', '.join(
                arg_name.replace('-', '_') for arg_name in missing_arguments)

            raise TypeError(
                f'{yaml_component.name}() missing {len(missing_arguments)} required '
                f'{argument_or_arguments}: {arguments}.')
        
        return task_inputs
    
    @classmethod
    def _add_volumes_and_mounts(
        cls, op: ResourceOpPipelineTask
    ) -> ResourceOpPipelineTask:
        return add_volumes_and_mounts(
            task=op,
            volumes=[
                V1Volume(
                    name=cls.RESOURCE_OP_OUTPUT_VOLUME_NAME,
                    ephemeral=V1EphemeralVolumeSource(
                        volume_claim_template=V1PersistentVolumeClaimTemplate(
                            spec=V1PersistentVolumeClaimSpec(
                                access_modes=['ReadWriteOnce'],
                            )
                        )
                    )
                ),
            ],
            volume_mounts=[
                V1VolumeMount(
                    mount_path=cls.RESOURCE_OP_OUTPUT_VOLUME_MOUNT_PATH,
                    name=cls.RESOURCE_OP_OUTPUT_VOLUME_NAME,
                ),
            ],
        )
    
    @classmethod
    def _inject_settings_to_resource_op(
        cls, op: ResourceOpPipelineTask
    ) -> ResourceOpPipelineTask:
        injections = [
            cls._add_volumes_and_mounts,
        ]
        for injection in injections:
            op = injection(op)
        return op
    
    @classmethod
    def _get_wrapped_op_factory(
        cls, 
        op_factory: YamlComponent
    ) -> Callable[..., ResourceOpPipelineTask]:
        @functools.wraps(op_factory)
        def wrapped_op_factory(
            *args,
            resource_name: str = '',
            resource_kind: str = '',
            **kwargs,
        ):
            component_spec = op_factory.component_spec
            inputs = cls.extract_task_inputs_from_yaml_component(
                yaml_component=op_factory,
                *args,
                **kwargs,
            )
            execute_locally = \
                pipeline_context.Pipeline.get_default_pipeline() is None
            op = cls.ResourceOpPipelineTask(
                component_spec=component_spec,
                args=inputs,
                execute_locally=execute_locally,
                resource_name=resource_name,
                resource_kind=resource_kind,
            )
            op = cls._inject_settings_to_resource_op(op)
            op = op.set_caching_options(cls.cache_enabled)
            return op
        return wrapped_op_factory
    
    def __new__(cls, 
        *args,
        resource_name: str = '',
        resource_kind: str = '',
        **kwargs,
    ) -> ResourceOpPipelineTask:
        return cls._to_op_factory()(
            *args,
            resource_name=resource_name,
            resource_kind=resource_kind,
            **kwargs
        )
