import os
import sys
import tempfile
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from typing import Generator, Any, NamedTuple, Callable
from types import ModuleType

from kfp.dsl import PipelineTask

from .ops import PythonFunctionOp


@contextmanager
def register_temporary_module_from_str(fn_str: str) -> Generator[ModuleType, Any, None]:
    with tempfile.NamedTemporaryFile('w', suffix='.py') as tmp_file:
        tmp_file_path = tmp_file.name
        tmp_file.write(fn_str)
        tmp_file.seek(0)
        module_name = os.path.basename(tmp_file_path).split('.')[0]
        spec = spec_from_file_location(module_name, tmp_file_path)
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        try:
            yield module
        finally:
            pass


class TemplateFillArg(NamedTuple):
    arg_name: str
    arg_type: type


class TemplateFillOpFactory:
    def __new__(
        cls, *, 
        template: str,
        fn_name: str,
        fn_args: list[TemplateFillArg]
    ) -> Callable[..., PipelineTask]:
        fn_args_repr = ', '.join([
            f'{arg_name}: {arg_type.__name__}'
            for arg_name, arg_type in fn_args
        ])
        fn_str = f'''\
def {fn_name}({fn_args_repr}) -> str:
    return f"""{template}"""
'''
        with register_temporary_module_from_str(fn_str) as module:
            class FillOp(PythonFunctionOp):
                function = getattr(module, fn_name)
            fill_op_factory = FillOp._to_op_factory()
        return fill_op_factory
