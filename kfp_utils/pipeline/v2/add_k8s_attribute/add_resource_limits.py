from kfp.dsl import PipelineTask

def add_resource_limits(
    task: PipelineTask,
    resource_limits: dict[str, str],
) -> PipelineTask:
    for key, value in resource_limits.items():
        if key == 'cpu':
            task.set_cpu_limit(value)
        elif key == 'memory':
            task.set_memory_limit(value)
        else:
            task.set_accelerator_type(key).set_accelerator_limit(value)
    return task
