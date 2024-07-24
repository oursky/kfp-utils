from kfp.dsl import PipelineTask

def add_resource_requests(
    task: PipelineTask,
    resource_requests: dict[str, str],
) -> PipelineTask:
    for key, value in resource_requests.items():
        if key == 'cpu':
            task.set_cpu_request(value)
        elif key == 'memory':
            task.set_memory_request(value)
        else:
            raise ValueError(f'Unsupported resource request type {key}')
    return task
