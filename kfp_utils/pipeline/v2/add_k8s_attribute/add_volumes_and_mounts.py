from typing import Optional

from kfp.dsl import PipelineTask
from kfp.kubernetes import (
    mount_pvc,
    add_ephemeral_volume,
    use_config_map_as_volume,
    use_secret_as_volume,
)
from kubernetes.client.models import (
    V1Volume,
    V1VolumeMount,
    V1ObjectMeta,
    V1PersistentVolumeClaimSpec,
)
from ..config import DEFAULT_EPHEMERAL_VOLUME_SIZE


def add_volumes_and_mounts(
    task: PipelineTask,
    volumes: list[V1Volume],
    volume_mounts: list[V1VolumeMount],
):
    name_to_volume_map = {
        volume.name: volume
        for volume in volumes
    }
    for volume_mount in volume_mounts:
        volume = name_to_volume_map.get(volume_mount.name)
        if volume is None:
            raise ValueError(f'No volume found for mount {volume_mount.name}')
        elif volume.persistent_volume_claim is not None:
            task = mount_pvc(
                task=task,
                pvc_name=volume.persistent_volume_claim.claim_name,
                mount_path=volume_mount.mount_path,
            )
        elif volume.ephemeral is not None and volume.ephemeral.volume_claim_template is not None:
            metadata: Optional[V1ObjectMeta] = volume.ephemeral.volume_claim_template.metadata
            spec: V1PersistentVolumeClaimSpec = volume.ephemeral.volume_claim_template.spec
            storage_size = (
                spec.resources.requests.get('storage')
                if (
                    spec.resources is not None 
                    and isinstance(spec.resources.requests, dict)
                )
                else DEFAULT_EPHEMERAL_VOLUME_SIZE
            )
            task = add_ephemeral_volume(
                task=task,
                volume_name=volume.name,
                mount_path=volume_mount.mount_path,
                access_modes=(spec.access_modes or []),
                size=storage_size,
                storage_class_name=spec.storage_class_name,
                **(
                    {
                        'labels': metadata.labels,
                        'annotations': metadata.annotations,
                    } if metadata is not None else {}
                )
            )
        elif volume.secret is not None:
            task = use_secret_as_volume(
                task=task,
                secret_name=volume.secret.secret_name,
                mount_path=volume_mount.mount_path,
            )
        elif volume.config_map is not None:
            task = use_config_map_as_volume(
                task=task,
                config_map_name=volume.config_map.name,
                mount_path=volume_mount.mount_path,
            )
        else:
            raise ValueError(f'Unsupported volume type for volume {volume.name}')
    return task
