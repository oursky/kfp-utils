from typing import Optional

from google.protobuf import json_format
from kubernetes.client.models import (
    V1Affinity,
    V1NodeAffinity,
    V1NodeSelector,
    V1NodeSelectorTerm,
    V1PodAffinity,
    V1PodAntiAffinity,
    V1PodAffinityTerm,
    V1LabelSelector,
    V1WeightedPodAffinityTerm,
    V1NodeSelectorRequirement,
    V1PreferredSchedulingTerm,
)
from kfp.dsl import PipelineTask
from kfp.kubernetes.common import get_existing_kubernetes_config_as_message
from kfp.kubernetes import kubernetes_executor_config_pb2 as pb


def convert_requirements(
    requirements: Optional[list[V1NodeSelectorRequirement]]
) -> list[pb.SelectorRequirement] | None:
    if requirements is None:
        return None
    return [
        pb.SelectorRequirement(
            key=requirement.key,
            operator=requirement.operator,
            values=requirement.values,
        )
        for requirement in requirements
    ]


def convert_label_selector(
    selector: Optional[list[V1LabelSelector]]
) -> tuple[list[pb.SelectorRequirement], dict[str, str]]:
    if selector is None:
        return [], {}
    match_exps = convert_requirements(selector.match_expressions)
    match_labels = selector.match_labels or {}
    return match_exps, match_labels


def convert_node_selector_term(
    term: V1NodeSelectorTerm,
    weight: Optional[int] =  None
) -> pb.NodeAffinityTerm:
    match_expressions = convert_requirements(term.match_expressions)
    match_fields = convert_requirements(term.match_fields)
    return pb.NodeAffinityTerm(
        match_expressions=match_expressions,
        match_fields=match_fields,
        weight=weight,
    )


def convert_pod_affinity_term(
    term: V1PodAffinityTerm,
    weight: Optional[int] = None,
    anti: bool = False,
) -> pb.PodAffinityTerm:
    pod_exps, pod_labels = \
        convert_label_selector(term.label_selector)
    namespace_exps, namespace_labels = \
        convert_label_selector(term.namespace_selector)
    return pb.PodAffinityTerm(
        match_pod_expressions=pod_exps,
        match_pod_labels=pod_labels,
        topology_key=term.topology_key,
        namespaces=term.namespaces,
        match_namespace_expressions=namespace_exps,
        match_namespace_labels=namespace_labels,
        weight=weight,
        anti=anti,
    )


def add_node_affinity(
    task: PipelineTask, 
    node_affinity: Optional[V1NodeAffinity]
) -> PipelineTask:
    if node_affinity is None:
        return task
    msg = get_existing_kubernetes_config_as_message(task)
    required: Optional[V1NodeSelector] = \
        node_affinity.required_during_scheduling_ignored_during_execution
    if required is not None:
        node_selector_terms: list[V1NodeSelectorTerm] = \
            required.node_selector_terms
        for term in node_selector_terms:
            msg.node_affinity.append(
                convert_node_selector_term(term)
            )
    preferred: Optional[list[V1PreferredSchedulingTerm]] = \
        node_affinity.preferred_during_scheduling_ignored_during_execution
    if preferred is not None:
        for term in preferred:
            msg.node_affinity.append(
                convert_node_selector_term(term.preference, term.weight)
            )
    task.platform_config["kubernetes"] = json_format.MessageToDict(msg)
    return task


def add_pod_affinity(
    task: PipelineTask,
    affinity: Optional[V1PodAffinity],
) -> PipelineTask:
    if affinity is None:
        return task
    msg = get_existing_kubernetes_config_as_message(task)
    required: Optional[list[V1PodAffinityTerm]] = \
        affinity.required_during_scheduling_ignored_during_execution
    if required is not None:
        for term in required:
            msg.pod_affinity.append(
                convert_pod_affinity_term(term)
            )
    preferred: Optional[list[V1WeightedPodAffinityTerm]] = \
        affinity.preferred_during_scheduling_ignored_during_execution
    if preferred is not None:
        for weighted_term in preferred:
            weight = weighted_term.weight
            term = weighted_term.pod_affinity_term
            msg.pod_affinity.append(
                convert_pod_affinity_term(term, weight=weight)
            )
    task.platform_config["kubernetes"] = json_format.MessageToDict(msg)
    return task


def add_pod_anti_affinity(
    task: PipelineTask,
    affinity: Optional[V1PodAntiAffinity],
) -> PipelineTask:
    if affinity is None:
        return task
    msg = get_existing_kubernetes_config_as_message(task)
    required: Optional[list[V1PodAffinityTerm]] = \
        affinity.required_during_scheduling_ignored_during_execution
    if required is not None:
        for term in required:
            msg.pod_affinity.append(
                convert_pod_affinity_term(term, anti=True)
            )
    preferred: Optional[list[V1WeightedPodAffinityTerm]] = \
        affinity.preferred_during_scheduling_ignored_during_execution
    if preferred is not None:
        for weighted_term in preferred:
            weight = weighted_term.weight
            term = weighted_term.pod_affinity_term
            msg.pod_affinity.append(
                convert_pod_affinity_term(term, weight=weight, anti=True)
            )
    task.platform_config["kubernetes"] = json_format.MessageToDict(msg)
    return task


def add_affinity(task: PipelineTask, affinity: V1Affinity):
    task = add_node_affinity(task, affinity.node_affinity)
    task = add_pod_affinity(task, affinity.pod_affinity)
    task = add_pod_anti_affinity(task, affinity.pod_anti_affinity)
    return task
