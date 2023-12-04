# TODO - validate solution - must obey XHSTTS rules

# we want to generate solution events where we make asignments where necessary without caring about constraints - except assign constraints since we want to generate a solution event
#
# randomly assign time resources to all instance events where necessary in accordance to the rules (not constraints)
# split some events into smaller pieces randomly too in accordance to the rules (not constraints)
from collections import defaultdict
import random

from XHSTTS.xhstts import XHSTTS, XHSTTSInstance


def get_n_random_events_to_split(n: int, instance_events: list[XHSTTSInstance.Event]):
    """
    returns a tuple of n random solution events and
    (len(instance_events) - n) solution events both based on the instance events.
    """
    # TODO: make a set when you rewrite named tuples as frozen? / eq set to false dataclasses as we can have a set of objects but not a tuple that has a list element
    events_to_split = random.sample(instance_events, n)
    remaining_events = [
        event for event in instance_events if event not in events_to_split
    ]

    sol_events_to_split = list(
        map(XHSTTSInstance.create_solution_event, events_to_split)
    )
    remaining_sol_events = list(
        map(XHSTTSInstance.create_solution_event, remaining_events)
    )

    return sol_events_to_split, remaining_sol_events


def random_split(sol_events: list[XHSTTSInstance.SolutionEvent], max_split=6):
    """returns a list of solution events by randomly splitting instance events according to the rules (duration etc)"""
    split_sol_events = []
    for event in sol_events:
        if max_split > 0 and event.Duration > 1:
            num_splits = min(
                max_split, event.Duration, random.randint(2, event.Duration)
            )
            split_duration = event.Duration // num_splits

            for _ in range(num_splits - 1):
                split_event = event._replace(Duration=split_duration)
                split_sol_events.append(split_event)
            last_split_event = event._replace(
                Duration=event.Duration - (split_duration * (num_splits - 1))
            )
            split_sol_events.append(last_split_event)
        else:
            # no split necesary
            split_sol_events.append(event)
    return split_sol_events


def assign_random_times(
    sol_events: list[XHSTTSInstance.SolutionEvent], instance: XHSTTSInstance
):
    time_refs = list(instance.Times.keys())
    return [
        event
        if event.TimeReference
        else event._replace(TimeReference=random.choice(time_refs))
        for event in sol_events
    ]


def find_resource_type(resources: list[XHSTTSInstance.EventResource], role: str):
    for elem in resources:
        if elem.Role == role:
            return elem.ResourceTypeReference
    raise Exception(f"Did not find the '{role}' role in resources.")


def assign_random_resources(
    sol_events: list[XHSTTSInstance.SolutionEvent], instance: XHSTTSInstance
):
    # TODO: do this partitioning in the instance not here
    partitioned_resources_refs = defaultdict(list)
    for ref, resource in instance.Resources.items():
        partitioned_resources_refs[resource.ResourceTypeReference].append(ref)
    new_sol_events = []
    for event in sol_events:
        resources = [
            resource
            if resource.Reference
            else resource._replace(
                Reference=random.choice(
                    partitioned_resources_refs[
                        find_resource_type(
                            instance.Events[event.InstanceEventReference].Resources,
                            resource.Role,
                        )
                    ]
                )
            )
            for resource in event.Resources
        ]
        new_sol_events.append(event._replace(Resources=resources))

    return new_sol_events


def random_solution(instance: XHSTTSInstance) -> list[XHSTTSInstance.SolutionEvent]:
    instance_events = instance.Events

    n_events, rest = get_n_random_events_to_split(
        len(instance_events) // 3, list(instance_events.values())
    )  # use grid search cv? to find the best number to split ? or better way to decide if we split if constraints not telling us to split.Can splitting yield a solution when non splitting cannot?  seems that this should be the case!

    # n_events and rest should be new objects not point to the same ones as events - sol event objects preferably!
    sol_events = [sol_event for sol_event in rest]

    # split some events
    sol_events.extend(random_split(n_events))

    # assign time and resources randomly anywhere necessary
    result: list[XHSTTSInstance.SolutionEvent] = assign_random_resources(
        assign_random_times(sol_events, instance), instance
    )

    return result


if __name__ == "__main__":
    from pathlib import Path

    root_dir = Path(__file__).parent.parent.parent.parent
    data_dir = root_dir.joinpath("data/ALL_INSTANCES")

    dataset_sudoku4x4 = XHSTTS(data_dir.joinpath("ArtificialSudoku4x4.xml"))
    dataset_abramson15 = XHSTTS(data_dir.joinpath("ArtificialAbramson15.xml"))
    dataset_brazil3 = XHSTTS(data_dir.joinpath("BrazilInstance3.xml"))

    for dataset in (
        dataset_brazil3,
    ):  # (dataset_sudoku4x4, dataset_abramson15, dataset_brazil1):
        random.seed(23)

        assert dataset.num_instances() == 1

        instance = dataset.get_instance(index=0)

        # get solution
        result = random_solution(instance)

        # evaluate
        evaluation = instance.evaluate_solution(result)

        print("\n---Random Evaluation---\n", evaluation)
        print("---Random Evaluation---\n\n")
