from collections import defaultdict
from copy import deepcopy
import random

from XHSTTS.utils import Cost
from XHSTTS.xhstts import XHSTTSInstance


class Solution:
    def __init__(self, sol_events: list[XHSTTSInstance.SolutionEvent]):
        self.sol_events = sol_events
        self.cost: Cost = None
        self.eval: int = None
        self.needs_eval_update: bool = True
        self.k = 10
        self.original_events: dict[str, set[int]] = defaultdict(set)
        self._parse_sol_events()

    def evaluate(self, instance: XHSTTSInstance) -> int:
        """
        returns the negative of the cost because 0 is the best cost.
        weights the infeasible cost 10 times the objective cost.
        TODO: choose better evaluation and investigate how it affects population
        """
        if self.needs_eval_update:
            self.cost = instance.evaluate_solution(self.sol_events)
            self.eval = -(
                self.k * self.cost.Infeasibility_Value + self.cost.Objective_Value
            )
            self.needs_eval_update = False
        return self.eval
        # return -(self.cost.Infeasibility_Value + self.cost.Objective_Value)
        # return -self.cost.Infeasibility_Value

    def is_feasible(self) -> bool:
        return self.cost.Infeasibility_Value == 0

    def is_feasible_and_solves_objectives(self):
        return self.cost.Infeasibility_Value == 0 and self.cost.Objective_Value == 0

    def _parse_sol_events(self):
        for idx, ev in enumerate(self.sol_events):
            self.original_events[ev.InstanceEventReference].add(idx)

    def recalculate_indices(self):
        self.original_events = defaultdict(set)
        self._parse_sol_events()


def swap(list_a, list_b, pos_a, pos_b):
    list_a[pos_a], list_b[pos_b] = list_b[pos_b], list_a[pos_a]


def swap_time_refs(solution: Solution, instance: XHSTTSInstance, event_idx: int):
    other_idx = random.randint(0, len(solution.sol_events) - 1)
    while (
        instance.Events[
            solution.sol_events[other_idx].InstanceEventReference
        ].PreAssignedTimeReference
        is not None
    ):
        other_idx = random.randint(0, len(solution.sol_events) - 1)

    tmp_time_ref = solution.sol_events[event_idx].TimeReference
    solution.sol_events[event_idx] = solution.sol_events[event_idx]._replace(
        TimeReference=solution.sol_events[other_idx].TimeReference
    )
    solution.sol_events[other_idx] = solution.sol_events[other_idx]._replace(
        TimeReference=tmp_time_ref
    )

    return solution.sol_events[event_idx]


def mutate_time(
    instance: XHSTTSInstance,
    event: XHSTTSInstance.SolutionEvent,
    solution: Solution,
    event_idx: int,
    swap_percentage: float = 0.5,
):
    new_event = event
    if random.random() < swap_percentage:
        new_event = swap_time_refs(solution, instance, event_idx)
    else:
        new_time_reference = instance.get_random_time_reference()
        new_event = event._replace(TimeReference=new_time_reference)

    return new_event


def swap_resources(
    solution: Solution, instance: XHSTTSInstance, event: XHSTTSInstance.SolutionEvent
):
    values = list(range(0, len(event.Resources)))
    values_set = set(values)
    resource_mutated = False
    for _ in range(len(event.Resources)):
        resource_to_change_idx = values_set.pop()

        idx, resourceType = instance.find_resource_type(
            instance.Events[event.InstanceEventReference].Resources,
            event.Resources[resource_to_change_idx].Role,
        )

        if not instance.Events[event.InstanceEventReference].Resources[idx].Reference:
            print(instance.resource_swap_partition[resourceType])
            other_event_ref = random.choice(
                list(instance.resource_swap_partition[resourceType])
            )
            sol_event_idx = random.choice(
                list(solution.original_events[other_event_ref])
            )
            other_sol_event = solution.sol_events[sol_event_idx]
            for other_resource_index, other_resource in enumerate(
                other_sol_event.Resources
            ):
                if (
                    instance.find_resource_type(
                        instance.Events[
                            other_sol_event.InstanceEventReference
                        ].Resources,
                        other_resource.Role,
                    )
                    == resourceType
                ):
                    swap(
                        event.Resources,
                        other_sol_event.Resources,
                        resource_to_change_idx,
                        other_resource_index,
                    )
                    break
            resource_mutated = True
            break

    return resource_mutated, event


def mutate_resource(
    solution: Solution,
    instance: XHSTTSInstance,
    event: XHSTTSInstance.SolutionEvent,
    swap_percentage=0.5,
):
    resource_mutated = False
    if random.random() < swap_percentage:
        resource_mutated, event = swap_resources(solution, instance, event)
    else:
        values = list(range(0, len(event.Resources)))
        random.shuffle(values)
        values_set = set(values)

        for _ in range(len(event.Resources)):
            resource_to_change_idx = values_set.pop()
            is_preassigned, new_event_resource = (
                instance.get_random_and_valid_resource_reference(
                    event.Resources[resource_to_change_idx],
                    event.InstanceEventReference,
                )
            )
            if not is_preassigned:
                event.Resources[resource_to_change_idx] = new_event_resource
                resource_mutated = True
                break

    return resource_mutated, event


def mutate(solution: Solution, instance: XHSTTSInstance) -> None:

    i = random.randint(0, len(solution.sol_events) - 1)
    event = solution.sol_events[i]

    # randomly mutate an event
    new_event = event
    solution.needs_eval_update = True

    # decide between mutating time, mutating resource, splitting an event into two or merging two events
    rand_num = random.randint(1, 4)
    if rand_num == 1 or rand_num == 2:
        if random.random() > 0.5:
            # mutate time if not pre-assigned
            if not instance.Events[
                event.InstanceEventReference
            ].PreAssignedTimeReference:
                new_event = mutate_time(instance, event, solution, i)
            else:
                # choose a non-preassigned resource
                _, new_event = mutate_resource(solution, instance, event)
        else:
            # choose a non-preassigned resource
            resource_mutated, new_event = mutate_resource(solution, instance, event)
            if not resource_mutated:
                # mutate time
                new_event = mutate_time(instance, event, solution, i)
        solution.sol_events[i] = new_event

    elif rand_num == 3:
        # split event
        split_event_idx = random.choice(list(range(0, len(solution.sol_events))))
        event_to_split = solution.sol_events[split_event_idx]
        if (
            event_to_split.Duration > 1
            and event_to_split.SplitMinDuration < event_to_split.Duration
            and event_to_split.SplitMaxAmount
            > len(solution.original_events[event_to_split.InstanceEventReference])
        ):
            split_event(solution, instance, split_event_idx)

    elif rand_num == 4:
        # merge two events
        merge_event_idx = random.choice(list(range(0, len(solution.sol_events))))
        event_to_merge = solution.sol_events[merge_event_idx]
        if (not event_to_merge.IsOriginal) and (
            len(solution.original_events[event_to_merge.InstanceEventReference]) > 1
        ):
            merge_event(solution, merge_event_idx)


def neighbor(solution: Solution, instance: XHSTTSInstance) -> Solution:
    # solution should not be modified!

    new_solution = Solution(deepcopy(solution.sol_events))
    new_solution.k = solution.k
    idx = random.randint(0, len(new_solution.sol_events) - 1)
    event = new_solution.sol_events[idx]

    new_event = event
    solution.needs_eval_update = True

    # decide between mutating time, mutating resource, splitting an event into two or merging two events
    rand_num = random.randint(1, 4)
    if rand_num == 1 or rand_num == 2:
        if random.random() > 0.5:
            # mutate time if not pre-assigned
            if not instance.Events[
                event.InstanceEventReference
            ].PreAssignedTimeReference:
                new_event = mutate_time(
                    instance, event, new_solution, idx, swap_percentage=0.5
                )
            else:
                # choose a non-preassigned resource
                _, new_event = mutate_resource(instance, event)
        else:
            # choose a non-preassigned resource
            resource_mutated, new_event = mutate_resource(solution, instance, event)
            if not resource_mutated:
                # mutate time
                new_event = mutate_time(
                    instance, event, new_solution, idx, swap_percentage=0.5
                )
        new_solution.sol_events[idx] = new_event

    elif rand_num == 3:
        # split event
        split_event_idx = random.choice(list(range(0, len(new_solution.sol_events))))
        event_to_split = new_solution.sol_events[split_event_idx]
        if (
            event_to_split.Duration > 1
            and event_to_split.SplitMinDuration < event_to_split.Duration
            and event_to_split.SplitMaxAmount
            > len(new_solution.original_events[event_to_split.InstanceEventReference])
        ):
            split_event(new_solution, instance, split_event_idx)

    elif rand_num == 4:
        # merge two events
        merge_event_idx = random.choice(list(range(0, len(new_solution.sol_events))))
        event_to_merge = new_solution.sol_events[merge_event_idx]
        if (not event_to_merge.IsOriginal) and (
            len(new_solution.original_events[event_to_merge.InstanceEventReference]) > 1
        ):
            merge_event(new_solution, merge_event_idx)

    return new_solution


def split_event(
    solution: Solution,
    instance: XHSTTSInstance,
    idx,
):
    """
    Splits an event with a duration >= 2 into events with a different time but the same resources
    """
    event: XHSTTSInstance.SolutionEvent = solution.sol_events[idx]
    assert event.Duration > 1, f"event duration must be > 1, got {event.Duration}"
    assert (
        event.SplitMinDuration < event.Duration
    ), f"event duration must be < event split min duration"
    assert event.SplitMaxAmount > len(
        solution.original_events[event.InstanceEventReference]
    ), f"current number of splits must be < max number of splits"

    # remove the event from the solution events list
    del solution.sol_events[idx]

    # remove the event from the set of events for that instance event
    solution.original_events[event.InstanceEventReference].remove(idx)

    # split event
    first_event_duration = int(event.Duration // 2)
    second_event_duration = event.Duration - first_event_duration

    for duration in (first_event_duration, second_event_duration):
        child_event = XHSTTSInstance.SolutionEvent(
            InstanceEventReference=event.InstanceEventReference,
            Duration=duration,
            TimeReference=instance.get_random_time_reference(),
            Resources=event.Resources,
            SplitMaxAmount=event.SplitMaxAmount,
            SplitMinAmount=event.SplitMinAmount,
            SplitMinDuration=event.SplitMinDuration,
            SplitMaxDuration=event.SplitMaxDuration,
            IsOriginal=False,
        )
        solution.original_events[event.InstanceEventReference].add(
            len(solution.sol_events)
        )
        solution.sol_events.append(child_event)

    solution.recalculate_indices()

    return solution


def merge_event(solution: Solution, idx):
    """
    Merges two sub events to events with a different time but the resources of the first event
    """

    event: XHSTTSInstance.SolutionEvent = solution.sol_events[idx]
    assert (not event.IsOriginal) and (
        len(solution.original_events[event.InstanceEventReference]) > 1
    ), "only sub events can be merged"

    # remove the event from the set of events for that instance event
    solution.original_events[event.InstanceEventReference].remove(idx)

    # remove a second event to merge with
    second_idx = solution.original_events[event.InstanceEventReference].pop()

    second_event = solution.sol_events[second_idx]

    # remove the event from the solution events list
    del solution.sol_events[idx]

    # remove the second event solution events from the list
    if idx < second_idx:
        del solution.sol_events[second_idx - 1]
    else:
        del solution.sol_events[second_idx]

    # merge events
    merged_event = XHSTTSInstance.SolutionEvent(
        InstanceEventReference=event.InstanceEventReference,
        Duration=event.Duration + second_event.Duration,
        TimeReference=event.TimeReference,
        Resources=event.Resources,
        SplitMaxAmount=event.SplitMaxAmount,
        SplitMinAmount=event.SplitMinAmount,
        SplitMinDuration=event.SplitMinDuration,
        SplitMaxDuration=event.SplitMaxDuration,
        IsOriginal=False,
    )

    solution.original_events[event.InstanceEventReference].add(len(solution.sol_events))
    solution.sol_events.append(merged_event)

    solution.recalculate_indices()

    return solution


def kempe():
    pass
