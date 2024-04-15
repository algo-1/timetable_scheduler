from collections import defaultdict, deque
from copy import deepcopy
from enum import Enum, auto
import itertools
import random

from XHSTTS.utils import Cost, Mode
from XHSTTS.xhstts import XHSTTSInstance


class Solution:
    def __init__(self, sol_events: list[XHSTTSInstance.SolutionEvent]):
        self.sol_events = sol_events
        self.cost: Cost = None
        self.eval: int = None
        self.needs_eval_update: bool = True
        self.k = 1000
        self.mode = Mode.Hard
        self.original_events: dict[str, set[int]] = defaultdict(set)
        self._parse_sol_events()

    def evaluate(self, instance: XHSTTSInstance) -> int:
        """
        returns the negative of the cost because 0 is the best cost.
        weights the infeasible cost 10 times the objective cost.
        TODO: choose better evaluation and investigate how it affects population
        """
        if self.needs_eval_update:
            if self.mode == Mode.Hard:
                self.cost = instance.evaluate_solution(self.sol_events, mode=Mode.Hard)
                self.eval = -self.cost.Infeasibility_Value
            else:
                # make it impossible to accept hard cost violations, perhaps use epsilon here as may need to increaase first?
                self.cost = instance.evaluate_solution(self.sol_events, mode=Mode.Soft)
                if self.cost.Infeasibility_Value > 0:
                    self.eval = -float("inf")
                else:
                    self.eval = -self.cost.Objective_Value

            self.needs_eval_update = False
        return self.eval
        # return -(self.cost.Infeasibility_Value + self.cost.Objective_Value)
        # return -self.cost.Infeasibility_Value

    def is_feasible(self) -> bool:
        return self.cost.Infeasibility_Value == 0

    def is_feasible_and_solves_objectives(self):
        return (
            self.mode == Mode.Soft
            and self.cost.Infeasibility_Value == 0
            and self.cost.Objective_Value == 0
        )

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
        new_time_reference = (
            random.choice(list(event.PreferredTimes))
            if event.PreferredTimes
            else instance.get_random_time_reference()
        )
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


def mutate(
    solution: Solution,
    instance: XHSTTSInstance,
    merge_split_move=True,
) -> None:

    i = random.randint(0, len(solution.sol_events) - 1)
    event = solution.sol_events[i]

    # randomly mutate an event
    new_event = event
    solution.needs_eval_update = True

    # decide between mutating time, mutating resource, splitting an event into two or merging two events
    rand_num = random.randint(1, 10) if merge_split_move else random.randint(1, 5)
    if rand_num == 1 or rand_num == 2 or rand_num == 3 or rand_num == 4:
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

    elif rand_num == 6 or rand_num == 7 or rand_num == 8:
        # split event
        split_event_idx = random.choice(list(range(0, len(solution.sol_events))))
        event_to_split = solution.sol_events[split_event_idx]
        if (
            not instance.Events[
                event_to_split.InstanceEventReference
            ].PreAssignedTimeReference
            and event_to_split.Duration > 1
            and event_to_split.SplitMinDuration < event_to_split.Duration
            and event_to_split.SplitMaxAmount
            > len(solution.original_events[event_to_split.InstanceEventReference])
        ):
            split_event(solution, instance, split_event_idx)

    elif rand_num == 9 or rand_num == 10:
        # merge two events
        merge_event_idx = random.choice(list(range(0, len(solution.sol_events))))
        event_to_merge = solution.sol_events[merge_event_idx]
        if (not event_to_merge.IsOriginal) and (
            len(solution.original_events[event_to_merge.InstanceEventReference]) > 1
        ):
            merge_event(solution, merge_event_idx)

    elif rand_num == 5:
        # kempe move
        kempe_move(solution, instance, double=True)


def neighbor(
    solution: Solution,
    instance: XHSTTSInstance,
    merge_split_move=True,
) -> Solution:
    # solution should not be modified!

    new_solution = Solution(deepcopy(solution.sol_events))
    new_solution.mode = solution.mode
    idx = random.randint(0, len(new_solution.sol_events) - 1)
    event = new_solution.sol_events[idx]

    new_event = event
    solution.needs_eval_update = True

    # decide between mutating time, mutating resource, splitting an event into two or merging two events
    rand_num = random.randint(1, 10) if merge_split_move else random.randint(1, 5)
    if rand_num == 1 or rand_num == 2 or rand_num == 3 or rand_num == 4:
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
                _, new_event = mutate_resource(solution, instance, event)
        else:
            # choose a non-preassigned resource
            resource_mutated, new_event = mutate_resource(solution, instance, event)
            if not resource_mutated:
                # mutate time
                new_event = mutate_time(
                    instance, event, new_solution, idx, swap_percentage=0.5
                )
        new_solution.sol_events[idx] = new_event

    elif rand_num == 6 or rand_num == 7 or rand_num == 8:
        # split event
        split_event_idx = random.choice(list(range(0, len(new_solution.sol_events))))
        event_to_split = new_solution.sol_events[split_event_idx]

        if (
            not instance.Events[
                event_to_split.InstanceEventReference
            ].PreAssignedTimeReference
            and event_to_split.Duration > 1
            and event_to_split.SplitMinDuration < event_to_split.Duration
            and event_to_split.SplitMaxAmount
            > len(new_solution.original_events[event_to_split.InstanceEventReference])
        ):
            split_event(new_solution, instance, split_event_idx)

    elif rand_num == 9 or rand_num == 10:
        # merge two events
        merge_event_idx = random.choice(list(range(0, len(new_solution.sol_events))))
        event_to_merge = new_solution.sol_events[merge_event_idx]
        if (not event_to_merge.IsOriginal) and (
            len(new_solution.original_events[event_to_merge.InstanceEventReference]) > 1
        ):
            merge_event(new_solution, merge_event_idx)

    elif rand_num == 5:
        # kempe move
        kempe_move(new_solution, instance, double=True)

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
    assert not instance.Events[
        event.InstanceEventReference
    ].PreAssignedTimeReference, "cannot split event with a pre-assigned time"
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
            TimeReference=(
                random.choice(list(event.PreferredTimes))
                if event.PreferredTimes
                else instance.get_random_time_reference()
            ),
            Resources=event.Resources,
            SplitMaxAmount=event.SplitMaxAmount,
            SplitMinAmount=event.SplitMinAmount,
            SplitMinDuration=event.SplitMinDuration,
            SplitMaxDuration=event.SplitMaxDuration,
            PreferredTimes=event.PreferredTimes,
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
        PreferredTimes=event.PreferredTimes,
        IsOriginal=False,
    )

    solution.original_events[event.InstanceEventReference].add(len(solution.sol_events))
    solution.sol_events.append(merged_event)

    solution.recalculate_indices()

    return solution


def has_common_resource(
    sol_event_a: XHSTTSInstance.SolutionEvent, sol_event_b: XHSTTSInstance.SolutionEvent
):
    for resource_a in sol_event_a.Resources:
        for resource_b in sol_event_b.Resources:
            if resource_a.Reference == resource_b.Reference:
                return True
    return False


def get_connected_components(U, edges):
    chains = []
    visited = set()
    for u in U:
        if u not in visited:
            chain = set()
            stack = deque([u])
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    chain.add(node)
                    stack.extend(edges[node])
            chains.append(chain)

    return chains


def kempe_move(solution: Solution, instance: XHSTTSInstance, double=False):
    sol_ev1_idx = random.choice(range(len(solution.sol_events)))
    sol_ev1 = solution.sol_events[sol_ev1_idx]

    # ensure not pre-assigned, we can loop because all instances have assign times constraints
    while (
        instance.Events[sol_ev1.InstanceEventReference].PreAssignedTimeReference
        is not None
    ):
        sol_ev1_idx = random.choice(range(len(solution.sol_events)))
        sol_ev1 = solution.sol_events[sol_ev1_idx]

    sol_ev2_idx = random.choice(range(len(solution.sol_events)))
    sol_ev2 = solution.sol_events[sol_ev2_idx]

    # ensure times are different and not pre-assigned
    while (
        sol_ev1.TimeReference == sol_ev2.TimeReference
        and instance.Events[sol_ev2.InstanceEventReference].PreAssignedTimeReference
        is not None
    ):
        sol_ev2_idx = random.choice(range(len(solution.sol_events)))
        sol_ev2 = solution.sol_events[sol_ev2_idx]

    # construct bipartite graph where members in the same set have the same time reference and consist of all non-preassigned events that are assigned that time (note this is only taking starting times into account)
    # The graph is a conflict graph as edges are from elems of U to V such that u and v share a resource.
    U = set([sol_ev1_idx])
    V = set([sol_ev2_idx])
    edges = defaultdict(list)

    # add nodes
    for idx, sol_event in enumerate(solution.sol_events):
        if (
            sol_event.TimeReference == sol_ev1.TimeReference
            and not instance.Events[
                sol_event.InstanceEventReference
            ].PreAssignedTimeReference
            and sol_event != sol_ev1
        ):
            U.add(idx)
        elif (
            sol_event.TimeReference == sol_ev2.TimeReference
            and not instance.Events[
                sol_event.InstanceEventReference
            ].PreAssignedTimeReference
            and sol_event != sol_ev2
        ):
            V.add(idx)

    # add edges
    for u in U:
        for v in V:
            if has_common_resource(solution.sol_events[u], solution.sol_events[v]):
                edges[u].append(v)
                edges[v].append(u)

    # Identify connected components (chains) in the bipartite graph
    chains = get_connected_components(U, edges)

    chain_to_swap = None
    if double:
        # select 2 chains to swap if possible
        if len(chains) > 1:
            list_of_chains_to_swap = random.sample(chains, k=2)

            # flatten the 2d list
            chain_to_swap = list(itertools.chain.from_iterable(list_of_chains_to_swap))
        else:
            chain_to_swap = random.choice(chains)
    else:
        # Select a chain to swap
        chain_to_swap = random.choice(chains)

    # Perform swap within the selected chain
    for sol_event_idx in chain_to_swap:
        if sol_event_idx in U:
            solution.sol_events[sol_event_idx] = solution.sol_events[
                sol_event_idx
            ]._replace(TimeReference=sol_ev2.TimeReference)
        else:
            solution.sol_events[sol_event_idx] = solution.sol_events[
                sol_event_idx
            ]._replace(TimeReference=sol_ev1.TimeReference)

    return solution


def get_consecutive_times(instance: XHSTTSInstance, start_time_ref, duration):
    time_ref_idx = instance.instance_time_refs_indices[start_time_ref] + 1

    time_refs = [start_time_ref]

    for _ in range(duration - 1):
        if time_ref_idx < len(instance.instance_time_refs_indices_list):
            time_refs.append(instance.instance_time_refs_indices_list[time_ref_idx][0])
            time_ref_idx += 1
    # print(duration, time_refs)
    return time_refs


def ejection_chains(solution: Solution, instance: XHSTTSInstance):
    # used for repairing in-feasible solutions
    # when you modify a solution you better soecify that it needs eval update!
    # TODO: we need to analyse a solution and return a list of (constraint, events involved in defect)
    # we queue the defects and pop, repair, analyse, update queue until queue empty or limit reached, we always keep best solution stored and return that.
    # then depending on the defect repair & analyse again etc until calm
    # make hard & soft phases explicit so when hard is done ejection chains are called if solution is infeasible to try to find a feasible solution, set a limit on the repair and that it cannot get worse than the solution that came into it.
    # pass it into the soft constraints optimisation phase
    # initialisation splits
    # prefer times fields - required, or maybe field_all ?? same for splits & distribute splits gen partitions where senssible (min(duration, max amount)not a large number, assume max duration is never large)
    # links
    pass
