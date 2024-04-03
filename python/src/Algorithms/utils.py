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

    def evaluate(self, instance: XHSTTSInstance) -> int:
        """
        returns the negative of the cost because 0 is the best cost.
        weights the infeasible cost 10 times the objective cost.
        TODO: choose better evaluation and investigate how it affects population
        """
        if self.needs_eval_update:
            self.cost = instance.evaluate_solution(self.sol_events)
            self.eval = -(
                10 * self.cost.Infeasibility_Value + self.cost.Objective_Value
            )
            self.needs_eval_update = False
        return self.eval
        # return -(self.cost.Infeasibility_Value + self.cost.Objective_Value)
        # return -self.cost.Infeasibility_Value

    def is_feasible(self) -> bool:
        return self.cost.Infeasibility_Value == 0

    def is_feasible_and_solves_objectives(self):
        return self.cost.Infeasibility_Value == 0 and self.cost.Objective_Value == 0


def swap(list_a, list_b, pos_a, pos_b):
    list_a[pos_a], list_b[pos_b] = list_b[pos_b], list_a[pos_a]


def swap_time_refs(solution: Solution, event_idx: int):
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
        new_event = swap_time_refs(solution, event_idx)
    else:
        new_time_reference = instance.get_random_time_reference()
        new_event = event._replace(TimeReference=new_time_reference)

    return new_event


def mutate_resource(instance: XHSTTSInstance, event: XHSTTSInstance):
    values = list(range(0, len(event.Resources)))
    random.shuffle(values)
    values_set = set(values)
    resource_mutated = False
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
    # decide between mutating the time or one of the resources
    rand_num = random.randint(0, len(new_event.Resources))
    if rand_num == len(new_event.Resources):
        # mutate time if not pre-assigned
        if not instance.Events[event.InstanceEventReference].PreAssignedTimeReference:
            new_event = mutate_time(instance, event, solution, i)
        else:
            # choose a non-preassigned resource
            _, new_event = mutate_resource(instance, event)
    else:
        # choose a non-preassigned resource
        resource_mutated, new_event = mutate_resource(instance, event)
        if not resource_mutated:
            # mutate time
            new_event = mutate_time(instance, event, solution, i)
    solution.sol_events[i] = new_event


def neighbor(solution: Solution, instance: XHSTTSInstance) -> Solution:
    # solution should not be modified!

    new_solution = Solution(deepcopy(solution.sol_events))
    idx = random.randint(0, len(new_solution.sol_events) - 1)
    event = new_solution.sol_events[idx]

    new_event = event
    solution.needs_eval_update = True
    # decide between mutating the time or one of the resources
    rand_num = random.randint(0, len(new_event.Resources))
    if rand_num == len(new_event.Resources):
        # mutate time if not pre-assigned
        if not instance.Events[event.InstanceEventReference].PreAssignedTimeReference:
            new_event = mutate_time(
                instance, event, new_solution, idx, swap_percentage=0.5
            )
        else:
            # choose a non-preassigned resource
            _, new_event = mutate_resource(instance, event)
    else:
        # choose a non-preassigned resource
        resource_mutated, new_event = mutate_resource(instance, event)
        if not resource_mutated:
            # mutate time
            new_event = mutate_time(
                instance, event, new_solution, idx, swap_percentage=0.5
            )

    new_solution.sol_events[idx] = new_event

    return new_solution
