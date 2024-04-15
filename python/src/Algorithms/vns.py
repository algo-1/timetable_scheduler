from collections import defaultdict
from copy import deepcopy
import itertools
import random
import time
from Algorithms.genetic2 import genetic_algorithm
from Algorithms.local_search import local_search
from Algorithms.random_algorithm import random_solution
from Algorithms.simulated_annealing import simulated_annealing
from Algorithms.tabu_search import ils_tabu
from Algorithms.utils import (
    Solution,
    get_connected_components,
    has_common_resource,
    swap,
)
from XHSTTS.utils import Mode
from XHSTTS.xhstts import XHSTTS, XHSTTSInstance


def change_time(sol: Solution, instance: XHSTTSInstance):
    solution = Solution(deepcopy(sol.sol_events))
    solution.mode = sol.mode
    idx = random.randint(0, len(solution.sol_events) - 1)
    event = solution.sol_events[idx]
    while (
        instance.Events[event.InstanceEventReference].PreAssignedTimeReference
        is not None
    ):
        idx = random.randint(0, len(solution.sol_events) - 1)
        event = solution.sol_events[idx]

    new_time_reference = (
        random.choice(list(event.PreferredTimes))
        if event.PreferredTimes
        else instance.get_random_time_reference()
    )

    new_event = event._replace(TimeReference=new_time_reference)
    solution.sol_events[idx] = new_event

    return solution


def swap_time(sol: Solution, instance: XHSTTSInstance):
    solution = Solution(deepcopy(sol.sol_events))
    solution.mode = sol.mode
    idx = random.randint(0, len(solution.sol_events) - 1)
    while (
        instance.Events[
            solution.sol_events[idx].InstanceEventReference
        ].PreAssignedTimeReference
        is not None
    ):
        idx = random.randint(0, len(solution.sol_events) - 1)

    other_idx = random.randint(0, len(solution.sol_events) - 1)

    tmp_time_ref = solution.sol_events[idx].TimeReference
    solution.sol_events[idx] = solution.sol_events[idx]._replace(
        TimeReference=solution.sol_events[other_idx].TimeReference
    )
    solution.sol_events[other_idx] = solution.sol_events[other_idx]._replace(
        TimeReference=tmp_time_ref
    )

    return solution


def change_resource(sol: Solution, instance: XHSTTSInstance):
    solution = Solution(deepcopy(sol.sol_events))
    solution.mode = sol.mode
    idx = random.randint(0, len(solution.sol_events) - 1)
    event = solution.sol_events[idx]

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

    return solution


def swap_resource(sol: Solution, instance: XHSTTSInstance):
    solution = Solution(deepcopy(sol.sol_events))
    solution.mode = sol.mode
    event = random.choice(solution.sol_events)

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

    return Solution


def kempe_vns(sol: Solution, instance: XHSTTSInstance, double=False):
    solution = Solution(deepcopy(sol.sol_events))
    solution.mode = sol.mode
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


# Solution -> Solution


def generate_initial_solution(instance: XHSTTSInstance) -> Solution:
    best_random_solution: Solution = max(
        [Solution(random_solution(instance)) for _ in range(1000)],
        key=lambda x: x.evaluate(instance),
    )
    print("best random : ", best_random_solution.cost)
    return best_random_solution


def variable_neighborhood_search(
    instance: XHSTTSInstance,
    input_solution_events: list[XHSTTSInstance.SolutionEvent],
    max_iterations: int = 100,
    assign_resources: bool = False,
    time_limit: int = 600,
):
    start_time = time.time()
    sol_changes_made = False

    neighborhoods = [change_time, swap_time, kempe_vns]

    # Define neighborhoods
    if assign_resources:
        neighborhoods.extend([change_resource, swap_resource])

    # Initialize best solution
    best_solution = Solution(deepcopy(input_solution_events))
    best_solution.evaluate(instance)

    if best_solution.is_feasible() and not sol_changes_made:
        instance.evaluate_solution(best_solution.sol_events, debug=True)
        best_solution.mode = Mode.Soft
        best_solution.needs_eval_update = True
        sol_changes_made = True
        best_solution.evaluate(instance)

    # Main loop
    iteration = 0
    while iteration < max_iterations and time.time() - start_time < time_limit:
        iter_start_time = time.time()

        # Select neighborhood
        curr_neighborhood = neighborhoods[iteration % len(neighborhoods)]

        # Apply local search within the selected neighborhood
        new_solution = Solution(
            local_search(
                instance,
                sol_events=curr_neighborhood(best_solution, instance).sol_events,
                max_no_improvement=20,
                max_restart_count=0,
                num_neighbours=500,
                neighbourhood=curr_neighborhood,
            )
        )

        # Update best solution if improvement is found
        if new_solution.evaluate(instance) > best_solution.evaluate(instance):
            best_solution = new_solution

        if best_solution.is_feasible() and not sol_changes_made:
            instance.evaluate_solution(best_solution.sol_events, debug=True)
            best_solution.mode = Mode.Soft
            best_solution.needs_eval_update = True
            sol_changes_made = True

        elapsed_time = time.time() - iter_start_time
        print(
            f"SA Iteration: {iteration} time taken: {elapsed_time} current energy {new_solution.evaluate(instance)} best energy {best_solution.evaluate(instance)} best_cost {best_solution.cost} time so far {(time.time() - start_time)}"
        )

        # Termination condition
        if best_solution.is_feasible_and_solves_objectives():
            break

        iteration += 1

    return best_solution.sol_events


if __name__ == "__main__":
    from pathlib import Path

    root_dir = Path(__file__).parent.parent.parent.parent
    data_dir = root_dir.joinpath("data/ALL_INSTANCES")

    lewitt = XHSTTS(data_dir.joinpath("SouthAfricaLewitt2009.xml"))
    dataset_sudoku4x4 = XHSTTS(data_dir.joinpath("ArtificialSudoku4x4.xml"))
    brazil2 = XHSTTS(data_dir.joinpath("BrazilInstance2.xml"))
    spainschool = XHSTTS(data_dir.joinpath("SpainSchool.xml"))

    instance = spainschool.get_instance(index=0)

    ils_tabu_sd_result = ils_tabu(instance, max_iterations=1)

    print("ILS-Tabu-SD result below")
    evaluation = instance.evaluate_solution(ils_tabu_sd_result, debug=True)
    print(
        f"\n---ILS-Tabu-SD Evaluation ({instance.name})---\n",
        evaluation,
    )

    # vns_res = variable_neighborhood_search(instance, ils_tabu_sd_result)

    # evaluation = instance.evaluate_solution(vns_res, debug=True)

    # print(
    #     f"\n---VNS Evaluation ({instance.name})---\n",
    #     evaluation,
    # )

    # annealing_result = simulated_annealing(instance, vns_res)
    # evaluation = instance.evaluate_solution(annealing_result, debug=True)

    # print(
    #     f"\n---Simulated Annealing Benchmark ({instance.name}) Evaluation ---\n",
    #     evaluation,
    #     "\n",
    # )

    vns_res = (
        variable_neighborhood_search(instance, ils_tabu_sd_result)
        if instance.name != "Spanish school"
        else variable_neighborhood_search(
            instance,
            ils_tabu_sd_result,
            time_limit=120,
            assign_resources=True,
        )
    )

    evaluation = instance.evaluate_solution(vns_res, debug=True)

    print(
        f"\n---VNS Evaluation ({instance.name})---\n",
        evaluation,
    )
