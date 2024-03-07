from contextlib import redirect_stdout
import io
from multiprocessing import current_process
import random
import sys
from Algorithms.random_algorithm import random_solution
from Algorithms.genetic2 import genetic_algorithm
from Algorithms.simulated_annealing import simulated_annealing
from Algorithms.utils import swap
from XHSTTS.xhstts import XHSTTS, Constraint, XHSTTSInstance
from XHSTTS.utils import Cost
from copy import deepcopy
import concurrent.futures
from pathlib import Path


class LocalSearchSolution:
    def __init__(self, sol_events: list[XHSTTSInstance.SolutionEvent]):
        self.sol_events = sol_events
        self.cost: Cost = None

    # TODO use cache here / add logic to prevent useless calls to instance.evaluate_solution
    def evaluate(self, instance: XHSTTSInstance) -> int:
        """
        returns the negative of the cost because 0 is the best cost.
        weights the infeasible cost 10 times the objective cost.
        TODO: choose better evaluation and investigate how it affects population
        """
        self.cost = instance.evaluate_solution(self.sol_events)
        return -(100 * self.cost.Infeasibility_Value + self.cost.Objective_Value)
        # return -(self.cost.Infeasibility_Value + self.cost.Objective_Value)
        # return -self.cost.Infeasibility_Value

    def is_feasible(self) -> bool:
        return self.cost.Infeasibility_Value == 0


def mutate(solution: LocalSearchSolution, instance: XHSTTSInstance) -> None:
    # TODO: scrap deepcopy in this function
    for i, event in enumerate(solution.sol_events):
        # randomly mutate an event
        new_event = deepcopy(event)
        if random.random() < 0.01:
            # decide between mutating the time or one of the resources
            rand_num = random.randint(0, len(new_event.Resources))
            if rand_num == len(new_event.Resources):
                # replace time ref
                if not instance.Events[
                    event.InstanceEventReference
                ].PreAssignedTimeReference:
                    new_time_reference = instance.get_random_time_reference()
                    new_event = event._replace(TimeReference=new_time_reference)
            else:
                resource_to_change_idx = (
                    rand_num  # rand_num is guaranteed to be a valid index
                )
                new_event_resource = instance.get_random_and_valid_resource_reference(
                    new_event.Resources[resource_to_change_idx],
                    new_event.InstanceEventReference,
                )
                new_event.Resources[resource_to_change_idx] = new_event_resource

            solution.sol_events[i] = new_event


# def mutate(solution: LocalSearchSolution, instance: XHSTTSInstance) -> None:
#     for i, event in enumerate(solution.sol_events):
#         # randomly mutate an event
#         new_event = event
#         if random.random() < 0.01:
#             # replace time ref
#             if not instance.Events[
#                 event.InstanceEventReference
#             ].PreAssignedTimeReference:
#                 new_time_reference = instance.get_random_time_reference()
#                 new_event = event._replace(TimeReference=new_time_reference)

#             for k in range(len(event.Resources)):
#                 # randomly replace resource with a resource of the same type and role
#                 if random.random() < 0.01:
#                     new_event_resource = (
#                         instance.get_random_and_valid_resource_reference(
#                             new_event.Resources[k], new_event.InstanceEventReference
#                         )
#                     )
#                     new_event.Resources[k] = new_event_resource

#             solution.sol_events[i] = new_event
#         # randomly swap times and resources with other events??

#         # if random.random() < 0.001:
#         #     other_idx = random.randint(0, len(solution.sol_events) - 1)
#         #     tmp_time_ref = solution.sol_events[i].TimeReference
#         #     solution.sol_events[i] = solution.sol_events[i]._replace(
#         #         TimeReference=solution.sol_events[other_idx].TimeReference
#         #     )
#         #     solution.sol_events[other_idx] = solution.sol_events[other_idx]._replace(
#         #         TimeReference=tmp_time_ref
#         #     )

#         #     for k in range(len(solution.sol_events[i].Resources)):
#         #         other_event_resource_idx = random.randint(
#         #             0, len(solution.sol_events[other_idx].Resources) - 1
#         #         )
#         #         if (
#         #             solution.sol_events[i].Resources[k].Role
#         #             == solution.sol_events[other_idx]
#         #             .Resources[other_event_resource_idx]
#         #             .Role
#         #             and instance.get_resources()[
#         #                 solution.sol_events[other_idx]
#         #                 .Resources[other_event_resource_idx]
#         #                 .Reference
#         #             ].ResourceTypeReference
#         #             == instance.get_resources()[
#         #                 solution.sol_events[i].Resources[k].Reference
#         #             ].ResourceTypeReference
#         #         ):
#         #             # swap
#         #             swap(
#         #                 solution.sol_events[i].Resources,
#         #                 solution.sol_events[other_idx].Resources,
#         #                 k,
#         #                 other_event_resource_idx,
#         #             )


def local_search(
    instance: XHSTTSInstance,
    max_iterations: int = 10_000,
    sol_events=[],
    restart_count=0,
) -> list[XHSTTSInstance.SolutionEvent]:
    best_random_solution = None
    if sol_events:
        current_solution = LocalSearchSolution(deepcopy(sol_events))
    else:
        best_random_solution: LocalSearchSolution = sorted(
            [LocalSearchSolution(random_solution(instance)) for _ in range(10)],
            key=lambda x: x.evaluate(instance),
            reverse=True,
        )[0]
        current_solution = best_random_solution

    current_solution.evaluate(instance)

    no_improvement = 0

    for iteration in range(max_iterations):
        # Generate neighbors by performing small changes to the current solution
        neighbors = [
            LocalSearchSolution(deepcopy(current_solution.sol_events))
            for _ in range(10)
        ]
        for neighbor in neighbors:
            mutate(neighbor, instance)

        # Evaluate the neighbors
        for neighbor in neighbors:
            neighbor.evaluate(instance)

        # Select the best neighbor
        best_neighbor = max(neighbors, key=lambda x: x.evaluate(instance))
        # print(sorted([x.evaluate(instance) for x in neighbors], reverse=True))

        # Check if the best neighbor is an improvement
        if best_neighbor.evaluate(instance) > current_solution.evaluate(instance):
            current_solution = deepcopy(best_neighbor)
            no_improvement = 0  # reset the no improvements
        else:
            # No improvement for 5 iterations, terminate the search
            no_improvement += 1
            # print(f"no consecutive improvements {no_improvement}")
            if no_improvement > 4:  # TODO make constant
                if restart_count < 3:  # TODO make constant
                    restart_count += 1
                    restart_solution = LocalSearchSolution(
                        local_search(
                            instance,
                            sol_events=sol_events,
                            restart_count=restart_count,
                        )
                    )
                    # Check if the new solution is an improvement
                    if restart_solution.evaluate(instance) > current_solution.evaluate(
                        instance
                    ):
                        current_solution = deepcopy(restart_solution)
                        no_improvement = 0
                else:
                    break

        # Check if a satisfactory solution has been found.
        if current_solution.is_feasible():
            return current_solution.sol_events

    if not sol_events:
        instance.evaluate_solution(best_random_solution.sol_events, debug=True)
        print("\nbest random: ", best_random_solution.cost)

    return current_solution.sol_events


def process_instance(instance: XHSTTSInstance):
    root_dir = Path(__file__).parent.parent.parent.parent
    solutions_dir = root_dir.joinpath("benchmark_solutions")
    print(
        f"Process {current_process().name} started working on {instance.name}",
        flush=True,
    )

    with io.StringIO() as buffer, redirect_stdout(buffer):

        print(f"-----{instance.name}   {len(instance .Constraints)} constraints-----")

        genetic_result = genetic_algorithm(instance)
        evaluation = instance.evaluate_solution(genetic_result, debug=True)

        print("\n---Genetic Evaluation---\n", evaluation)

        local_search_result = local_search(instance, sol_events=genetic_result)
        evaluation = instance.evaluate_solution(local_search_result, debug=True)

        print("\n---Local Search Benchmark Evaluation ---\n", evaluation, "\n")

        annealing_result = simulated_annealing(instance, local_search_result)
        evaluation = instance.evaluate_solution(annealing_result, debug=True)

        print("\n---Simulated Annealing Benchmark Evaluation ---\n", evaluation, "\n")

        # Write the output and solution XML to files
        output_path = solutions_dir.joinpath(Path(f"output_{instance.name}.txt"))
        xml_path = solutions_dir.joinpath(Path(f"solution_{instance.name}.xml"))

        with open(output_path, "w") as output_file:
            output_file.write(buffer.getvalue())

        XHSTTSInstance.sol_events_to_xml(annealing_result, instance, xml_path)

    return f"{instance.name} finished"


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent.parent
    data_dir = root_dir.joinpath("data/ALL_INSTANCES")

    dataset_sudoku4x4 = XHSTTS(data_dir.joinpath("ArtificialSudoku4x4.xml"))
    dataset_abramson15 = XHSTTS(data_dir.joinpath("ArtificialAbramson15.xml"))
    dataset_brazil3 = XHSTTS(data_dir.joinpath("BrazilInstance3.xml"))
    benchmark_dataset = XHSTTS(data_dir.parent.joinpath("XHSTT-2014.xml"))

    dataset_names = {
        dataset_sudoku4x4: "ArtificialSudoku4x4",
        dataset_abramson15: "ArtificialAbramson15",
        dataset_brazil3: "BrazilInstance3.xml",
    }

    random.seed(23)

    print(f"number of benchmark instances = {benchmark_dataset.num_instances()}")

    benchmark_instances = [
        benchmark_dataset.get_instance(index=idx)
        for idx in range(benchmark_dataset.num_instances())
    ]

    assert len(benchmark_instances) == 25

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # futures = [
        #     executor.submit(process_instance, task)
        #     for task in [benchmark_instances[i] for i in range(10, 16)]
        #     + [benchmark_instances[21]]
        # ]

        futures = [
            executor.submit(process_instance, task)
            for task in [
                dataset_sudoku4x4.get_first_instance(),
                benchmark_instances[21],
            ]
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)
