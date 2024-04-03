from contextlib import redirect_stdout
import io
from multiprocessing import current_process
import random
import sys
import time
from Algorithms.random_algorithm import random_solution
from Algorithms.genetic2 import genetic_algorithm
from Algorithms.simulated_annealing import simulated_annealing
from Algorithms.utils import Solution, mutate, swap
from XHSTTS.xhstts import XHSTTS, Constraint, XHSTTSInstance
from XHSTTS.utils import Cost
from copy import deepcopy
import concurrent.futures
from pathlib import Path


def local_search(
    instance: XHSTTSInstance,
    max_iterations: int = 10_000,
    sol_events=[],
    restart_count=0,
) -> list[XHSTTSInstance.SolutionEvent]:
    best_random_solution = None
    if sol_events:
        current_solution = Solution(deepcopy(sol_events))
    else:
        best_random_solution: Solution = sorted(
            [Solution(random_solution(instance)) for _ in range(10)],
            key=lambda x: x.evaluate(instance),
            reverse=True,
        )[0]
        current_solution = best_random_solution

    current_solution.evaluate(instance)

    no_improvement = 0

    for iteration in range(max_iterations):
        # Generate neighbors by performing small changes to the current solution
        neighbors = [Solution(deepcopy(current_solution.sol_events)) for _ in range(10)]
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
                    restart_solution = Solution(
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
        if current_solution.is_feasible_and_solves_objectives():
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
        start_time = time.time()

        print(f"-----{instance.name}   {len(instance .Constraints)} constraints-----")

        genetic_result = genetic_algorithm(instance)
        evaluation = instance.evaluate_solution(genetic_result, debug=True)

        print("\n---Genetic Evaluation---\n", evaluation)

        # local_search_result = local_search(instance, sol_events=genetic_result)
        # evaluation = instance.evaluate_solution(local_search_result, debug=True)

        # print("\n---Local Search Benchmark Evaluation ---\n", evaluation, "\n")

        annealing_result = simulated_annealing(instance, genetic_result)
        evaluation = instance.evaluate_solution(annealing_result, debug=True)

        print("\n---Simulated Annealing Benchmark Evaluation ---\n", evaluation, "\n")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{instance.name} finished in {elapsed_time} seconds.")

        # Write the output and solution XML to files
        output_path = solutions_dir.joinpath(
            Path(
                f"output_{instance.name}_spread_fixed_evaluate_min_split_low_iter_annealing.txt"
            )
        )
        xml_path = solutions_dir.joinpath(
            Path(
                f"solution_{instance.name}_spread_fixed_evaluate_min_split_low_iter_annealing.xml"
            )
        )

        with open(output_path, "w") as output_file:
            output_file.write(buffer.getvalue())

        XHSTTSInstance.sol_events_to_xml(annealing_result, instance, xml_path)

    return f"{instance.name} finished"  # annealing_result


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent.parent
    data_dir = root_dir.joinpath("data/ALL_INSTANCES")

    dataset_sudoku4x4 = XHSTTS(data_dir.joinpath("ArtificialSudoku4x4.xml"))
    dataset_abramson15 = XHSTTS(data_dir.joinpath("ArtificialAbramson15.xml"))
    hdtt4 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt4.xml"))
    hdtt5 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt5.xml"))
    hdtt6 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt6.xml"))
    hdtt7 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt7.xml"))
    hdtt8 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt8.xml"))
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
        #     + [dataset_sudoku4x4.get_first_instance(), benchmark_instances[21]]
        # ]

        futures = [
            executor.submit(process_instance, task)
            for task in [
                hdtt4.get_first_instance(),
                hdtt5.get_first_instance(),
                hdtt6.get_first_instance(),
                hdtt7.get_first_instance(),
                hdtt8.get_first_instance(),
                dataset_sudoku4x4.get_first_instance(),
                benchmark_instances[3],
                benchmark_instances[21],
                benchmark_instances[0],
                dataset_abramson15.get_first_instance(),
                dataset_brazil3.get_first_instance(),
                benchmark_instances[10],
                benchmark_instances[11],
            ]
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as e:
                print(e, flush=True)
