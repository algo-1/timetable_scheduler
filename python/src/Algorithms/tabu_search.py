from collections import deque
from copy import deepcopy
import random
import math
import time
from Algorithms.genetic2 import genetic_algorithm
from Algorithms.local_search import local_search
from Algorithms.random_algorithm import random_solution
from Algorithms.simulated_annealing import simulated_annealing
from Algorithms.utils import Solution, mutate_resource, mutate_time, neighbor, swap
from XHSTTS.xhstts import XHSTTSInstance, XHSTTS
from XHSTTS.utils import Cost


def generate_initial_solution(instance: XHSTTSInstance) -> Solution:
    best_random_solution: Solution = max(
        [Solution(random_solution(instance)) for _ in range(1000)],
        key=lambda x: x.evaluate(instance),
    )
    print("best random : ", best_random_solution.cost)
    return best_random_solution


def generate_initial_neighbourhoods(instance: XHSTTSInstance, n=10) -> list[Solution]:
    best_neighbourhoods = sorted(
        [Solution(random_solution(instance)) for _ in range(1000)],
        key=lambda x: x.evaluate(instance),
        reverse=True,
    )[:n]
    print("best neighbourhood initial cost: ", best_neighbourhoods[0].cost)
    return best_neighbourhoods


def get_tabu_entry(solution: Solution):
    return tuple(x.TimeReference for x in solution.sol_events) + tuple(
        resource for event in solution.sol_events for resource in event.Resources
    )


def tabu_search(
    instance: XHSTTSInstance,
    input_solution_events: list[XHSTTSInstance.SolutionEvent] = None,
    tabu_size: int = 20,
    max_iterations: int = 5000,
    num_neighbours: int = 500,
    max_no_improvement: int = 20,
) -> list[XHSTTSInstance.SolutionEvent]:
    current_solution = None
    if input_solution_events:
        current_solution = Solution(input_solution_events)
    else:
        current_solution = generate_initial_solution(instance)
    current_energy = current_solution.evaluate(instance)

    best_solution = current_solution
    tabu_list = deque([])

    num_iterations = 0
    no_improvement = 0

    while num_iterations < max_iterations:
        start_time = time.time()

        num_iterations += 1

        neighbors = [
            neighbor(current_solution, instance) for _ in range(num_neighbours)
        ]

        new_solution = max(neighbors, key=lambda x: x.evaluate(instance))

        new_energy = new_solution.evaluate(instance)
        print(f"-- {new_energy} {current_energy} epoch: {num_iterations}")
        if new_energy > current_energy:
            no_improvement = 0
            print(f"improved{new_energy} {current_energy} epoch: {num_iterations}")
            tabu_entry = get_tabu_entry(new_solution)
            if tabu_entry not in tabu_list:
                current_solution = new_solution
                current_energy = new_energy
                tabu_list.append(tabu_entry)
                if len(tabu_list) > tabu_size:
                    tabu_list.popleft()
            else:
                print(
                    "Already in tabu list"
                )  # won't happen as by it had lower energy, to see if you come across a prev sol, check this before new_energy > current_energy
        else:
            no_improvement += 1
            pass

        if new_energy > best_solution.evaluate(instance):
            best_solution = deepcopy(new_solution)

        if best_solution.is_feasible_and_solves_objectives():
            break

        if no_improvement >= max_no_improvement:
            break

        end_time = time.time()
        elapsed_time = end_time - start_time

        if num_iterations % 20 == 0:
            print(
                f"Tabu Search Iteration: {num_iterations} time taken: {elapsed_time} best cost = {best_solution.cost}  tabu size = {len(tabu_list)} {new_energy} {current_energy}"
            )

    print(f"number of Tabu Search iterations = {num_iterations}")
    return best_solution.sol_events


if __name__ == "__main__":
    from pathlib import Path

    root_dir = Path(__file__).parent.parent.parent.parent
    data_dir = root_dir.joinpath("data/ALL_INSTANCES")

    dataset_sudoku4x4 = XHSTTS(data_dir.joinpath("ArtificialSudoku4x4.xml"))
    dataset_abramson15 = XHSTTS(data_dir.joinpath("ArtificialAbramson15.xml"))
    dataset_brazil3 = XHSTTS(data_dir.joinpath("BrazilInstance3.xml"))
    aus_bghs98 = XHSTTS(data_dir.joinpath("AustraliaBGHS98.xml"))
    hdtt4 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt4.xml"))
    hdtt5 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt5.xml"))
    hdtt6 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt6.xml"))
    hdtt7 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt7.xml"))
    hdtt8 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt8.xml"))
    brazil2 = XHSTTS(data_dir.joinpath("BrazilInstance2.xml"))
    lewitt = XHSTTS(data_dir.joinpath("SouthAfricaLewitt2009.xml"))
    woodlands = XHSTTS(data_dir.joinpath("SouthAfricaWoodlands2009.xml"))
    spainschool = XHSTTS(data_dir.joinpath("SpainSchool.xml"))
    aus_tes99 = XHSTTS(data_dir.joinpath("AustraliaTES99.xml"))

    dataset_names = {
        dataset_sudoku4x4: "ArtificialSudoku4x4",
        dataset_abramson15: "ArtificialAbramson15",
        dataset_brazil3: "BrazilInstance3.xml",
    }
    for dataset in (
        # aus_tes99,
        # woodlands,
        # brazil2,
        lewitt,
        # dataset_brazil3,
        # dataset_sudoku4x4,
        # hdtt4,
        # hdtt5,
        # hdtt6,
        # hdtt7,
        # hdtt7,
        # hdtt8,
        # dataset_abramson15,
        # spainschool,
    ):  # dataset_abramson15, dataset_brazil3):
        random.seed(23)

        instance = dataset.get_instance(index=0)

        # get solution
        genetic_result = genetic_algorithm(instance)

        print("genetic result below")
        evaluation = instance.evaluate_solution(genetic_result, debug=True)
        print(
            f"\n---Genetic Evaluation ({instance.name})---\n",
            evaluation,
        )

        tabu_search_result = tabu_search(instance, genetic_result)

        # evaluate
        evaluation = instance.evaluate_solution(tabu_search_result, debug=True)

        print(
            f"\n---Tabu Search Evaluation ({instance.name})---\n",
            evaluation,
        )

        annealing_result = simulated_annealing(instance, tabu_search_result)
        evaluation = instance.evaluate_solution(annealing_result, debug=True)

        print(
            f"\n---Simulated Annealing Benchmark ({instance.name}) Evaluation ---\n",
            evaluation,
            "\n",
        )

        # multi neigh tabu |vlns??

        # best_sol = None
        # best_eval = -float("inf")
        # for neigh in generate_initial_neighbourhoods(instance, n=10):
        #     res = Solution(
        #         tabu_search(instance, input_solution_events=neigh.sol_events)
        #     )
        #     if res.evaluate(instance) > best_eval:
        #         best_sol = res
        #         best_eval = res.evaluate(instance)

        # print(
        #     f"\n---Tabu Search Evaluation ({instance.name})---\n",
        #     instance.evaluate_solution(best_sol.sol_events, debug=True),
        # )

        # print("-" * 50)

        # multi tabu -> genetic
        # input_population = []
        # costs = []
        # for idx, neigh in enumerate(generate_initial_neighbourhoods(instance, n=20)):
        #     res = local_search(instance, sol_events=neigh.sol_events)
        #     # print(
        #     #     f"\n---Tabu Search {idx} Evaluation ({instance.name})---\n",
        #     #     instance.evaluate_solution(res, debug=True),
        #     # )
        #     costs.append(instance.evaluate_solution(res).Infeasibility_Value)
        #     input_population.append(res)

        # print("tabu infeasibility costs", costs)

        # print("genetic result below")
        # genetic_result = genetic_algorithm(instance, input_population=input_population)
        # evaluation = instance.evaluate_solution(genetic_result, debug=True)
        # print(
        #     f"\n---Genetic Evaluation ({instance.name})---\n",
        #     evaluation,
        # )
        # annealing_result = simulated_annealing(instance, genetic_result)
        # evaluation = instance.evaluate_solution(annealing_result, debug=True)

        # print("\n---Simulated Annealing Benchmark Evaluation ---\n", evaluation, "\n")

        # save the solution as an xml file
        # solutions_dir = root_dir.joinpath("solutions")
        # file_path = solutions_dir.joinpath(f"tabu_search_solution_{instance.name}.xml")
        # XHSTTSInstance.sol_events_to_xml(result, instance, file_path)
