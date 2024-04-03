from copy import deepcopy
import random
import math
import time
from Algorithms.random_algorithm import random_solution
from Algorithms.utils import Solution, neighbor, swap
from XHSTTS.xhstts import XHSTTSInstance, XHSTTS
from XHSTTS.utils import Cost


def generate_initial_solution(instance: XHSTTSInstance) -> Solution:
    best_random_solution: Solution = max(
        [Solution(random_solution(instance)) for _ in range(1000)],
        key=lambda x: x.evaluate(instance),
    )
    print("best random : ", best_random_solution.cost)
    return best_random_solution


def acceptance_probability(
    current_energy: int, new_energy: int, temperature: float
) -> float:
    if new_energy > current_energy:
        return 1.0
    return math.exp((new_energy - current_energy) / temperature)


def simulated_annealing(
    instance: XHSTTSInstance,
    input_solution_events: list[XHSTTSInstance.SolutionEvent] = None,
) -> list[XHSTTSInstance.SolutionEvent]:
    current_solution = None
    if input_solution_events:
        current_solution = Solution(input_solution_events)
    else:
        current_solution = generate_initial_solution(instance)
    current_energy = current_solution.evaluate(instance)

    best_solution = current_solution
    temperature = 2
    temperature_decay = 0.999999

    num_iterations = 0

    while temperature > 0.1:
        start_time = time.time()

        num_iterations += 1
        new_solution = neighbor(current_solution, instance)
        new_energy = new_solution.evaluate(instance)

        if (
            acceptance_probability(current_energy, new_energy, temperature)
            > random.random()
        ):
            current_solution = new_solution
            current_energy = new_energy

        if new_energy > best_solution.evaluate(instance):
            best_solution = new_solution

        if best_solution.is_feasible_and_solves_objectives():
            break

        temperature *= temperature_decay

        end_time = time.time()
        elapsed_time = end_time - start_time

        if num_iterations % 1000 == 0:
            print(
                f"SA Iteration: {num_iterations} time taken: {elapsed_time} current energy {current_energy} best energy {best_solution.evaluate(instance)} best_cost {best_solution.cost}"
            )

    print(f"number of Simulated Annealing iterations = {num_iterations}")
    return best_solution.sol_events


if __name__ == "__main__":
    from pathlib import Path

    root_dir = Path(__file__).parent.parent.parent.parent
    data_dir = root_dir.joinpath("data/ALL_INSTANCES")

    dataset_sudoku4x4 = XHSTTS(data_dir.joinpath("ArtificialSudoku4x4.xml"))
    dataset_abramson15 = XHSTTS(data_dir.joinpath("ArtificialAbramson15.xml"))
    dataset_brazil3 = XHSTTS(data_dir.joinpath("BrazilInstance3.xml"))
    aus_bghs98 = XHSTTS(data_dir.joinpath("AustraliaBGHS98.xml"))

    dataset_names = {
        dataset_sudoku4x4: "ArtificialSudoku4x4",
        dataset_abramson15: "ArtificialAbramson15",
        dataset_brazil3: "BrazilInstance3.xml",
    }
    for dataset in (dataset_abramson15,):
        random.seed(23)

        assert dataset.num_instances() == 1

        instance = dataset.get_instance(index=0)

        # get solution
        result = simulated_annealing(instance)

        # evaluate
        evaluation = instance.evaluate_solution(result, debug=True)

        print(
            f"\n---Simulated Annealing Evaluation ({dataset_names[dataset]})---\n",
            evaluation,
        )

        # save the solution as an xml file
        solutions_dir = root_dir.joinpath("solutions")
        file_path = solutions_dir.joinpath(
            f"annealing_solution_{dataset_names[dataset]}.xml"
        )
        XHSTTSInstance.sol_events_to_xml(result, instance, file_path)
