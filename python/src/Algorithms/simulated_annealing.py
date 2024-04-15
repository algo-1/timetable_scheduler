# the parameters actually make it such that worse solutions are not accepted so this is just hill climbing with first improvememnt heuristic but can be modified to make it simulkated annealing :)

from copy import deepcopy
import random
import math
import time
from Algorithms.random_algorithm import random_solution
from Algorithms.utils import Mode, Solution, kempe_move, neighbor, swap
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
    current_energy: int, new_energy: int, temperature: float, weight=1000
) -> float:
    if new_energy > current_energy:
        return 1.0
    return math.exp(weight * (new_energy - current_energy) / temperature)


def simulated_annealing(
    instance: XHSTTSInstance,
    input_solution_events: list[XHSTTSInstance.SolutionEvent] = None,
    initial_temperature: int = 2,
    temperature_decay: float = 0.999995,
    lowest_temperature: int = 0.1,
    weight: float = 1000,
    time_limit: int = 20000,
    max_iterations: int = 5_000_000,
) -> list[XHSTTSInstance.SolutionEvent]:
    current_solution = None
    if input_solution_events:
        current_solution = Solution(input_solution_events)
    else:
        current_solution = generate_initial_solution(instance)
    current_energy = current_solution.evaluate(instance)

    best_solution = current_solution
    temperature = initial_temperature
    num_iterations = 0
    sol_changes_made = False
    no_improvement = 0
    start_time = time.time()

    best_solution.evaluate(instance)
    if best_solution.is_feasible() and not sol_changes_made:
        instance.evaluate_solution(best_solution.sol_events, debug=True)
        best_solution.mode = Mode.Soft
        best_solution.needs_eval_update = True
        current_solution.mode = Mode.Soft
        current_solution.needs_eval_update = True
        sol_changes_made = True
        current_energy = current_solution.evaluate(instance)

    while temperature > lowest_temperature and time.time() - start_time < time_limit:
        while True:
            iter_start_time = time.time()
            num_iterations += 1
            new_solution = neighbor(current_solution, instance)
            new_energy = new_solution.evaluate(instance)
            if (
                acceptance_probability(current_energy, new_energy, temperature, weight)
                > random.random()
            ):
                current_solution = new_solution
                current_energy = new_energy

            if new_energy > best_solution.evaluate(instance):
                best_solution = deepcopy(new_solution)
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement > 1000:
                no_improvement = 0
                break

            if best_solution.is_feasible_and_solves_objectives():
                break

            if best_solution.is_feasible() and not sol_changes_made:
                instance.evaluate_solution(best_solution.sol_events, debug=True)
                best_solution.mode = Mode.Soft
                best_solution.needs_eval_update = True
                current_solution.mode = Mode.Soft
                current_solution.needs_eval_update = True
                sol_changes_made = True
                current_energy = current_solution.evaluate(instance)

            iter_end_time = time.time()
            elapsed_time = iter_end_time - iter_start_time

            if num_iterations % 1000 == 0:
                print(
                    f"SA Iteration: {num_iterations} time taken: {elapsed_time} current energy {current_energy} best energy {best_solution.evaluate(instance)} best_cost {best_solution.cost} time so far {(time.time() - start_time)}"
                )
        if best_solution.is_feasible_and_solves_objectives():
            break

        temperature *= temperature_decay

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
    hdtt5 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt5.xml"))

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
            f"\n---Simulated Annealing Evaluation ({instance.name})---\n",
            evaluation,
        )

        # save the solution as an xml file
        solutions_dir = root_dir.joinpath("solutions")
        file_path = solutions_dir.joinpath(f"annealing_solution_{instance.name}.xml")
        XHSTTSInstance.sol_events_to_xml(result, instance, file_path)
