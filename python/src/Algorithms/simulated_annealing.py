from copy import deepcopy
import random
import math
from Algorithms.random_algorithm import random_solution
from Algorithms.utils import swap
from XHSTTS.xhstts import XHSTTSInstance, XHSTTS
from XHSTTS.utils import Cost


class Solution:
    def __init__(self, sol_events: list[XHSTTSInstance.SolutionEvent]):
        self.sol_events = sol_events
        self.cost: Cost = None

    def evaluate(self, instance: XHSTTSInstance) -> int:
        self.cost = instance.evaluate_solution(self.sol_events)
        return -(10 * self.cost.Infeasibility_Value + self.cost.Objective_Value)

    def is_feasible(self) -> bool:
        return self.cost.Infeasibility_Value == 0


def generate_initial_solution(instance: XHSTTSInstance) -> Solution:
    from Algorithms.genetic2 import genetic_algorithm
    from Algorithms.local_search import local_search

    # best_random_solution: Solution = sorted(
    #     [Solution(random_solution(instance)) for _ in range(100)],
    #     key=lambda x: x.evaluate(instance),
    #     reverse=True,
    # )[0]
    # print("best random : ", best_random_solution.cost)
    # return best_random_solution
    genetic_result = genetic_algorithm(instance)
    evaluation = instance.evaluate_solution(genetic_result, debug=True)
    print(
        f"\n---Genetic Evaluation ({instance.name})---\n",
        evaluation,
    )
    local_search_result = local_search(instance, sol_events=genetic_result)

    # evaluate local search result
    evaluation = instance.evaluate_solution(local_search_result, debug=True)

    print(
        f"\n---Local Search Benchmark Evaluation {instance.name} ---\n",
        evaluation,
        "\n",
    )
    return Solution(local_search_result)


def neighbor(solution: Solution, instance: XHSTTSInstance) -> Solution:
    # TODO: scrap deepcopy in this function

    new_solution_events = []
    for event in solution.sol_events:
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

        new_solution_events.append(new_event)

    return Solution(new_solution_events)


def acceptance_probability(
    current_energy: int, new_energy: int, temperature: float
) -> float:
    if new_energy > current_energy:
        return 1.0
    return math.exp((new_energy - current_energy) / temperature)


def simulated_annealing(
    instance: XHSTTSInstance, input_solution_events: list[XHSTTSInstance.SolutionEvent]
) -> list[XHSTTSInstance.SolutionEvent]:
    current_solution = None
    if input_solution_events:
        current_solution = Solution(input_solution_events)
    else:
        current_solution = generate_initial_solution(instance)
    current_energy = current_solution.evaluate(instance)

    best_solution = current_solution
    temperature = 1
    temperature_decay = 0.99995

    while temperature > 0.1:
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

        temperature *= temperature_decay

    return best_solution.sol_events


if __name__ == "__main__":
    from pathlib import Path

    root_dir = Path(__file__).parent.parent.parent.parent
    data_dir = root_dir.joinpath("data/ALL_INSTANCES")

    dataset_sudoku4x4 = XHSTTS(data_dir.joinpath("ArtificialSudoku4x4.xml"))
    dataset_abramson15 = XHSTTS(data_dir.joinpath("ArtificialAbramson15.xml"))
    dataset_brazil3 = XHSTTS(data_dir.joinpath("BrazilInstance3.xml"))

    dataset_names = {
        dataset_sudoku4x4: "ArtificialSudoku4x4",
        dataset_abramson15: "ArtificialAbramson15",
        dataset_brazil3: "BrazilInstance3.xml",
    }
    for dataset in (dataset_sudoku4x4, dataset_brazil3):
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
