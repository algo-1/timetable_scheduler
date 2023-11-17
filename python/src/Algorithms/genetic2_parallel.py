import random
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
from Algorithms.random_algorithm import random_solution
from XHSTTS.utils import Cost
from XHSTTS.xhstts import XHSTTS, XHSTTSInstance

POPULATION_SIZE = 100
NGEN = 85


class Solution:
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
        return -(10 * self.cost.Infeasibility_Value + self.cost.Objective_Value)

    def is_feasible(self) -> bool:
        return self.cost.Infeasibility_Value == 0


def mutate(solution: Solution) -> None:
    # TODO: make faster is there a nice way to use vectors/numpy arrays?
    random.shuffle(solution.sol_events)
    for i, event in enumerate(solution.sol_events):
        if (
            random.random() < 0.01
        ):  # TODO parameterise this value and how do we decide it?
            # Randomly select a different event.
            other_idx = random.randint(0, len(solution.sol_events) - 1)
            other_event = solution.sol_events[other_idx]

            # Swap the two events. # TODO: refactor sol_events to dataclasses!
            tmp = event

            # swap times
            event = event._replace(TimeReference=other_event.TimeReference)

            assert tmp != event or (tmp.TimeReference == other_event.TimeReference)

            other_event = other_event._replace(TimeReference=tmp.TimeReference)

            # swap resources
            selected = set()
            if other_event.Resources:
                for i in range(len(event.Resources)):
                    if random.random() < 0.01:
                        # Randomly select a resource from the other event.
                        other_event_resource_idx = random.randint(
                            0, len(other_event.Resources) - 1
                        )
                        if other_event_resource_idx not in selected:
                            # swap
                            (
                                event.Resources[i],
                                other_event.Resources[other_event_resource_idx],
                            ) = (
                                other_event.Resources[other_event_resource_idx],
                                event.Resources[i],
                            )
                            selected.add(other_event_resource_idx)

            solution.sol_events[i] = event
            solution.sol_events[other_idx] = other_event


def crossover(sol1: Solution, sol2: Solution) -> tuple[Solution]:
    random.shuffle(sol1.sol_events)
    random.shuffle(sol2.sol_events)

    # Randomly select a crossover point.
    crossover_point = random.randint(0, len(sol1.sol_events) - 1)

    # Create two new offspring solutions.
    offspring1 = Solution(
        sol1.sol_events[:crossover_point] + sol2.sol_events[crossover_point:]
    )
    offspring2 = Solution(
        sol2.sol_events[:crossover_point] + sol1.sol_events[crossover_point:]
    )

    return offspring1, offspring2


def tournament_selection(
    population: list[Solution], instance: XHSTTSInstance, size: int
) -> list[Solution]:
    """
    Selects the best solutions from the population using tournament selection.

    Args:
        population: A list of solutions.
        fitness: A list of fitness values for the solutions in the population.
        tournament_size: The size of the tournament pool.

    Returns:
        A list of the best solutions selected from the population.
    """
    selected_solutions = []
    for _ in range(size):
        # Randomly select a subset of individuals from the population.
        tournament_pool = random.sample(population, 10)

        # Choose the best individual from the tournament pool.
        winner = max(
            tournament_pool, key=lambda individual: individual.evaluate(instance)
        )

        selected_solutions.append(winner)

    return selected_solutions


# slower than non-parallel possibly due to overhead, TODO: profile non-parallel - vectorise with numpy?
def generate_offspring_parallel(selected_solutions, num_processes):
    with Pool(processes=num_processes) as pool:
        offspring = pool.map(
            lambda args: crossover(*args),
            [
                (selected_solutions[i], selected_solutions[i + 1])
                for i in range(0, len(selected_solutions), 2)
            ],
        )

        return [solution for pair in offspring for solution in pair]


# issue with not being able to pickle <class 'XHSTTS.xhstts.SolutionEvent'> TODO: profile non-parallel - vectorise with numpy?
# def generate_offspring_parallel(selected_solutions, num_processes):
#     pool = multiprocessing.Pool(processes=num_processes)
#     offspring = pool.starmap(
#         crossover,
#         [
#             (selected_solutions[i], selected_solutions[i + 1])
#             for i in range(0, len(selected_solutions), 2)
#         ],
#     )
#     pool.close()
#     pool.join()
#     return [solution for pair in offspring for solution in pair]


def genetic_algorithm(instance) -> list[XHSTTSInstance.SolutionEvent]:
    # Create a population of solutions.

    # best of 1000 random solutions
    population: list[Solution] = sorted(
        [Solution(random_solution(instance)) for _ in range(1000)],
        key=lambda x: x.evaluate(instance),
        reverse=True,
    )[:POPULATION_SIZE]

    best_random = population[0].cost

    random.shuffle(population)

    best_solution = None

    for idx in range(NGEN):
        # Select the best solutions from the population.
        selected_solutions = tournament_selection(
            population, instance, 20
        )  # parameterise this number

        # Crossover the selected solutions to produce new offspring.
        offspring = generate_offspring_parallel(selected_solutions, num_processes=8)

        # Mutate the offspring.
        for offspring_solution in offspring:
            mutate(offspring_solution)

        # Evaluate the offspring.
        for offspring_solution in offspring:
            offspring_solution.evaluate(instance)

        # Add the offspring to the population and remove the worst individuals.
        population = selected_solutions + offspring
        population = sorted(
            population, key=lambda solution: solution.evaluate(instance), reverse=True
        )[:100]

        # print([x.evaluate(instance) for x in population])

        # Check if a satisfactory solution has been found.
        best_solution = max(
            population, key=lambda solution: solution.evaluate(instance)
        )
        if best_solution.is_feasible():
            return best_solution.sol_events

        # print(f"Generation: {idx + 1}")

    print("best random: ", best_random)

    return best_solution.sol_events


if __name__ == "__main__":
    from pathlib import Path

    root_dir = Path(__file__).parent.parent.parent.parent
    data_dir = root_dir.joinpath("data/ALL_INSTANCES")

    dataset_sudoku4x4 = XHSTTS(data_dir.joinpath("ArtificialSudoku4x4.xml"))
    dataset_abramson15 = XHSTTS(data_dir.joinpath("ArtificialAbramson15.xml"))

    dataset_names = {
        dataset_sudoku4x4: "ArtificialSudoku4x4",
        dataset_abramson15: "ArtificialAbramson15",
    }

    for dataset in (dataset_sudoku4x4,):  # , dataset_abramson15):
        random.seed(23)

        assert dataset.num_instances() == 1

        instance = dataset.get_instance(index=0)

        # get solution
        result = genetic_algorithm(instance)

        # evaluate
        evaluation = instance.evaluate_solution(result)

        print(f"\n---Evaluation ({dataset_names[dataset]})---\n", evaluation)
