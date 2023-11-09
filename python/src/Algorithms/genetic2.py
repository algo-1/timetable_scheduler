import random
from Algorithms.random_algorithm import random_solution
from XHSTTS.utils import Cost
from XHSTTS.xhstts import XHSTTS, XHSTTSInstance

POPULATION_SIZE = 100
NGEN = 40


class Solution:
    def __init__(self, sol_events: list[XHSTTSInstance.SolutionEvent]):
        self.sol_events = sol_events
        self.cost: Cost = None

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
    for i, event in enumerate(solution.sol_events):
        if (
            random.random() < 0.5
        ):  # TODO parameterise this value and how do we decide it?
            # Randomly select a different event.
            other_idx = random.randint(0, len(solution.sol_events) - 1)
            other_event = solution.sol_events[other_idx]

            # Swap the two events. # TODO: refactor sol_events to dataclasses!
            tmp = event
            event = event._replace(
                TimeReference=other_event.TimeReference, Resources=other_event.Resources
            )
            assert tmp != event or (
                tmp.TimeReference == other_event.TimeReference
                and tmp.Resources == other_event.Resources
            )
            other_event = other_event._replace(
                TimeReference=tmp.TimeReference, Resources=tmp.Resources
            )
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


def genetic_algorithm(instance) -> list[XHSTTSInstance.SolutionEvent]:
    # Create a population of solutions.
    population: list[Solution] = []
    for _ in range(POPULATION_SIZE):
        solution = Solution(random_solution(instance))
        population.append(solution)

    best_solution = None

    # Repeat until a satisfactory solution is found.
    for idx in range(NGEN):
        # Select the best solutions from the population.
        selected_solutions = tournament_selection(population, instance, 20)

        # Crossover the selected solutions to produce new offspring.
        offspring: list[Solution] = []
        for i in range(0, len(selected_solutions), 2):
            new_solution1, new_solution2 = crossover(
                selected_solutions[i], selected_solutions[i + 1]
            )
            offspring.append(new_solution1)
            offspring.append(new_solution2)

        # Mutate the offspring.
        for offspring_solution in offspring:
            mutate(offspring_solution)

        # Evaluate the offspring.
        for offspring_solution in offspring:
            offspring_solution.evaluate(instance)

        # Add the offspring to the population and remove the worst individuals.
        population = selected_solutions + offspring
        population = sorted(
            population, key=lambda solution: solution.evaluate(instance)
        )[:100]

        # Check if a satisfactory solution has been found.
        best_solution = max(
            population, key=lambda solution: solution.evaluate(instance)
        )
        if best_solution.is_feasible():
            return best_solution.sol_events

        print(f"Generation: {idx + 1}")

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

    for dataset in (dataset_sudoku4x4, dataset_abramson15):
        random.seed(23)

        assert dataset.num_instances() == 1

        instance = dataset.get_instance(index=0)

        # get solution
        result = genetic_algorithm(instance)

        # evaluate
        evaluation = instance.evaluate_solution(result)

        print(f"\n---Evaluation ({dataset_names[dataset]})---\n", evaluation)
