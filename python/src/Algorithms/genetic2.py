import random

import numpy as np
from Algorithms.random_algorithm import random_solution
from XHSTTS.utils import Cost
from XHSTTS.xhstts import XHSTTS, XHSTTSInstance

POPULATION_SIZE = 200
NGEN = 40  # 85


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
        # return -(self.cost.Infeasibility_Value + self.cost.Objective_Value)
        # return -self.cost.Infeasibility_Value

    def is_feasible(self) -> bool:
        return self.cost.Infeasibility_Value == 0


def mutate(solution: Solution, instance: XHSTTSInstance) -> None:
    for i, event in enumerate(solution.sol_events):
        # randomly mutate an event
        if random.random() < 0.01:
            # replace time ref
            if not instance.Events[
                event.InstanceEventReference
            ].PreAssignedTimeReference:
                new_time_reference = instance.get_random_time_reference()
                new_event = event._replace(TimeReference=new_time_reference)

            for k in range(len(event.Resources)):
                # randomly replace resource with a resource of the same type and role
                if random.random() < 0.01:
                    new_event_resource = (
                        instance.get_random_and_valid_resource_reference(
                            new_event.Resources[k], new_event.InstanceEventReference
                        )
                    )
                    new_event.Resources[k] = new_event_resource

            solution.sol_events[i] = new_event

        # randomly swap times and resources with other events??

        if random.random() < 0.01:
            other_idx = random.randint(0, len(solution.sol_events) - 1)
            swap(
                solution.sol_events[i].TimeReference,
                solution.sol_events[other_idx].TimeReference,
            )

            # for k in range(len(solution.sol_events[i].Resources)):
            #     other_event_resource_idx = random.randint(
            #         0, len(solution.sol_events[other_idx].Resources) - 1
            #     )
            #     if (
            #         solution.sol_events[i].Resources[k].Role
            #         == solution.sol_events[other_idx]
            #         .Resources[other_event_resource_idx]
            #         .Role
            #         and instance.get_resources()[
            #             solution.sol_events[other_idx]
            #             .Resources[other_event_resource_idx]
            #             .Reference
            #         ].ResourceTypeReference
            #         == instance.get_resources()[
            #             event.Resources[k].Reference
            #         ].ResourceTypeReference
            #     ):
            #         # swap
            #         swap(
            #             solution.sol_events[i].Resources[k],
            #             solution.sol_events[other_idx].Resources[
            #                 other_event_resource_idx
            #             ],
            #         )

        # TODO
        # randomly split events
        # how do you handle crossover then when lengths are no longer the same? diff type of cross over but still ensure all events are present?
        # maybe decide if to take all splits from event x randomly
        # also current crossover is ordered thats not ideal


def swap(a, b):
    a, b = b, a


def crossover(sol1: Solution, sol2: Solution) -> tuple[Solution]:
    # random.shuffle(sol1.sol_events)
    # random.shuffle(sol2.sol_events)

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

    # best of 1000 random solutions
    population: list[Solution] = sorted(
        [Solution(random_solution(instance)) for _ in range(1000)],
        key=lambda x: x.evaluate(instance),
        reverse=True,
    )[:POPULATION_SIZE]

    best_random = population[0].cost

    random.shuffle(population)

    best_solution = None

    print("Generation Begin!")
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
            mutate(offspring_solution, instance)

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

        print(f"Generation: {idx + 1}")

    print("\nbest random: ", best_random)

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

    for dataset in (
        dataset_sudoku4x4,
        dataset_abramson15,
    ):  # (dataset_sudoku4x4,):  # dataset_abramson15):
        random.seed(23)

        assert dataset.num_instances() == 1

        instance = dataset.get_instance(index=0)

        # get solution
        result = genetic_algorithm(instance)

        # evaluate
        evaluation = instance.evaluate_solution(result, debug=True)

        print(
            f"\n---Genetic Evaluation ({dataset_names[dataset]})---\n",
            evaluation,
        )

        # save the solution as an xml file
        solutions_dir = root_dir.joinpath("solutions")
        file_path = solutions_dir.joinpath(
            f"genetic_solution_{dataset_names[dataset]}.xml"
        )
        XHSTTSInstance.sol_events_to_xml(result, instance, file_path)
