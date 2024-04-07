from copy import deepcopy
import math
import random
import time

import numpy as np
from Algorithms.random_algorithm import random_solution
from Algorithms.utils import Mode, Solution, mutate, swap
from XHSTTS.utils import Cost
from XHSTTS.xhstts import XHSTTS, XHSTTSInstance


def crossover(
    sol1: Solution, sol2: Solution, uniform_percentage: int
) -> tuple[Solution, Solution]:
    # random.shuffle(sol1.sol_events)
    # random.shuffle(sol2.sol_events)

    # uniform crossover
    offspring1_events = []
    offspring2_events = []

    for ref in sol1.original_events:
        sol1_idxs = sol1.original_events[ref]
        sol2_idxs = sol2.original_events[ref]
        if random.random() < 0.5:
            # swap
            for index in sol2_idxs:
                offspring1_events.append(sol2.sol_events[index])

            for index in sol1_idxs:
                offspring2_events.append(sol1.sol_events[index])

        else:
            for index in sol1_idxs:
                offspring1_events.append(sol1.sol_events[index])

            for index in sol2_idxs:
                offspring2_events.append(sol2.sol_events[index])

    new_sol1 = Solution(offspring1_events)
    new_sol1.mode = sol1.mode
    new_sol2 = Solution(offspring2_events)
    new_sol2.mode = sol2.mode

    return new_sol1, new_sol2


def tournament_selection(
    population: list[Solution],
    instance: XHSTTSInstance,
    size: int,
    sample_size: int = None,
) -> list[Solution]:

    random.shuffle(population)
    sample_size = sample_size if sample_size else math.floor(math.sqrt(len(population)))

    selected_solutions = []
    for _ in range(size):
        # Randomly select a subset of individuals from the population.
        tournament_pool = random.sample(population, sample_size)

        # Choose the best individual from the tournament pool.
        winner = max(
            tournament_pool, key=lambda individual: individual.evaluate(instance)
        )

        selected_solutions.append(deepcopy(winner))

    return selected_solutions


def genetic_algorithm(
    instance: XHSTTSInstance,
    tournament_size: int = None,
    num_selected_for_crossover: int = None,
    mutation_rate: float = 0.2,
    input_solution: list[XHSTTSInstance.SolutionEvent] = [],
    input_population: list[list[XHSTTSInstance.SolutionEvent]] = [],
    crossover_uniform_percentage: float = 0.5,
    POPULATION_SIZE: int = 150,
    NGEN: int = 500,
) -> list[XHSTTSInstance.SolutionEvent]:
    # Create a population of solutions.
    sol_changes_made = False

    if tournament_size:
        assert (
            tournament_size < POPULATION_SIZE
        ), f"tournament_size ({tournament_size}) should be less than the POPULATION_SIZE ({POPULATION_SIZE})"

    population: list[Solution] = None
    best_random_solution = None
    if input_solution:
        POPULATION_SIZE = min(POPULATION_SIZE, len(input_solution) * 2)
        population = [
            Solution(deepcopy(input_solution)) for _ in range(POPULATION_SIZE)
        ]
        NGEN = max(NGEN, int(len(population[0].sol_events) // 2))
        # mutate the population as currently all == input_solution TODO: better mutation as this isn't a great input into genetic or just scrap entirely?
        map(lambda sol: mutate(sol, instance), population)
    elif input_population:
        population = [Solution(chromosome) for chromosome in input_population]
        NGEN = max(NGEN, int(len(population[0].sol_events) // 2))
        POPULATION_SIZE = len(population)

        print(f"population size = {POPULATION_SIZE}\nnumber of generations = {NGEN}")

    else:
        random_1000 = sorted(
            [Solution(random_solution(instance)) for _ in range(1000)],
            key=lambda x: x.evaluate(instance),
            reverse=True,
        )

        # set population size & number of generations according to number of sol events (assumes each random solution has the same number of sol_events)
        # POPULATION_SIZE = min(POPULATION_SIZE, len(random_1000[0].sol_events) * 2)
        NGEN = max(NGEN, int(len(random_1000[0].sol_events) // 1.5))

        print(f"population size = {POPULATION_SIZE}\nnumber of generations = {NGEN}")

        population = random_1000[:POPULATION_SIZE]

        best_random_solution = deepcopy(population[0])

    random.shuffle(population)

    best_solution = None

    print("Generation Begin!")
    for idx in range(NGEN):
        start_time = time.time()
        # Select the best solutions from the population.
        num_selection = (
            num_selected_for_crossover
            if num_selected_for_crossover
            else int(len(population) // 2)
        )
        # ensure even number
        num_selection = num_selection + 1 if (num_selection % 2 != 0) else num_selection

        tournament_start_time = time.time()
        selected_solutions = tournament_selection(
            population, instance, num_selection, sample_size=tournament_size
        )
        tournament_elapsed_time = time.time() - tournament_start_time

        # Crossover the selected solutions to produce new offspring.
        crossover_start_time = time.time()
        offspring: list[Solution] = []
        for i in range(0, len(selected_solutions), 2):
            new_solution1, new_solution2 = crossover(
                selected_solutions[i],
                selected_solutions[i + 1],
                uniform_percentage=crossover_uniform_percentage,
            )
            offspring.append(new_solution1)
            offspring.append(new_solution2)
        crossover_elapsed_time = time.time() - crossover_start_time

        # Mutate the offspring.
        mutation_start_time = time.time()
        for offspring_solution in offspring:
            if random.random() < mutation_rate:
                mutate(offspring_solution, instance)
        mutation_elapsed_time = time.time() - mutation_start_time

        # Evaluate the offspring.
        # for offspring_solution in offspring:
        #     offspring_solution.evaluate(instance)

        # Add the offspring to the population and remove the worst individuals.

        parents_and_children = selected_solutions + offspring

        diff = POPULATION_SIZE - len(parents_and_children)
        print(f"diff = {diff}")
        population = (
            parents_and_children + random.choices(population, k=diff)
            if diff > 0
            else parents_and_children[:POPULATION_SIZE]
        )

        # print([x.evaluate(instance) for x in population])

        # Check if a satisfactory solution has been found.
        max_start_time = time.time()
        best_solution = deepcopy(
            max(population, key=lambda solution: solution.evaluate(instance))
        )
        max_elapsed_time = time.time() - max_start_time

        if best_solution.is_feasible_and_solves_objectives():
            if not input_solution:
                instance.evaluate_solution(best_random_solution.sol_events, debug=True)
                print("\nbest random: ", best_random_solution.cost)
            return best_solution.sol_events

        if best_solution.is_feasible() and not sol_changes_made:
            instance.evaluate_solution(best_solution.sol_events, debug=True)
            print("\nbest feasible  solution so far")
            for sol in population:
                sol.mode = Mode.Soft
                sol.needs_eval_update = True
            sol_changes_made = True

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(
            f"Generation: {idx + 1} tourn time: {tournament_elapsed_time} cross time: {crossover_elapsed_time} mut time: {mutation_elapsed_time} max func time {max_elapsed_time} total time: {elapsed_time} best sol: {best_solution.cost}"
        )

    if not input_solution and not input_population:
        instance.evaluate_solution(best_random_solution.sol_events, debug=True)
        print("\nbest random: ", best_random_solution.cost)

    return best_solution.sol_events


if __name__ == "__main__":
    from pathlib import Path

    root_dir = Path(__file__).parent.parent.parent.parent
    data_dir = root_dir.joinpath("data/ALL_INSTANCES")

    dataset_sudoku4x4 = XHSTTS(data_dir.joinpath("ArtificialSudoku4x4.xml"))
    dataset_abramson15 = XHSTTS(data_dir.joinpath("ArtificialAbramson15.xml"))
    dataset_brazil3 = XHSTTS(data_dir.joinpath("BrazilInstance3.xml"))
    aus_bghs98 = XHSTTS(data_dir.joinpath("AustraliaBGHS98.xml"))
    aus_sahs96 = XHSTTS(data_dir.joinpath("AustraliaSAHS96.xml"))
    aus_tes99 = XHSTTS(data_dir.joinpath("AustraliaTES99.xml"))
    italy4 = XHSTTS(data_dir.joinpath("ItalyInstance4.xml"))
    hdtt4 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt4.xml"))
    hdtt5 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt5.xml"))
    hdtt6 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt6.xml"))
    hdtt7 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt7.xml"))
    hdtt8 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt8.xml"))
    brazil2 = XHSTTS(data_dir.joinpath("BrazilInstance2.xml"))
    lewitt = XHSTTS(data_dir.joinpath("SouthAfricaLewitt2009.xml"))
    woodlands = XHSTTS(data_dir.joinpath("SouthAfricaWoodlands2009.xml"))
    spainschool = XHSTTS(data_dir.joinpath("SpainSchool.xml"))

    dataset_names = {
        dataset_sudoku4x4: "ArtificialSudoku4x4",
        dataset_abramson15: "ArtificialAbramson15",
        dataset_brazil3: "BrazilInstance3.xml",
    }

    for dataset in (
        # spainschool,
        lewitt,
        # hdtt8,
        # dataset_sudoku4x4,
        # italy4,
        # dataset_abramson15,
        # aus_bghs98,
        # aus_sahs96,
        # aus_tes99,
    ):  # (dataset_sudoku4x4,):  # dataset_abramson15):
        random.seed(23)

        assert dataset.num_instances() == 1

        instance = dataset.get_instance(index=0)

        # get solution
        result = genetic_algorithm(instance)

        # evaluate
        evaluation = instance.evaluate_solution(result, debug=True)

        print(
            f"\n---Genetic Evaluation ({instance.name})---\n",
            evaluation,
        )

        # save the solution as an xml file
        solutions_dir = root_dir.joinpath("solutions")
        file_path = solutions_dir.joinpath(f"genetic_solution_{instance.name}.xml")
        XHSTTSInstance.sol_events_to_xml(result, instance, file_path)
