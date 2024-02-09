import random
from Algorithms.random_algorithm import random_solution
from Algorithms.genetic2 import genetic_algorithm
from Algorithms.simulated_annealing import simulated_annealing
from XHSTTS.xhstts import XHSTTS, Constraint, XHSTTSInstance
from XHSTTS.utils import Cost
from copy import deepcopy


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


def swap(a, b):
    a, b = b, a


def local_search(
    instance: XHSTTSInstance, max_iterations: int = 10_000, sol_events=[]
) -> list[XHSTTSInstance.SolutionEvent]:
    best_random_solution = None
    if sol_events:
        current_solution = LocalSearchSolution(sol_events)
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
            current_solution = best_neighbor
            no_improvement = 0  # reset the no improvements
        else:
            # No improvement for 20 iterations, terminate the search
            no_improvement += 1
            # print(f"no consecutive improvements {no_improvement}")
            if no_improvement > 19:  # TODO make constant
                break

        # Check if a satisfactory solution has been found.
        if current_solution.is_feasible():
            return current_solution.sol_events

    if not sol_events:
        print("\nbest random: ", best_random_solution.cost)

    return current_solution.sol_events


if __name__ == "__main__":
    from pathlib import Path

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
    for idx in range(1, benchmark_dataset.num_instances()):
        instance = benchmark_dataset.get_instance(index=idx)
        print(instance.name)
        if instance.name in (
            # "BrazilInstance2",
            # "FalkonG2012",
            # "StPaul",
            # "GreeceHighSchool1",
            # "Kottenpark2003",
            "Lewitt2009",
        ):
            print(
                f"-----{instance.name}   {len(instance.Constraints)} constraints-----"
            )

            genetic_result = genetic_algorithm(instance)

            evaluation = instance.evaluate_solution(genetic_result, debug=True)

            print(
                f"\n---Genetic Evaluation ({instance.name})---\n",
                evaluation,
            )

            # perform local search
            local_search_result = local_search(instance, sol_events=genetic_result)

            # evaluate local search result
            evaluation = instance.evaluate_solution(local_search_result, debug=True)

            print(
                f"\n---Local Search Benchmark Evaluation {instance.name} ---\n",
                evaluation,
                "\n",
            )

            # perform annealing
            annealing_result = simulated_annealing(instance, local_search_result)

            # evaluate simulated annealing search result
            evaluation = instance.evaluate_solution(annealing_result, debug=True)

            print(
                f"\n---Simulated Annealing Benchmark Evaluation {instance.name} ---\n",
                evaluation,
                "\n",
            )

    # for dataset in (
    #     dataset_sudoku4x4,
    #     dataset_abramson15,
    #     dataset_brazil3,
    # ):  #  (dataset_sudoku4x4, dataset_abramson15):  # dataset_abramson15):  benchmark_dataset,
    #     random.seed(23)

    #     assert dataset.num_instances() == 1

    #     instance = dataset.get_instance(index=0)

    #     from Algorithms.genetic2 import genetic_algorithm

    #     genetic_result = genetic_algorithm(instance)

    #     evaluation = instance.evaluate_solution(genetic_result, debug=True)

    #     print(
    #         f"\n---Genetic Evaluation ({dataset_names[dataset]})---\n",
    #         evaluation,
    #     )

    #     # perform local search
    #     local_search_result = local_search(instance, sol_events=genetic_result)

    #     # evaluate local search result
    #     evaluation = instance.evaluate_solution(local_search_result, debug=True)

    #     print(
    #         f"\n---Local Search Evaluation ({dataset_names[dataset]})---\n",
    #         evaluation,
    #     )

    #     print(len(local_search_result))

    # # perform local search
    # local_search_result = local_search(instance, sol_events=local_search_result)

    # # evaluate local search result
    # evaluation = instance.evaluate_solution(local_search_result, debug=True)

    # print(
    #     f"\n---Local Search Evaluation ({dataset_names[dataset]})---\n",
    #     evaluation,
    # )
