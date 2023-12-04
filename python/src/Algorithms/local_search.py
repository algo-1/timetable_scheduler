import random
from Algorithms.random_algorithm import random_solution
from XHSTTS.xhstts import XHSTTS, XHSTTSInstance
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
                        if (
                            other_event_resource_idx not in selected
                            and instance.get_resources()[
                                other_event.Resources[
                                    other_event_resource_idx
                                ].Reference
                            ].ResourceTypeReference
                            == instance.get_resources()[
                                event.Resources[i].Reference
                            ].ResourceTypeReference
                        ):
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


def local_search(
    instance: XHSTTSInstance,
    max_iterations: int = 10_000,
) -> LocalSearchSolution:
    best_random_solution: LocalSearchSolution = sorted(
        [LocalSearchSolution(random_solution(instance)) for _ in range(1000)],
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
        print(sorted([x.evaluate(instance) for x in neighbors], reverse=True))

        # Check if the best neighbor is an improvement
        if best_neighbor.evaluate(instance) > current_solution.evaluate(instance):
            current_solution = best_neighbor
            no_improvement = 0  # reset the no improvements
        else:
            # No improvement for 20 iterations, terminate the search
            no_improvement += 1
            print(f"no consecutive improvements {no_improvement}")
            if no_improvement > 19:  # TODO make constant
                break

        # Check if a satisfactory solution has been found.
        if current_solution.is_feasible():
            return current_solution

    print("\nbest random: ", best_random_solution.cost)

    return current_solution


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
        dataset_brazil3,
    ):  # (dataset_sudoku4x4, dataset_abramson15):  # dataset_abramson15):
        random.seed(23)

        assert dataset.num_instances() == 1

        instance = dataset.get_instance(index=0)

        # perform local search
        local_search_result = local_search(instance)

        # evaluate local search result
        evaluation = instance.evaluate_solution(local_search_result.sol_events)

        print(
            f"\n---Local Search Evaluation ({dataset_names[dataset]})---\n",
            evaluation,
            "\n",
        )
