# TODO - validate solution - must obey XHSTTS rules

# we want to generate solution events where we make asignments where necessary without caring about constraints - except assign constraints since we want to generate a solution event
#
# randomly assign time resources to all instance events where necessary in accordance to the rules (not constraints)
# split some events into smaller pieces randomly too in accordance to the rules (not constraints)
from collections import defaultdict
from itertools import cycle
import random

from XHSTTS.xhstts import XHSTTS, XHSTTSInstance


def random_split(sol_events: list[XHSTTSInstance.SolutionEvent]):
    """returns a list of solution events by randomly splitting instance events according to the rules (duration etc)"""
    # TODO: incorporate event.SplitMinDuration  & event.SplitMaxDuration? NOTE: must be callled before assigning times
    split_sol_events = []
    for event in sol_events:
        if (
            event.SplitMaxAmount > 0
            and event.Duration > 1
            # and event.SplitMinAmount != 0  # TODO: add if else to avoid div by 0 if choosing min amount split
            # and event.SplitMaxAmount != float("inf") we still want to split even if no split constraints -- hdtt problems for example
            and not event.TimeReference
        ):
            num_splits = min(
                event.SplitMaxAmount,
                event.Duration,
            )
            split_duration = event.Duration // num_splits

            for _ in range(num_splits - 1):
                split_event = event._replace(Duration=split_duration)
                split_sol_events.append(split_event)
            last_split_event = event._replace(
                Duration=event.Duration - (split_duration * (num_splits - 1))
            )
            split_sol_events.append(last_split_event)
        else:
            # no split necesary
            split_sol_events.append(event)
    return split_sol_events


def assign_random_times(
    sol_events: list[XHSTTSInstance.SolutionEvent], instance: XHSTTSInstance
):
    time_refs = list(instance.Times.keys())
    random.shuffle(time_refs)
    time_cycle = cycle(time_refs)

    return [
        event if event.TimeReference else event._replace(TimeReference=next(time_cycle))
        for event in sol_events
    ]


def find_resource_type(resources: list[XHSTTSInstance.EventResource], role: str):
    for elem in resources:
        if elem.Role == role:
            return elem.ResourceTypeReference
    raise Exception(f"Did not find the '{role}' role in resources.")


def assign_random_resources(
    sol_events: list[XHSTTSInstance.SolutionEvent], instance: XHSTTSInstance
):
    # TODO: do this partitioning in the instance not here
    partitioned_resources_refs = defaultdict(list)
    for ref, resource in instance.Resources.items():
        partitioned_resources_refs[resource.ResourceTypeReference].append(ref)
    new_sol_events = []
    for event in sol_events:
        resources = [
            (
                resource
                if resource.Reference
                else resource._replace(
                    Reference=random.choice(
                        partitioned_resources_refs[
                            find_resource_type(
                                instance.Events[event.InstanceEventReference].Resources,
                                resource.Role,
                            )
                        ]
                    )
                )
            )
            for resource in event.Resources
        ]
        new_sol_events.append(event._replace(Resources=resources))

    return new_sol_events


def random_solution(instance: XHSTTSInstance) -> list[XHSTTSInstance.SolutionEvent]:
    instance_events = instance.Events

    # convert to solution event object and split based on constraints
    sol_events = random_split(
        list(map(XHSTTSInstance.create_solution_event, list(instance_events.values())))
    )

    # assign time and resources randomly anywhere necessary
    result: list[XHSTTSInstance.SolutionEvent] = assign_random_resources(
        assign_random_times(sol_events, instance), instance
    )
    # print(list(map(lambda x: x.Reference, list(instance_events.values())[:5])))
    # print(list(map(lambda x: x.InstanceEventReference, result[:5])))
    return result


if __name__ == "__main__":
    from pathlib import Path

    root_dir = Path(__file__).parent.parent.parent.parent
    data_dir = root_dir.joinpath("data/ALL_INSTANCES")

    dataset_sudoku4x4 = XHSTTS(data_dir.joinpath("ArtificialSudoku4x4.xml"))
    dataset_abramson15 = XHSTTS(data_dir.joinpath("ArtificialAbramson15.xml"))
    dataset_brazil3 = XHSTTS(data_dir.joinpath("BrazilInstance3.xml"))
    aus_bghs98 = XHSTTS(data_dir.joinpath("AustraliaBGHS98.xml"))
    italy4 = XHSTTS(data_dir.joinpath("ItalyInstance4.xml"))
    aus_sahs96 = XHSTTS(data_dir.joinpath("AustraliaSAHS96.xml"))
    aus_tes99 = XHSTTS(data_dir.joinpath("AustraliaTES99.xml"))
    stpaul = XHSTTS(data_dir.joinpath("EnglandStPaul.xml"))
    spainschool = XHSTTS(data_dir.joinpath("SpainSchool.xml"))
    brazil2 = XHSTTS(data_dir.joinpath("BrazilInstance2.xml"))
    lewitt = XHSTTS(data_dir.joinpath("SouthAfricaLewitt2009.xml"))
    woodlands = XHSTTS(data_dir.joinpath("SouthAfricaWoodlands2009.xml"))
    hdtt4 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt4.xml"))
    hdtt5 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt5.xml"))
    hdtt6 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt6.xml"))
    hdtt7 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt7.xml"))
    hdtt8 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt8.xml"))

    dataset_names = {
        dataset_sudoku4x4: "ArtificialSudoku4x4",
        dataset_abramson15: "ArtificialAbramson15",
        dataset_brazil3: "BrazilInstance3.xml",
    }
    for dataset in (
        stpaul,
        italy4,
        aus_bghs98,
        aus_sahs96,
        aus_tes99,
        spainschool,
        brazil2,
        lewitt,
        dataset_sudoku4x4,
        dataset_abramson15,
        woodlands,
        hdtt4,
        hdtt5,
        hdtt6,
        hdtt7,
        hdtt8,
    ):  # (dataset_sudoku4x4, dataset_abramson15, dataset_brazil3):
        random.seed(23)

        assert dataset.num_instances() == 1

        instance = dataset.get_instance(index=0)

        # get solution
        result = instance.get_solutions()[
            -1
        ]  # random_solution( instance)  # testing eval on benchmark solutions instance.get_solutions()[-1]

        # evaluate
        print(f"\n\n--- Random Evaluation ({instance.name}) ---")

        evaluation = instance.evaluate_solution(result, debug=True)

        print("\n", evaluation)

        print("\n\n")

        # save the solution as an xml file
        # solutions_dir = root_dir.joinpath("solutions")
        # file_path = solutions_dir.joinpath(
        #     f"random_solution_{dataset_names[dataset]}.xml"
        # )
        # XHSTTSInstance.sol_events_to_xml(result, instance, file_path)
