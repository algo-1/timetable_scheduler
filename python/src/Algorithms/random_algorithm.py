# TODO - validate solution - must obey XHSTTS rules

# we want to generate solution events where we make asignments where necessary without caring about constraints - except assign constraints since we want to generate a solution event
#
# randomly assign time resources to all instance events where necessary in accordance to the rules (not constraints)
# split some events into smaller pieces randomly too in accordance to the rules (not constraints)
from collections import defaultdict
from itertools import cycle
import random

from Algorithms.utils import get_consecutive_times
from XHSTTS.xhstts import XHSTTS, XHSTTSInstance


def generate_n_durations(N, input_total_duration, min_duration, max_duration):
    durations = []
    total_duration = input_total_duration

    for index in range(N - 1):
        integer = random.randint(
            1,
            min(total_duration - (N - index - 1), max_duration),
        )  # ensure each duration is at least 1
        durations.append(integer)
        total_duration -= integer

    # The last integer is the remaining value of total_duration
    durations.append(total_duration)

    return durations


def random_split(sol_events: list[XHSTTSInstance.SolutionEvent]):
    split_sol_events = []
    for event in sol_events:
        if (
            event.SplitMaxAmount > 0
            and event.Duration > 1
            and event.SplitMinAmount <= min(event.Duration, event.SplitMaxAmount)
        ):
            # if constraint specifies an exact split amount
            if (
                event.SplitMinAmount == event.SplitMaxAmount
                and event.Duration % event.SplitMaxAmount == 0
            ):
                num_splits = event.SplitMaxAmount
                # if constraint specifies an exact duration
                if (
                    event.SplitMinDuration == event.SplitMaxDuration
                    and event.SplitMaxAmount * event.SplitMaxDuration == event.Duration
                ):
                    for _ in range(num_splits):
                        split_event = event._replace(Duration=event.SplitMaxDuration)
                        split_sol_events.append(split_event)
                else:
                    durations = generate_n_durations(
                        num_splits,
                        event.Duration,
                        event.SplitMinAmount,
                        event.SplitMaxDuration,
                    )
                    for split_duration in durations:
                        split_event = event._replace(Duration=split_duration)
                        split_sol_events.append(split_event)
            else:
                num_splits = random.randint(
                    event.SplitMinAmount, min(event.Duration, event.SplitMaxAmount)
                )
                durations = generate_n_durations(
                    num_splits,
                    event.Duration,
                    event.SplitMinAmount,
                    event.SplitMaxDuration,
                )

                for split_duration in durations:
                    split_event = event._replace(Duration=split_duration)
                    split_sol_events.append(split_event)
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


def get_non_clashing_time(
    sol_event_resources,
    resource_busy_times,
    times,
):
    busy_times = set()
    for resource_ref, _ in sol_event_resources:
        if resource_ref in resource_busy_times:
            busy_times |= resource_busy_times[resource_ref]

    for t in times:
        if t not in busy_times:
            return t

    return random.choice(list(times))


def assign_times(
    sol_events: list[XHSTTSInstance.SolutionEvent], instance: XHSTTSInstance
):
    instance_time_refs = list(instance.Times.keys())
    linked_events = instance.linked_events
    merged_sets = []
    resource_busy_times = defaultdict(set)

    # merge sets of linked times if they share a time
    for s in linked_events:
        # Check if s has any common element with existing merged sets
        merged = False
        for merged_set in merged_sets:
            if s in merged_set:
                merged_set.update(s)
                merged = True
                break

        # If s has no common element with existing merged sets, add it as a new merged set
        if not merged:
            merged_sets.append(s)

    # assign each linked set a time try use a pre-assigned time / preferred time else random
    linked_assigned_times = [None] * len(merged_sets)
    for idx, s in enumerate(merged_sets):
        assigned = False
        for ref in s:
            if instance.Events[ref].PreAssignedTimeReference:
                linked_assigned_times[idx] = instance.Event[
                    ref
                ].PreAssignedTimeReference
                assigned = True
                break
        for ref in s:
            if instance.Events[ref].PreferredTimes:
                linked_assigned_times[idx] = random.choice(
                    list(instance.Events[ref].PreferredTimes)
                )
                assigned = True
                break
        if not assigned:
            linked_assigned_times[idx] = random.choice(instance_time_refs)

    result = []

    for sol_event in sol_events:
        if sol_event.TimeReference:
            new_sol_event = sol_event
            result.append(sol_event)
        else:
            # try assign linked time
            link = False
            for idx, s in enumerate(merged_sets):
                if sol_event.InstanceEventReference in s:
                    new_sol_event = sol_event._replace(
                        TimeReference=linked_assigned_times[idx]
                    )
                    result.append(new_sol_event)
                    link = True
                    break
            # try assign preferred time
            if not link:
                if sol_event.PreferredTimes:
                    new_sol_event = sol_event._replace(
                        TimeReference=get_non_clashing_time(
                            sol_event.Resources,
                            resource_busy_times,
                            sol_event.PreferredTimes,
                        )
                    )
                    result.append(
                        # sol_event._replace(
                        #     TimeReference=random.choice(list(sol_event.PreferredTimes))
                        # )
                        new_sol_event
                    )
                else:
                    # assign random time
                    new_sol_event = sol_event._replace(
                        # TimeReference=random.choice(instance_time_refs)
                        TimeReference=get_non_clashing_time(
                            sol_event.Resources,
                            resource_busy_times,
                            instance_time_refs,
                        )
                    )
                    result.append(new_sol_event)

        # add the busy times for all the events resources
        for t in get_consecutive_times(
            instance, new_sol_event.TimeReference, new_sol_event.Duration
        ):
            for resource_ref, _ in new_sol_event.Resources:
                resource_busy_times[resource_ref].add(t)

    return result


def find_resource_type(resources: list[XHSTTSInstance.EventResource], role: str):
    for elem in resources:
        if elem.Role == role:
            return elem.ResourceTypeReference
    raise Exception(f"Did not find the '{role}' role in resources.")


def get_non_busy_resource(
    instance,
    event: XHSTTSInstance.SolutionEvent,
    resource,
    resource_busy_times,
):
    partitioned_resources_refs = instance.partitioned_resources_refs
    resource_ref_options = partitioned_resources_refs[
        find_resource_type(
            instance.Events[event.InstanceEventReference].Resources,
            resource.Role,
        )
    ]

    for ref in resource_ref_options:
        # resource not used yet at all
        if ref not in resource_busy_times:
            return ref

        # chwck if resource is busy
        busy_times = resource_busy_times[ref]
        is_busy = False
        for t in get_consecutive_times(instance, event.TimeReference, event.Duration):
            if t in busy_times:
                is_busy = True
                break

        # return resource if not busy
        if not is_busy:
            return ref

    # if not possible with the arrangement of resources, just return a random resource
    return random.choice(resource_ref_options)


def assign_random_resources(
    sol_events: list[XHSTTSInstance.SolutionEvent], instance: XHSTTSInstance
):
    resource_busy_times = defaultdict(set)

    new_sol_events = []
    for event in sol_events:
        updated_resources = []
        for resource in event.Resources:
            resource_ref = None
            if resource.Reference:
                resource_ref = resource.Reference
                updated_resources.append(resource)
            else:
                resource_ref = get_non_busy_resource(
                    instance, event, resource, resource_busy_times
                )
                updated_resources.append(resource._replace(Reference=resource_ref))

            for t in get_consecutive_times(
                instance, event.TimeReference, event.Duration
            ):

                resource_busy_times[resource_ref].add(t)

        new_sol_events.append(event._replace(Resources=updated_resources))

    return new_sol_events


def random_solution(instance: XHSTTSInstance) -> list[XHSTTSInstance.SolutionEvent]:
    instance_events = instance.Events

    # convert to solution event object and split based on constraints
    sol_events = random_split(
        list(map(XHSTTSInstance.create_solution_event, list(instance_events.values())))
    )

    # assign time and resources randomly anywhere necessary
    result: list[XHSTTSInstance.SolutionEvent] = assign_random_resources(
        assign_times(sol_events, instance), instance
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
    italy1 = XHSTTS(data_dir.joinpath("ItalyInstance1.xml"))
    italy4 = XHSTTS(data_dir.joinpath("ItalyInstance4.xml"))
    aus_sahs96 = XHSTTS(data_dir.joinpath("AustraliaSAHS96.xml"))
    aus_tes99 = XHSTTS(data_dir.joinpath("AustraliaTES99.xml"))
    stpaul = XHSTTS(data_dir.joinpath("EnglandStPaul.xml"))
    spainschool = XHSTTS(data_dir.joinpath("SpainSchool.xml"))
    brazil2 = XHSTTS(data_dir.joinpath("BrazilInstance2.xml"))
    brazil3 = XHSTTS(data_dir.joinpath("BrazilInstance3.xml"))
    brazil4 = XHSTTS(data_dir.joinpath("BrazilInstance4.xml"))
    finlandcollege = XHSTTS(data_dir.joinpath("FinlandCollege.xml"))
    finlandhigh = XHSTTS(data_dir.joinpath("FinlandHighSchool.xml"))
    finlandsecondary = XHSTTS(data_dir.joinpath("FinlandSecondarySchool.xml"))
    greekhigh1 = XHSTTS(data_dir.joinpath("GreeceHighSchool1.xml"))
    greekthirdhigh2010 = XHSTTS(
        data_dir.joinpath("GreeceThirdHighSchoolPatras2010.xml")
    )
    lewitt = XHSTTS(data_dir.joinpath("SouthAfricaLewitt2009.xml"))
    kosova = XHSTTS(data_dir.joinpath("KosovaInstance1.xml"))
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
        # stpaul,
        # aus_bghs98,
        # aus_sahs96,
        # aus_tes99,
        brazil2,
        brazil3,
        brazil4,
        lewitt,
        dataset_sudoku4x4,
        dataset_abramson15,
        woodlands,
        hdtt4,
        hdtt5,
        hdtt6,
        hdtt7,
        hdtt8,
        finlandcollege,
        finlandhigh,
        finlandsecondary,
        greekhigh1,
        greekthirdhigh2010,
        kosova,
        italy4,
        spainschool,
    ):  # (dataset_sudoku4x4, dataset_abramson15, dataset_brazil3):
        random.seed(23)

        assert dataset.num_instances() == 1

        instance = dataset.get_instance(index=0)

        # get solution
        result = random_solution(
            instance
        )  # testing eval on benchmark solutions instance.get_solutions()[-1]

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
