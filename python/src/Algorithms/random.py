# TODO - validate solution - must obey XHSTTS rules

# we want to generate solution events where we make asignments where necessary without caring about constraints - except assign constraints since we want to generate a solution event
#
# randomly assign time resources to all instance events where necessary in accordance to the rules (not constraints)
# split some events into smaller pieces randomly too in accordance to the rules (not constraints)

from ..XHSTTS.XHSTTS import XHSTTS


def get_n_random_events_to_split():
    """
    returns a tuple of n random solution events and
    (len(instance_events) - n) solution events both based on the instance events.

    >>> get_n_random_events_to_split(3, [event1, event2, event3, event4, event5])
    ( [sol_event2, sol_event4, sol_event5], [sol_event1, sol_event3] )
    """
    pass


def random_split():
    """returns a list of solution events by randomly splitting instance events according to the rules (duration etc)"""
    pass


def assign_random_times():
    pass


def assign_random_resources():
    pass


if __name__ == "__main__":
    dataset = XHSTTS(
        "/Users/harry/tcd/fyp/timetabling_solver/data/ALL_INSTANCES/ArtificialAbramson15.xml"
    )

    assert dataset.num_instances() == 1

    instance = dataset.get_instance(index=0)

    instance_events = instance.Events

    n_events, rest = get_n_random_events_to_split(
        len(instance_events) // 3, instance_events
    )

    # n_events and rest should be new objects not point to the same ones as events - sol event objects preferably!
    sol_events = [rest]

    # split some events
    for event in n_events:
        sol_events.extend(random_split(event))

    # assign time and resources randomly anywhere necessary
    result = list(assign_random_resources(assign_random_times(sol_events)))

    evaluation = instance.evaluate_solution(instance.create_solution(result))

    print(evaluation)
