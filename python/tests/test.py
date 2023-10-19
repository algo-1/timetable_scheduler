# Inspired by https://github.com/danfergo/HSTTScheduler/blob/master/HSTTScheduler/tests/XHSTTImporterArtificialAbramson15Test.cpp

import pytest
from pathlib import Path
from python.src.XHSTTS.XHSTTS import XHSTTS

base_dir = Path(__file__).parent.parent.parent

data_dir = base_dir.joinpath(Path("data"))


@pytest.fixture
def artificial_abrahamson_15_dataset_path():
    return data_dir.joinpath("ALL_INSTANCES/ArtificialAbramson15.xml")


def test_artificial_abrahamson_15_dataset(artificial_abrahamson_15_dataset_path):
    dataset = XHSTTS(artificial_abrahamson_15_dataset_path)
    instance = dataset.get_first_instance()

    num_events = len(instance.get_events())
    num_times = len(instance.get_times())
    num_resources = len(instance.get_resources())
    num_constraints = len(instance.get_constraints())
    num_solutions = len(instance.get_solutions())  # solutions referencing that instance

    # Assuming that a solution is a list of solution events;
    num_events_in_first_solution = len(dataset.get_solutions()[0])
    num_events_in_second_solution = len(dataset.get_solutions()[1])

    cost1 = XHSTTS.evaluate_solution(
        instance.get_solutions()[0], instance.get_constraints()
    )
    cost2 = XHSTTS.evaluate_solution(
        instance.get_solutions()[1], instance.get_constraints()
    )

    assert num_events == 450
    assert num_times == 30
    assert num_resources == 15 + 15 + 15
    assert num_constraints == 2
    assert num_solutions == 2
    assert num_events_in_first_solution == 450
    assert num_events_in_second_solution == 450
    assert cost1 == (0, 0)
    assert cost2 == (3, 0)
