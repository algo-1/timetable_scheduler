# Inspired by https://github.com/danfergo/HSTTScheduler/blob/master/HSTTScheduler/tests/XHSTTImporterArtificialAbramson15Test.cpp

import pytest
from pathlib import Path
from python.src.XHSTTS.XHSTTS import XHSTTS, XHSTTSInstance
from python.src.XHSTTS.utils import Cost

base_dir = Path(__file__).parent.parent.parent

data_dir = base_dir.joinpath(Path("data"))


@pytest.fixture
def artificial_abrahamson_15_dataset():
    return XHSTTS(data_dir.joinpath("ALL_INSTANCES/ArtificialAbramson15.xml"))


@pytest.fixture
def artificial_sudoku_4x4_dataset():
    return XHSTTS(data_dir.joinpath("ALL_INSTANCES/ArtificialSudoku4x4.xml"))


def test_artificial_abrahamson_15_dataset(artificial_abrahamson_15_dataset):
    dataset = artificial_abrahamson_15_dataset
    instance: XHSTTSInstance = dataset.get_first_instance()

    num_events = len(instance.get_events())
    num_times = len(instance.get_times())
    num_resources = len(instance.get_resources())
    num_constraints = len(instance.get_constraints())
    num_solutions = len(instance.get_solutions())  # solutions referencing that instance

    # Assuming that a solution is a list of solution events;
    num_events_in_first_solution = len(instance.get_solutions()[0])
    num_events_in_second_solution = len(instance.get_solutions()[1])

    cost1 = instance.evaluate_solution(instance.get_solutions()[0])
    cost2 = instance.evaluate_solution(instance.get_solutions()[1])

    assert num_events == 450
    assert num_times == 30
    assert num_resources == 15 + 15 + 15
    assert num_constraints == 2
    assert num_solutions == 2
    assert num_events_in_first_solution == 450
    assert (
        num_events_in_second_solution == 447
    )  # incomplete solution as per description and manual checking
    assert cost1 == Cost(0, 0)
    assert cost2 == Cost(3, 0)


def test_sudoku_4x4_dataset(artificial_sudoku_4x4_dataset):
    dataset = artificial_sudoku_4x4_dataset
    instance: XHSTTSInstance = dataset.get_first_instance()

    num_events = len(instance.get_events())
    num_times = len(instance.get_times())
    num_resources = len(instance.get_resources())
    num_constraints = len(instance.get_constraints())
    num_solutions = len(instance.get_solutions())  # solutions referencing that instance

    num_events_in_first_solution = len(instance.get_solutions()[0])

    cost1 = instance.evaluate_solution(instance.get_solutions()[0])

    assert num_events == 16
    assert num_times == 4
    assert num_resources == 4 * 3
    assert num_constraints == 10
    assert num_solutions == 1
    assert num_events_in_first_solution == 16
    assert cost1 == Cost(0, 0)
