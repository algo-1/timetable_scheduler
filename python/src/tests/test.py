# Inspired by https://github.com/danfergo/HSTTScheduler/blob/master/HSTTScheduler/tests/XHSTTImporterArtificialAbramson15Test.cpp

import os
import pytest
from pathlib import Path
from XHSTTS.xhstts import XHSTTS, XHSTTSInstance
from XHSTTS.utils import Cost

base_dir = Path(__file__).parent.parent.parent.parent

data_dir = base_dir.joinpath(Path("data"))

all_instances_dir = data_dir.joinpath("ALL_INSTANCES/")


@pytest.fixture
def artificial_abrahamson_15_dataset():
    return XHSTTS(data_dir.joinpath("ALL_INSTANCES/ArtificialAbramson15.xml"))


@pytest.fixture
def artificial_sudoku_4x4_dataset():
    return XHSTTS(data_dir.joinpath("ALL_INSTANCES/ArtificialSudoku4x4.xml"))


@pytest.fixture
def all_instances():
    """
    All instances except NetherlandsKottenpark2008 because it has the notimoplemented OrderEventsConstraint
    """
    files = [
        all_instances_dir.joinpath(file)
        for file in sorted(os.listdir(all_instances_dir))
        if file != "NetherlandsKottenpark2008.xml"
    ]

    return [XHSTTS(dataset).get_first_instance() for dataset in files]


@pytest.fixture
def real_evaluation(all_instances):
    costs = [Cost(0, 0) for _ in range(len(all_instances))]

    costs[0] = Cost(3, 0)  # Abramson15
    costs[1] = Cost(32, 0)  # ArtificialAll11
    costs[2] = Cost(197, 0)  # ArtificialAll15
    costs[9] = Cost(0, 128)  # Australia BGHS98
    costs[11] = Cost(0, 20)  # Australia TES99
    costs[12] = Cost(0, 41)  # Brazil Instance 1
    costs[13] = Cost(0, 5)  # Brazil Instance 2
    costs[14] = Cost(0, 24)  # Brazil Instance 3
    costs[15] = Cost(0, 51)  # Brazil Instance 4
    costs[16] = Cost(0, 19)  # Brazil Instance 5
    costs[17] = Cost(0, 35)  # Brazil Instance 6
    costs[18] = Cost(0, 53)  # Brazil Instance 7
    costs[19] = Cost(0, 13)  # CzechVillageSchool
    costs[20] = Cost(0, 1263)  # Denmark FalkonG2012
    costs[21] = Cost(7, 2330)  # Denmark HasserG2012
    costs[22] = Cost(2, 2323)  # Denmark VejenG2009
    costs[23] = Cost(2, 1410)  # England StPaul
    costs[26] = Cost(0, 3)  # FinlandElemantarySchool
    costs[28] = Cost(0, 77)  # FinlandSecondarySchool
    costs[34] = Cost(0, 5)  # WesternGreeceUniversityInstance3
    costs[35] = Cost(0, 2)  # WesternGreeceUniversityInstance4 # my solution
    costs[37] = Cost(0, 12)  # ItalyInstance1
    costs[38] = Cost(0, 27)  # ItalyInstance4
    costs[40] = Cost(1, 566)  # Netherlands  GEPRO
    costs[41] = Cost(
        0, 29804
    )  # Kottenpark2003  -- only weird one, evaluation is differrent. Weird as idle times constraints works for evey other dataset
    costs[42] = Cost(0, 425)  # Kottenpark2005
    costs[43] = Cost(0, 1620)  # Kottenpark2009
    costs[45] = Cost(0, 0)  # SA Woodlands2009
    costs[46] = Cost(0, 335)  # Spain School
    costs[47] = Cost(0, 101)  # USA WHS09

    return {k: v for k, v in zip(all_instances, costs)}


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
        num_events_in_second_solution == 450
    )  # 447 in xml but missing events are now added during parsing
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


def test_evaluate_solution(all_instances, real_evaluation):
    for idx, instance in enumerate(all_instances):
        print(instance.name)
        assert (
            instance.evaluate_solution(instance.get_solutions()[-1], debug=True)
            == real_evaluation[instance]
        ), f"{instance.name} (idx={idx}) failed"
