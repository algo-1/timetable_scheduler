from contextlib import redirect_stdout
import concurrent.futures
import io
from multiprocessing import current_process
from pathlib import Path
import random
import time

from Algorithms.genetic2 import genetic_algorithm
from Algorithms.local_search import local_search
from Algorithms.random_algorithm import random_solution
from Algorithms.simulated_annealing import simulated_annealing
from Algorithms.tabu_search import ils_tabu, tabu_search
from Algorithms.vns import variable_neighborhood_search
from XHSTTS.xhstts import XHSTTS, XHSTTSInstance

root_dir = Path(__file__).parent.parent.parent.parent
data_dir = root_dir.joinpath("data/ALL_INSTANCES")
dataset_sudoku4x4 = XHSTTS(data_dir.joinpath("ArtificialSudoku4x4.xml"))
dataset_abramson15 = XHSTTS(data_dir.joinpath("ArtificialAbramson15.xml"))
dataset_brazil3 = XHSTTS(data_dir.joinpath("BrazilInstance3.xml"))
benchmark_dataset = XHSTTS(data_dir.parent.joinpath("XHSTT-2014.xml"))
hdtt4 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt4.xml"))
hdtt5 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt5.xml"))
hdtt6 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt6.xml"))
hdtt7 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt7.xml"))
hdtt8 = XHSTTS(data_dir.joinpath("ArtificialORLibrary-hdtt8.xml"))
benchmark_instances = [
    benchmark_dataset.get_instance(index=idx)
    for idx in range(benchmark_dataset.num_instances())
]
assert len(benchmark_instances) == 25

test_instances = [
    hdtt4.get_first_instance(),
    hdtt5.get_first_instance(),
    hdtt6.get_first_instance(),
    hdtt7.get_first_instance(),
    hdtt8.get_first_instance(),
    dataset_sudoku4x4.get_first_instance(),
    benchmark_instances[3],
    benchmark_instances[4],
    benchmark_instances[5],
    benchmark_instances[21],
    # benchmark_instances[0],
    dataset_abramson15.get_first_instance(),
    benchmark_instances[10],
    benchmark_instances[11],
    benchmark_instances[12],
    benchmark_instances[13],
    benchmark_instances[14],
    benchmark_instances[15],
    benchmark_instances[16],
    # benchmark_instances[17],
    benchmark_instances[22],
    benchmark_instances[23],
    # benchmark_instances[24],
]


def process_instance(instance: XHSTTSInstance):
    root_dir = Path(__file__).parent.parent.parent.parent
    solutions_dir = root_dir.joinpath("final_benchmark_solutions")
    print(
        f"Process {current_process().name} started working on {instance.name}",
        flush=True,
    )

    with io.StringIO() as buffer, redirect_stdout(buffer):
        start_time = time.time()

        print(f"-----{instance.name}   {len(instance .Constraints)} constraints-----")

        ils_tabu_sd_result = ils_tabu(instance)

        print("ILS-Tabu-SD result below")
        evaluation = instance.evaluate_solution(ils_tabu_sd_result, debug=True)
        print(
            f"\n---ILS-Tabu-SD Evaluation ({instance.name})---\n",
            evaluation,
        )

        vns_res = (
            variable_neighborhood_search(instance, ils_tabu_sd_result)
            if instance.name != "Spanish school"
            else variable_neighborhood_search(
                instance, ils_tabu_sd_result, assign_resources=True
            )
        )

        evaluation = instance.evaluate_solution(vns_res, debug=True)

        print(
            f"\n---VNS Evaluation ({instance.name})---\n",
            evaluation,
        )

        annealing_result = simulated_annealing(instance, vns_res)
        evaluation = instance.evaluate_solution(annealing_result, debug=True)

        print(
            f"\n---Simulated Annealing Benchmark ({instance.name}) Evaluation ---\n",
            evaluation,
            "\n",
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{instance.name} finished in {elapsed_time} seconds.")

        # Write the output and solution XML to files
        output_path = solutions_dir.joinpath(Path(f"danu3_output_{instance.name}.txt"))
        xml_path = solutions_dir.joinpath(Path(f"danu3_solution_{instance.name}.xml"))

        with open(output_path, "w") as output_file:
            output_file.write(buffer.getvalue())

        XHSTTSInstance.sol_events_to_xml(annealing_result, instance, xml_path)

    return f"{instance.name} finished"


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_instance, task) for task in test_instances]
        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as e:
                print(e, flush=True)
