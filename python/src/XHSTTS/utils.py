from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from collections import namedtuple


class Cost_Function_Type(Enum):
    Linear = auto()
    Quadratic = auto()
    Step = auto()


class Mode(Enum):
    Hard = auto()
    Soft = auto()


def cost_function(deviation: int, cost_type: Cost_Function_Type):
    """
    deviation : is non-negative.
    """
    if cost_type == Cost_Function_Type.Linear:
        return deviation
    elif cost_type == Cost_Function_Type.Quadratic:
        return deviation**2
    elif cost_type == Cost_Function_Type.Step:
        return 1 if deviation >= 0 else 0
    else:
        raise TypeError(
            f"cost_type should be a Cost_Function_Type enum. Instead got {cost_type} of type {type(cost_type)}."
        )


def cost_function_to_enum(cost_function: str):
    if cost_function == "Linear":
        return Cost_Function_Type.Linear
    elif cost_function == "Quadratic":
        return Cost_Function_Type.Quadratic
    elif cost_function == "Step":
        return Cost_Function_Type.Step
    else:
        raise Exception(f"Invalid cost_function named {cost_function}")


def cost(deviation: int, constraint_weight: int, cost_type: Cost_Function_Type):
    """
    deviation : is non-negative.\n
    constraint_weight : ranges from 0 to 100 inclusive.
    """
    return constraint_weight * cost_function(deviation, cost_type)


@dataclass
class Cost:
    Infeasibility_Value: int
    Objective_Value: int


if __name__ == "__main__":
    pass
