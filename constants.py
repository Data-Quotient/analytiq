from enum import Enum

class DQ_RULES(Enum):
    RANGE_CHECK = "Range Check"
    NULL_CHECK = "Null Check"
    UNIQUENESS_CHECK = "Uniqueness Check"
    CUSTOM_LAMBDA = "Custom Lambda"