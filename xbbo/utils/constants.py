from prometheus_client import Enum
VERY_SMALL_NUMBER = 1e-10
MAXINT = 2 ** 31 - 1


class Key():
    REGRET_VAL = "regret_validation"
    REGRET_TEST = "regret_test"
    COST = "cost"