"""Constant-zero reward for veRL's task-reward-disabled bare OPD arm."""


def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    **kwargs,
):
    del data_source, solution_str, ground_truth, extra_info, kwargs
    return 0.0
