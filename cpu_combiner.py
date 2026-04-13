from dataclasses import dataclass
from math import inf
from typing import List, Optional


@dataclass
class MetaStage:
    k: int
    num_states: int
    M: List[List[float]]
    argmin: List[List[int]]


def merge_metastages(left: MetaStage, right: MetaStage) -> MetaStage:
    if left.num_states != right.num_states:
        raise ValueError("state count mismatch")

    S = left.num_states
    out_M = [[inf for _ in range(S)] for _ in range(S)]
    out_argmin = [[-1 for _ in range(S)] for _ in range(S)]

    for sb in range(S):
        for se in range(S):
            best = inf
            best_mid = -1

            for mid in range(S):
                a = left.M[sb][mid]
                b = right.M[mid][se]

                if a == inf or b == inf:
                    continue

                cand = a + b
                if cand < best:
                    best = cand
                    best_mid = mid

            out_M[sb][se] = best
            out_argmin[sb][se] = best_mid

    # Normalize by subtracting block minimum — keeps metrics bounded across
    # many merges and matches the normalization in the CUDA kernel.
    block_min = min(
        out_M[s][d]
        for s in range(S)
        for d in range(S)
        if out_M[s][d] != inf
    )
    if block_min != inf:
        for s in range(S):
            for d in range(S):
                if out_M[s][d] != inf:
                    out_M[s][d] -= block_min

    return MetaStage(
        k=left.k + right.k,
        num_states=S,
        M=out_M,
        argmin=out_argmin,
    )


def reduce_one_level(stages: List[MetaStage]) -> List[MetaStage]:
    out = []
    i = 0
    while i + 1 < len(stages):
        out.append(merge_metastages(stages[i], stages[i + 1]))
        i += 2
    if i < len(stages):
        out.append(stages[i])
    return out


def reduce_tree(stages: List[MetaStage]) -> MetaStage:
    if not stages:
        raise ValueError("need at least one stage")
    cur = stages
    while len(cur) > 1:
        cur = reduce_one_level(cur)
    return cur[0]


def best_tailbiting_state(final_stage: MetaStage):
    S = final_stage.num_states
    best_metric = inf
    best_state = -1
    for s in range(S):
        val = final_stage.M[s][s]
        if val < best_metric:
            best_metric = val
            best_state = s
    return best_state, best_metric


def traceback(stages: List[MetaStage], start_state: int) -> List[int]:
    """
    Reconstruct the best path through the trellis given the list of MetaStages
    and the best starting (= ending, tail-biting) state.

    Args:
        stages: the original list of MetaStages before tree reduction,
                i.e. the per-step stages in order.
        start_state: the best tail-biting state from best_tailbiting_state().

    Returns:
        path: list of states visited at each step, length len(stages) + 1.
              path[0] == path[-1] == start_state (tail-biting property).
    """
    path = [start_state]
    current = start_state

    for stage in stages:
        # argmin[current][d] tells us the best next state d from current
        # but we stored argmin as the intermediate state during merging.
        # For a single basic stage, argmin[s][d] = -1 (no intermediate),
        # so we find the best d directly from M[current].
        row = stage.M[current]
        best_d = min(range(stage.num_states), key=lambda d: row[d])
        path.append(best_d)
        current = best_d

    return path