from dataclasses import dataclass
from math import inf
from typing import List


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
        out.append(stages[i])  # carry for odd stages

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
        val = final_stage.M[s][s] # only care ab diagonal for tail-biting
        if val < best_metric:
            best_metric = val
            best_state = s

    return best_state, best_metric