import numpy as np

import numpy as np


def DCG(Relevance):
    return np.sum(Relevance / np.log2(np.arange(2, len(Relevance) + 2)))


def nDCG(Recommendations, GT):
    Relevance = np.array(GetRelevantGT(Recommendations, GT), dtype=float)

    DCGScore = DCG(Relevance)

    # Ideal DCG: all relevant items (1s) at the top
    idealRanking = np.sort(Relevance)[::-1]
    IDCGScore = DCG(idealRanking)

    return DCGScore / IDCGScore if IDCGScore > 0 else 0


def GetRelevantGT(Recommendations, GT):
    RelevantGT = []
    for ID, _ in Recommendations:
        if ID in GT:
            RelevantGT.append(1)
        else:
            RelevantGT.append(0)
    return RelevantGT
