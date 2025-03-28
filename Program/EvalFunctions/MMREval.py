import numpy as np


def MMREval(Recommendations, GT):
    ActualScores = np.array(GetRelevantGT(Recommendations, GT), dtype=float)
    RecommendedScores = np.array([Score for _, Score in Recommendations], dtype=float)

    RelativeErrors = np.abs(ActualScores - RecommendedScores) / ActualScores
    return np.mean(RelativeErrors)


def GetRelevantGT(Recommendations, GT):
    RelevantGT = []
    for ID, _ in Recommendations:
        if ID in GT:
            RelevantGT.append(1)
        else:
            # prevent div by zero
            RelevantGT.append(0.000001)
    return RelevantGT