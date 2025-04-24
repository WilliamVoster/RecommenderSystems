import numpy as np


def MMREval(Recommendations, GT):
    ActualScores = np.array(GetRelevantGT(Recommendations, GT), dtype=float)

    if np.sum(ActualScores) == 0:
        return 0  # or skip this user in your eval loop

    RecommendedScores = np.array([Score for _, Score in Recommendations], dtype=float)
    RelativeErrors = np.abs(ActualScores - RecommendedScores) / ActualScores
    return np.mean(RelativeErrors)


def GetRelevantGT(Recommendations, GT):
    RelevantGT = []
    for ID, _ in Recommendations:
        if ID in GT:
            RelevantGT.append(1)
        else:
            RelevantGT.append(0.000001)
    return RelevantGT
