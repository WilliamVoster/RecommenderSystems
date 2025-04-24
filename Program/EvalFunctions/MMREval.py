import numpy as np


def MMREval(Recommendations, GT):
    # Binary vector: 1 if in GT, 0 otherwise
    ActualScores = np.array([1.0 if ID in GT else 0.0 for ID, _ in Recommendations], dtype=float)
    PredictedScores = np.array([Score for _, Score in Recommendations], dtype=float)

    # Avoid division by zero: set denominator to 1 if actual is 0
    denominator = np.where(ActualScores == 0, 1.0, ActualScores)
    RelativeErrors = np.abs(ActualScores - PredictedScores) / denominator

    return np.mean(RelativeErrors)
