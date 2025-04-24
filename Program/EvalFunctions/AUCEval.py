from sklearn.metrics import roc_auc_score


def AUCEval(Recommendations, GT):
    RelevantGTScores = GetRelevantGT(Recommendations, GT)
    RecommendedScores = [Score for _, Score in Recommendations]

    # Check if both 0 and 1 exist in RelevantGTScores
    unique_classes = set(RelevantGTScores)
    if len(unique_classes) < 2:
        return 0  # or return np.nan, or 0, depending on how you want to aggregate

    return roc_auc_score(RelevantGTScores, RecommendedScores)


def GetRelevantGT(Recommendations, GT):
    RelevantGT = []
    for ID, _ in Recommendations:
        if ID in GT:
            RelevantGT.append(1)
        else:
            RelevantGT.append(0)
    return RelevantGT
