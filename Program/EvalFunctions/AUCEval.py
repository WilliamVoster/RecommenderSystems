from sklearn.metrics import roc_auc_score


def AUCEval(Recommendations, GT):
    AUCScore = 0
    RelevantGTScores = GetRelevantGT(Recommendations, GT)
    RecommendedScores = [Score for _, Score in Recommendations]
    AUCScore = roc_auc_score(RelevantGTScores, RecommendedScores)
    return AUCScore


def GetRelevantGT(Recommendations, GT):
    RelevantGT = []
    for ID, _ in Recommendations:
        if ID in GT:
            RelevantGT.append(1)
        else:
            RelevantGT.append(0)
    return RelevantGT
