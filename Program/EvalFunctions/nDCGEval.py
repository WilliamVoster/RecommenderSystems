import numpy as np


def DCG(Relevance):
    return np.sum(Relevance / np.log2(np.arange(2, len(Relevance) + 2)))  # log base 2, starts at rank 1


def nDCG(Recommendations, GT):
    """
    Compute Normalized Discounted Cumulative Gain (nDCG).

    Parameters:
    - actual_clicks (list): Ground truth, 1 if article was clicked, 0 otherwise.
    - predicted_ranking (list): Ordered list of recommended article relevance scores.

    Returns:
    - nDCG score (float between 0 and 1).
    """
    GT = np.array(GetRelevantGT(Recommendations, GT))
    Recommendations = np.array(Recommendations)

    # Compute DCG for the given ranking
    DCGScore = DCG(Recommendations)

    # Compute IDCG (ideal ranking where all clicked articles are at the top)
    idealRanking = np.sort(GT)[::-1]  # Sort actual clicks in descending order
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
