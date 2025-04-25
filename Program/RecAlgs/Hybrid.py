def HybridRecommendations(PopRecommendations, CFRecommendations, CBRRecommendations, weights, amount):
    if not CFRecommendations or not CBRRecommendations:
        return sorted(PopRecommendations, key=lambda x: x[1], reverse=True)[:amount]

    # Convert lists to dictionaries for O(1) lookups
    pop_dict = dict(PopRecommendations)
    cf_dict = dict(CFRecommendations)
    cbr_dict = dict(CBRRecommendations)

    # Use set union directly in a list comprehension for speed
    all_articles = pop_dict.keys() | cf_dict.keys() | cbr_dict.keys()

    # Compute weighted scores directly in the list comprehension
    hybrid_scores = [
        (article,
         pop_dict.get(article, 0) * weights[0] +
         cf_dict.get(article, 0) * weights[1] +
         cbr_dict.get(article, 0) * weights[2])
        for article in all_articles
    ]

    return sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:amount]


