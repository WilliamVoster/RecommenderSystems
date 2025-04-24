def RecommendMostPopular(PossibleArticles, PastBehaviors, CurrentTime, TimePenaltyPerHour, TimePenaltyStart):
    ClickedIDs = set()
    for instance in PastBehaviors.itertuples(index=False):
        ClickedIDs.update(instance.ClickData)

    PossibleArticlesSortedWithScore = []

    for article in PossibleArticles.itertuples(index=False):
        NewsID = article.NewsID
        ReleaseDate = article.ReleaseDate

        ArticleIDClicked = NewsID + "-1"
        Score = sum(1 for click in ClickedIDs if click == ArticleIDClicked)

        # Time-based score adjustment
        DeltaTime = CurrentTime - ReleaseDate
        Score = ApplyTimeMultiplierToScore(DeltaTime, Score, TimePenaltyPerHour, TimePenaltyStart)

        PossibleArticlesSortedWithScore.append((NewsID, Score))

    PossibleArticlesSortedWithScore.sort(key=lambda x: x[1], reverse=True)
    return PossibleArticlesSortedWithScore[:10]


def ApplyTimeMultiplierToScore(DeltaTime, Score, TimePenaltyPerHour, TimePenaltyStart):
    if Score == 0:
        return Score
    # Get the total hours since publication
    HoursSincePublication = DeltaTime.total_seconds() / 3600  # Convert seconds to hours

    # Compute the base hourly score
    AvgScorePerHour = Score / HoursSincePublication if HoursSincePublication > 0 else Score

    # Apply penalty only if time exceeds the penalty start
    if HoursSincePublication > TimePenaltyStart:
        PenaltyHours = HoursSincePublication - TimePenaltyStart
        PenaltyFactor = 1 - (PenaltyHours * TimePenaltyPerHour)  # Reduce score based on penalty
        PenaltyFactor = max(PenaltyFactor, 0)  # Ensure it doesn't go below 0

        # Apply the penalty to the average score
        MultipliedScore = AvgScorePerHour * PenaltyFactor
    else:
        MultipliedScore = AvgScorePerHour  # No penalty applied

    return MultipliedScore
