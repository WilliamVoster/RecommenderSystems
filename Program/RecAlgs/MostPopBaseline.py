def RecommendMostPopular(PossibleArticles, PastBehaviors, CurrentTime, TimePenaltyPerHour, TimePenaltyStart):
    PossibleArticlesSortedWithScore = []
    for Article in PossibleArticles:
        Score = GetPopularityScore(Article, PastBehaviors, CurrentTime, TimePenaltyPerHour, TimePenaltyStart)
        ScoreTuple = (Article["NewsID"], Score)
        PossibleArticlesSortedWithScore.append(ScoreTuple)

    PossibleArticlesSortedWithScore.sort(key=lambda x: x[1], reverse=True)
    return PossibleArticlesSortedWithScore[:10]


def GetPopularityScore(Article, PastBehaviors, CurrentTime, TimePenaltyPerHour, TimePenaltyStart):
    Score = 0
    ArticleIDClicked = Article["NewsID"] + "-1"
    for Instance in PastBehaviors:
        if ArticleIDClicked in Instance["ClickData"]:
            Score += 1
    PubTime = Article["ReleaseDate"]
    DeltaTime = CurrentTime - PubTime
    Score = ApplyTimeMultiplierToScore(DeltaTime, Score, TimePenaltyPerHour, TimePenaltyStart)
    return Score


def ApplyTimeMultiplierToScore(DeltaTime, Score, TimePenaltyPerHour, TimePenaltyStart):
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
