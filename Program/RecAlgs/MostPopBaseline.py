def RecommendMostPopular(PossibleArticles, CurrentTime, GlobalPopularitySorted, TimePenaltyPerHour, TimePenaltyStart):
    AvailableIDs = set(PossibleArticles["NewsID"])
    ReleaseDates = dict(zip(PossibleArticles["NewsID"], PossibleArticles["ReleaseDate"]))

    Recommendations = []
    tried = 0

    for NewsID, BaseScore in GlobalPopularitySorted:
        if NewsID in AvailableIDs:
            ReleaseDate = ReleaseDates[NewsID]
            DeltaTime = CurrentTime - ReleaseDate
            Score = ApplyTimeMultiplierToScore(DeltaTime, BaseScore, TimePenaltyPerHour, TimePenaltyStart)
            Recommendations.append((NewsID, Score))

            if len(Recommendations) >= 10 and tried >= 10:
                break

        tried += 1

    return sorted(Recommendations, key=lambda x: x[1], reverse=True)[:10]


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
