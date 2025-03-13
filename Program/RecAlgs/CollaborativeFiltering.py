
import pandas as pd


def printScore(df:pd.DataFrame) -> None:

    for i, row in df.iterrows():
        print(row.tolist())



def initialize(interactions:pd.DataFrame) -> pd.DataFrame:

    print("Initializing collaborative filtering...")


    interaction_matrix = interactions.copy(deep=True)


    print("apply start")
    interaction_matrix.loc[:, "History"] = interaction_matrix["History"].apply(lambda x: x if pd.isna(x) else x.split())


    print("explode start")
    exploded = interaction_matrix.explode("History")


    print("pivot start")
    pivoted = exploded.pivot_table(
        index='UserID',
        columns='History',
        # values='Impression ID',
        aggfunc='size', #'count'
    ).fillna(0).astype(int)


    # printScore(pivoted)
    return pivoted


def getScore() -> float:
    pass





if __name__ == "__main__":

    initialize(None)



