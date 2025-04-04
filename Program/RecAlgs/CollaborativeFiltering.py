
import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp

def printScore(df:pd.DataFrame) -> None:

    for i, row in df.iterrows():
        print(row.tolist())

class CollaborativeFiltering:
    def __init__(
            self, 
            interactions: pd.DataFrame, 
            epochs: int = 10, 
            num_latent_factors: int = 3,
            lambda_regularization: int = 0.1):
        
        self.raw_interactions = interactions
        self.epochs = epochs
        self.num_latent_factors = num_latent_factors
        self.lambda_regularization = lambda_regularization

    def initialize(self):

        print("Initializing collaborative filtering...")
        interaction_matrix = self.interactions.copy(deep=True)


        print("apply start")
        interaction_matrix.loc[:, "History"] = interaction_matrix["History"].apply(lambda x: [] if pd.isna(x) else x.split())


        print("explode start")
        exploded = interaction_matrix.explode("History")


        print("sparse matrix start")
        user_mapping = {u: i for i, u in enumerate(exploded["UserID"].unique())}
        item_mapping = {u: i for i, u in enumerate(exploded["History"].unique())}

        user_indicies = exploded["UserID"].map(user_mapping).values
        item_indicies = exploded["History"].map(item_mapping).values

        interaction_matrix = sp.csr_matrix(
            (np.ones(len(user_indicies)), (user_indicies, item_indicies)),
            shape=(len(user_mapping), len(item_mapping))
        )

        self.df_interaction_matrix = pd.DataFrame.sparse.from_spmatrix(
            interaction_matrix, 
            index=user_mapping.keys(), 
            columns=item_mapping.keys()
        )
    
        self.calc_ALS_sparse(self.df_interaction_matrix, self.epochs)

        self.predicted_scores = torch.matmul(self.user_embeddings, self.item_embeddings.T)



        # return interaction_matrix, user_mapping, item_mapping


    def calc_ALS_sparse(self, interactions:pd.DataFrame, epochs: int = 10):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu" # override, since cpu has access to more memory than gpu

        values = interactions.values
        rows, cols = values.nonzero()
        data = values[rows, cols]

        print(f"Starting ALS using {device}")

        interaction_sparse = torch.sparse_coo_tensor(
            indices=torch.tensor([rows, cols], device=device),
            values=torch.tensor(data, dtype=torch.float32, device=device),
            size=(values.shape[0], values.shape[1])
        ).coalesce()
        print("interaction_sparse.shape: ", interaction_sparse.shape)


        num_users, num_items = interaction_sparse.shape
        user_embeddings = torch.rand(num_users, self.num_latent_factors, requires_grad=False, device=device)
        item_embeddings = torch.rand(num_items, self.num_latent_factors, requires_grad=False, device=device)
        print("user_embeddings.shape, item_embeddings.shape: ", user_embeddings.shape, item_embeddings.shape)

        eye = torch.eye(self.num_latent_factors, device=device)

        print("training start")
        for epoch in range(epochs):

            # Fix items, Solve for user factors
            item_approximation_matrix = torch.matmul(item_embeddings.T, item_embeddings)
            item_approximation_matrix += self.lambda_regularization * eye
            item_change = torch.sparse.mm(interaction_sparse, item_embeddings)
            user_embeddings = torch.linalg.solve(item_approximation_matrix, item_change.T).T


            # Fix users, Solve for item factors
            user_approximation_matrix = torch.matmul(user_embeddings.T, user_embeddings)
            user_approximation_matrix += self.lambda_regularization * eye
            user_change = torch.sparse.mm(interaction_sparse.T, user_embeddings)
            item_embeddings = torch.linalg.solve(user_approximation_matrix, user_change.T).T


            predicted = torch.matmul(user_embeddings, item_embeddings.T)
            observed = interaction_sparse._indices()
            actual = interaction_sparse._values()
            predicted_values = predicted[observed[0], observed[1]]
            loss = torch.nn.functional.mse_loss(predicted_values, actual)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        # extracts user and item factors
        # user_factors = user_embeddings.detach().numpy()
        # item_factors = item_embeddings.detach().numpy()

        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

        # return user_embeddings, item_embeddings


    def getRecommended(self, possible_articles:pd.DataFrame, user_id:int, k: int = 10) -> float: # return ranked list of 10 articles

        scores = self.predicted_scores[user_id]
        top_items = torch.topk(scores, k=k).indices.numpy()
        return top_items


def get_interactions_by_user(user_id: int, interaction_matrix: sp.csr_matrix) -> np.ndarray:

    user_interactions = interaction_matrix[user_id].toarray().flatten()
    interacted_items = user_interactions.nonzero()[0]
    return interacted_items

def get_interactions_by_item(item_id: int, interaction_matrix: sp.csr_matrix) -> np.ndarray:

    item_interactions = interaction_matrix[:, item_id].toarray().flatten()
    users_who_interacted = item_interactions.nonzero()[0]
    return users_who_interacted




def calc_ALS(interactions:pd.DataFrame, epochs: int = 10):

    interaction_tensor = torch.tensor(interactions.values, dtype=torch.float32)

    print("interaction_tensor.shape: ", interaction_tensor.shape)
    num_users, num_items = interaction_tensor.shape
    num_latent_factors = 3
    lambda_regularization = 0.1

    user_embeddings = torch.rand(num_users, num_latent_factors, requires_grad=False)
    item_embeddings = torch.rand(num_items, num_latent_factors, requires_grad=False)
    print("user_embeddings.shape, item_embeddings.shape: ", user_embeddings.shape, item_embeddings.shape)

    print("training start")
    for epoch in range(epochs):

        # Solve for user factors
        item_approximation_matrix = torch.matmul(item_embeddings.T, item_embeddings)
        item_approximation_matrix += lambda_regularization * torch.eye(num_latent_factors)
        item_change = torch.matmul(item_embeddings.T, interaction_tensor.T)
        user_embeddings = torch.linalg.solve(item_approximation_matrix, item_change).T


        # Solve for item factors
        user_approximation_matrix = torch.matmul(user_embeddings.T, user_embeddings)
        user_approximation_matrix += lambda_regularization * torch.eye(num_latent_factors)
        user_change = torch.matmul(user_embeddings.T, interaction_tensor)
        item_embeddings = torch.linalg.solve(user_approximation_matrix, user_change).T

        predicted = torch.matmul(user_embeddings, item_embeddings.T)
        mask = interaction_tensor > 0
        loss = torch.nn.functional.mse_loss(predicted[mask], interaction_tensor[mask])

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # extracts user and item factors
    # user_factors = user_embeddings.detach().numpy()
    # item_factors = item_embeddings.detach().numpy()

    return user_embeddings, item_embeddings



