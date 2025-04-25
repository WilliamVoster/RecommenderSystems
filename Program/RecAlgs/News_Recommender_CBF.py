import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


class NewsRecommenderCBF:
    def __init__(self, mind_data_path_items, mind_data_path_user):
        self._load_data(mind_data_path_items, mind_data_path_user)

        self.corpus = self._create_corpus()
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_df=0.8, min_df=2, ngram_range=(1, 2)
        )

        self.tfidf_matrix = self._create_tfidf_matrix()
        self.cat_embed = self._create_genre_matrix()
        self.combined_matrix = self._combine_emcodings()

    def _load_data(self, path_items, path_user_behavior):

        coloumns_items = [
            "news_id",
            "category",
            "sub_category",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ]

        coloumns_user = ["uID", "t", "ClickHist", "ImpLog"]

        try:
            data_i = pd.read_csv(
                path_items, sep="\t", header=None, names=coloumns_items
            )
            print(
                f"00 ----------> ITEM data loaded successfully: {len(data_i)} records!"
            )

            data_i = data_i.fillna("")

            data_u = pd.read_csv(
                path_user_behavior, sep="\t", header=None, names=coloumns_user
            )

            data_u = data_u.fillna("")
            data_u["t"] = pd.to_datetime(data_u["t"])
            print(
                f"01 ----------> USER data loaded successfully: {len(data_u)} records!"
            )
        except Exception as e:
            print(f"Error loading data: {e}")
            data_i = None
            data_u = None

        self.items = data_i
        self.user_behavior = data_u

    # ------------>ITEM side-----------

    def _create_corpus(self):
        if self.items is None:
            return []

        corpus = (self.items["title"] + " " + self.items["abstract"]).tolist()
        print(f"02 ----------> Corpus created: {len(corpus)} documents!")
        return corpus

    def _create_tfidf_matrix(self):
        if self.items is None:
            return None

        tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

        print(
            f"03 ----------> TF-IDF matrix created: {tfidf_matrix.shape[0]} documents, {tfidf_matrix.shape[1]} terms!"
        )

        return tfidf_matrix

    def _create_genre_matrix(self):
        if self.items is None:
            return None

        unique_categories = sorted(self.items["category"].dropna().unique())
        category_to_index = {
            category: i for i, category in enumerate(unique_categories)
        }

        unique_subcategories = sorted(self.items["sub_category"].dropna().unique())
        subcategory_to_index = {
            subcategory: i for i, subcategory in enumerate(unique_subcategories)
        }

        rows_category = []
        cols_category = []
        data_category = []
        rows_subcategory = []
        cols_subcategory = []
        data_subcategory = []

        for i, row in self.items.iterrows():
            cat = row["category"]
            if pd.notna(cat):
                rows_category.append(i)
                cols_category.append(category_to_index[cat])
                data_category.append(1.0)
            subcat = row["sub_category"]
            if pd.notna(subcat):
                rows_subcategory.append(i)
                cols_subcategory.append(subcategory_to_index[subcat])
                data_subcategory.append(1.0)

        category_matrix = csr_matrix(
            (data_category, (rows_category, cols_category)),
            shape=(len(self.items), len(unique_categories)),
        )
        subcategory_matrix = csr_matrix(
            (data_subcategory, (rows_subcategory, cols_subcategory)),
            shape=(len(self.items), len(unique_subcategories)),
        )

        self.cat_embed = hstack([category_matrix, subcategory_matrix])

        print(
            f"04 ----------> Category matrix created: {self.cat_embed.shape[0]} documents, {self.cat_embed.shape[1]} categories!"
        )

        return self.cat_embed

    def _combine_emcodings(self):

        comibned_matrix = hstack([self.tfidf_matrix, self.cat_embed])
        print(f"05 ----------> Combined matrix created, shape: {comibned_matrix.shape}")

        return comibned_matrix

    # ------------>USER side-----------

    def _build_click_profile(self, user):
        user_entries = self.user_behavior.loc[
            self.user_behavior.uID == user
        ].sort_values(by="t", ascending=False)

        if user_entries.empty:
            return None

        click_list = user_entries.iloc[0]["ClickHist"].split(" ")
        idx_map = pd.Series(self.items.index, index=self.items["news_id"])
        idxs = [idx_map[a] for a in click_list if a in idx_map]
        if not idxs:
            # print(user)
            return None

        M = self.combined_matrix[idxs]

        return csr_matrix(M.mean(axis=0))

    def _build_imp_profile(self, user, half_life=7):

        user_entries = self.user_behavior.loc[
            self.user_behavior.uID == user
        ].sort_values(by="t", ascending=False)

        if user_entries.empty:
            return None

        newest = user_entries.iloc[0]["t"]
        day_delta = (newest - user_entries["t"]).dt.total_seconds() / (24 * 3600)
        decay = 0.5 ** (day_delta / half_life)

        idx_map = pd.Series(
            self.items.index, index=self.items["news_id"]
        ).drop_duplicates()

        rows, weights = [], []

        for session_wt, log_str in zip(decay, user_entries["ImpLog"]):
            for tok in log_str.split(" "):
                aid, flag = tok.split("-", 1)
                if int(flag) != 1:
                    continue
                if aid not in idx_map:
                    continue

                rows.append(idx_map[aid])
                weights.append(session_wt)

        if not rows:
            return None

        X = self.combined_matrix[rows]
        w = np.array(weights)

        Xw = X.multiply(w[:, None])
        summed = Xw.sum(axis=0)
        user_vector = summed / w.sum()

        return csr_matrix(user_vector)

    def _build_user_profile(self, user, alpha=0.5):
        click_profile = self._build_click_profile(user)
        imp_profile = self._build_imp_profile(user)

        if click_profile is None and imp_profile is None:
            return None

        if click_profile is None:
            return imp_profile

        if imp_profile is None:
            return click_profile

        user_vector = click_profile + imp_profile.multiply(alpha)
        return csr_matrix(user_vector)

    # ------------>RECOMMENDATION-----------

    def recommend(self, user_id, top_n=10):
        user_profile = self._build_user_profile(user_id, alpha=0.8)
        if user_profile is None:
            return []

        A = user_profile
        B = self.combined_matrix

        sim_scores = cosine_similarity(A, B)

        if top_n <= 0:
            top_k_indices = np.argsort(sim_scores[0])[::-1]  # all sorted
        else:
            top_k_indices = np.argsort(sim_scores[0])[::-1][:top_n]

        ids = self.items.iloc[top_k_indices]["news_id"].tolist()
        scores = sim_scores[0, top_k_indices].tolist()

        return list(zip(ids, scores))  # returns list of (news_id, score)

    # Output Functions

    def get_item_frame(self):
        return self.items

    def get_user_frame(self):
        return self.user_behavior
