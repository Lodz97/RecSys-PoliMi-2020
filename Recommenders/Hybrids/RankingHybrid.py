from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.Recommender_utils import check_matrix
from Utils import RankMerger


class RankingHybrid(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    NB: Rec_1 is itemKNNCF, Rec_2 is userKNNCF

    """

    RECOMMENDER_NAME = "RankingHybrid"

    def __init__(self, URM_train, Recommender_1, Recommender_2):
        super(RankingHybrid, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):

        ranks1 = self.Recommender_1.recommend(user_id_array, cutoff=cutoff, remove_seen_flag=remove_seen_flag,
                                              items_to_compute=items_to_compute,
                                              remove_top_pop_flag=remove_top_pop_flag,
                                              remove_custom_items_flag=remove_custom_items_flag,
                                              return_scores=return_scores)
        ranks2 = self.Recommender_2.recommend(user_id_array, cutoff=cutoff, remove_seen_flag=remove_seen_flag,
                                              items_to_compute=items_to_compute,
                                              remove_top_pop_flag=remove_top_pop_flag,
                                              remove_custom_items_flag=remove_custom_items_flag,
                                              return_scores=return_scores)
        merged_ranks = RankMerger.merge(ranks1, ranks2)

        return merged_ranks
