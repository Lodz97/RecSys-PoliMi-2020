from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.Recommender_utils import check_matrix
import numpy as np

class ScoresHybridSpecializedFusion(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    NB: Rec_1 is itemKNNCF, Rec_2 is userKNNCF

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender"

    def __init__(self, URM_train, Recommender_cold, Recommender_warm):
        super(ScoresHybridSpecializedFusion, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_cold = Recommender_cold
        self.Recommender_warm = Recommender_warm

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):

        profile_length = np.ediff1d(self.URM_train.indptr)
        res1 = self.Recommender_cold.recommend(user_id_array, cutoff=cutoff, remove_seen_flag=remove_seen_flag,
              items_to_compute=items_to_compute, remove_top_pop_flag=remove_top_pop_flag,
              remove_custom_items_flag=remove_custom_items_flag, return_scores=return_scores)
        res2 = self.Recommender_warm.recommend(user_id_array, cutoff=cutoff, remove_seen_flag=remove_seen_flag,
                                            items_to_compute=items_to_compute, remove_top_pop_flag=remove_top_pop_flag,
                                            remove_custom_items_flag=remove_custom_items_flag,
                                            return_scores=return_scores)
        res = res2
        if return_scores:
            for i in range(0, len(user_id_array)):
                if profile_length[user_id_array[i]] < 6:
                    res[0][i] = res1[0][i]
        else:
            for i in range(0, len(user_id_array)):
                if profile_length[user_id_array[i]] < 6:
                    res[i] = res1[i]

        return res
