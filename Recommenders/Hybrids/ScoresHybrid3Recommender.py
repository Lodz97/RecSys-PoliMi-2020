from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.Recommender_utils import check_matrix
import numpy as np


class ScoresHybrid3Recommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    NB: Rec_1 is itemKNNCF, Rec_2 is userKNNCF

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2, Recommender_3):
        super(ScoresHybrid3Recommender, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        self.Recommender_3 = Recommender_3

    def fit(self, alpha=1, beta=1, gamma=1, norm_sc=True):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.norm_sc = norm_sc


    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        item_scores1 = self.Recommender_1._compute_item_score(user_id_array, items_to_compute)
        item_scores2 = self.Recommender_2._compute_item_score(user_id_array, items_to_compute)
        item_scores3 = self.Recommender_3._compute_item_score(user_id_array, items_to_compute)

        if self.norm_sc:
            mean1 = np.mean(item_scores1)
            mean2 = np.mean(item_scores2)
            mean3 = np.mean(item_scores3)
            std1 = np.std(item_scores1)
            std2 = np.std(item_scores2)
            std3 = np.std(item_scores3)
            item_scores1 = (item_scores1 - mean1) / std1
            item_scores2 = (item_scores2 - mean2) / std2
            item_scores3 = (item_scores3 - mean3) / std3
            # print(item_scores1)
            # print(item_scores2)

        item_scores = item_scores1 * self.alpha + item_scores2 * self.beta + item_scores3 * self.gamma

        return item_scores


