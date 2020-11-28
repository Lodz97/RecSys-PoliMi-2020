from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.Recommender_utils import check_matrix
import numpy as np

class ItemKNNScoresHybridRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    NB: Rec_1 is itemKNNCF, Rec_2 is userKNNCF

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2):
        super(ItemKNNScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2

    def fit(self, alpha=0.5):
        self.alpha = alpha


    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        user_profile_array = self.URM_train[user_id_array]
        user_weights_array = self.Recommender_2.W_sparse[user_id_array]

        if items_to_compute is not None:
            item_scores1 = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32) * np.inf
            item_scores_all1 = user_profile_array.dot(self.Recommender_1.W_sparse).toarray()
            item_scores1[:, items_to_compute] = item_scores_all1[:, items_to_compute]
            item_scores2 = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32) * np.inf
            item_scores_all2 = user_weights_array.dot(self.Recommender_2.URM_train).toarray()
            item_scores2[:, items_to_compute] = item_scores_all2[:, items_to_compute]
        else:
            #print(self.Recommender_1.W_sparse)
            #print(self.Recommender_2.W_sparse)
            item_scores1 = self.Recommender_1._compute_item_score(user_id_array, items_to_compute)
            item_scores2 = self.Recommender_2._compute_item_score(user_id_array, items_to_compute)

        mean1 = np.mean(item_scores1)
        mean2 = np.mean(item_scores2)
        std1 = np.std(item_scores1)
        std2 = np.std(item_scores2)
        item_scores1 = (item_scores1 - mean1) / std1
        item_scores2 = (item_scores2 - mean2) / std2
        # print(item_scores1)
        # print(item_scores2)

        item_scores = item_scores1 * self.alpha + item_scores2 * (1 - self.alpha)

        return item_scores
