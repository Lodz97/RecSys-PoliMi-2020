from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.Recommender_utils import check_matrix
import ScoresHybridP3alphaKNNCBF
import ScoresHybridRP3betaKNNCBF
from Base.DataIO import DataIO
from functools import partial
from Utils.PoolWithSubprocess import PoolWithSubprocess
import multiprocessing
import numpy as np

class ScoresHybridSpecializedAdaptive(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    NB: Rec_1 is itemKNNCF, Rec_2 is userKNNCF

    """


    RECOMMENDER_NAME = "ScoresHybridSpecializedAdaptive"

    def __init__(self, URM_train, ICM_train):
        super(ScoresHybridSpecializedAdaptive, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_warm = ScoresHybridRP3betaKNNCBF.ScoresHybridRP3betaKNNCBF(URM_train, ICM_train)
        self.Recommender_cold = ScoresHybridRP3betaKNNCBF.ScoresHybridRP3betaKNNCBF(URM_train, ICM_train)
        self.threshold = 5.9

    def fitRec(self, rec_args_name):
        rec = rec_args_name[0]
        args = rec_args_name[1]
        name = rec_args_name[2]
        rec.fit(**args)
        return [rec, name]

    def fit(self, topK_P_C=991, alpha_P_C=0.4705816992313091, beta_P_C=0, normalize_similarity_P_C=False, alpha_C=0.5,
            topK_C=700, shrink_C=200, similarity_C='jaccard', normalize_C=True, feature_weighting_C="TF-IDF",
            norm_scores_C=True,
            topK_P=991, alpha_P=0.4705816992313091, beta_P=0, normalize_similarity_P=False, alpha=0.5,
            topK=700, shrink=200, similarity='jaccard', normalize=True, feature_weighting="TF-IDF", norm_scores=True,
            threshold=5.9):
        '''self.Recommender_cold.fit(topK_P=topK_P_C, alpha_P=alpha_P_C,
                                  normalize_similarity_P=normalize_similarity_P_C, alpha=alpha_C, topK=topK_C,
                                  shrink=shrink_C, similarity=similarity_C, normalize=normalize_C,
                                  feature_weighting=feature_weighting_C, norm_scores=norm_scores_C)
        self.Recommender_warm.fit(topK_P=topK_P, alpha_P=alpha_P,
                                  normalize_similarity_P=normalize_similarity_P, alpha=alpha, topK=topK,
                                  shrink=shrink, similarity=similarity, normalize=normalize,
                                  feature_weighting=feature_weighting, norm_scores=norm_scores)'''
        cold_args = {"topK_P": topK_P_C, "alpha_P": alpha_P_C, "beta_P": beta_P_C,
                     "normalize_similarity_P": normalize_similarity_P_C, "alpha": alpha_C, "topK": topK_C,
                     "shrink": shrink_C, "similarity": similarity_C, "normalize": normalize_C,
                     "feature_weighting": feature_weighting_C, "norm_scores": norm_scores_C}
        warm_args = {"topK_P": topK_P, "alpha_P": alpha_P, "beta_P": beta_P,
                     "normalize_similarity_P": normalize_similarity_P, "alpha": alpha, "topK": topK,
                     "shrink": shrink, "similarity": similarity, "normalize": normalize,
                     "feature_weighting": feature_weighting, "norm_scores": norm_scores}
        tot_args = zip([self.Recommender_cold, self.Recommender_warm], [cold_args, warm_args], ["Cold", "Warm"])
        pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count() - 1), maxtasksperchild=1)
        resultList = pool.map(self.fitRec, tot_args)
        pool.close()
        pool.join()

        for el in resultList:
            if el[1] == "Cold":
                self.Recommender_cold = el[0]
            elif el[1] == "Warm":
                self.Recommender_warm = el[0]

        self.threshold = threshold


    def _compute_item_score(self, user_id_array, items_to_compute=None):

        profile_length = np.ediff1d(self.URM_train.indptr)
        item_scores_cold = self.Recommender_cold._compute_item_score(user_id_array, items_to_compute)
        item_scores_warm = self.Recommender_warm._compute_item_score(user_id_array, items_to_compute)
        item_scores = item_scores_warm
        for i in range(0, len(user_id_array)):
            if profile_length[user_id_array[i]] < self.threshold:
                item_scores[i] = item_scores_cold[i]

        return item_scores

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
                if profile_length[user_id_array[i]] < self.threshold:
                    res[0][i] = res1[0][i]
        else:
            for i in range(0, len(user_id_array)):
                if profile_length[user_id_array[i]] < self.threshold:
                    res[i] = res1[i]

        return res

    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"W_sparse_Cold_itemKNNCBF": self.Recommender_cold.itemKNNCBF.W_sparse,
                             "norm_scores_Cold": self.Recommender_cold.norm_scores,
                             "W_sparse_Cold_P3Alpha": self.Recommender_cold.RP3beta.W_sparse,
                             "alpha_Cold": self.Recommender_cold.alpha,
                             "W_sparse_Warm_itemKNNCBF": self.Recommender_warm.itemKNNCBF.W_sparse,
                             "norm_scores_Warm": self.Recommender_warm.norm_scores,
                             "W_sparse_Warm_P3Alpha": self.Recommender_warm.RP3beta.W_sparse,
                             "alpha_Warm": self.Recommender_warm.alpha,
                             "threshold": self.threshold}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")


    def load_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
            if attrib_name == "W_sparse_Cold_P3Alpha":
                self.Recommender_cold.RP3beta.W_sparse = data_dict[attrib_name]
            elif attrib_name == "W_sparse_Cold_itemKNNCBF":
                self.Recommender_cold.itemKNNCBF.W_sparse = data_dict[attrib_name]
            elif attrib_name == "alpha_Cold":
                self.Recommender_cold.alpha = data_dict[attrib_name]
            elif attrib_name == "norm_scores_Cold":
                self.Recommender_cold.norm_scores = data_dict[attrib_name]
            elif attrib_name == "W_sparse_Warm_P3Alpha":
                self.Recommender_warm.RP3beta.W_sparse = data_dict[attrib_name]
            elif attrib_name == "W_sparse_Warm_itemKNNCBF":
                self.Recommender_warm.itemKNNCBF.W_sparse = data_dict[attrib_name]
            elif attrib_name == "alpha_Warm":
                self.Recommender_warm.alpha = data_dict[attrib_name]
            elif attrib_name == "norm_scores_Warm":
                self.Recommender_warm.norm_scores = data_dict[attrib_name]
            elif attrib_name == "threshold":
                self.threshold = data_dict[attrib_name]

        self._print("Loading complete")
