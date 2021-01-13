#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
from Base.BaseRecommender import BaseRecommender
from Base.Recommender_utils import check_matrix
#import implicit

import scipy.sparse as sps

class PredefinedListRecommender(BaseRecommender):
    """PredefinedListRecommender recommender"""

    RECOMMENDER_NAME = "PredefinedListRecommenderRecommender"

    def __init__(self, URM_train):
        super(PredefinedListRecommender, self).__init__(URM_train)

        # convert to csc matrix for faster column-wise sum

        self.URM_train = URM_train
        self.USER_factors = None
        self.ITEM_factors = None



    def fit(self, factors=50, epochs=8):
        self.model = implicit.als.AlternatingLeastSquares(factors=factors, iterations=epochs, regularization=1e-3,
                                                          calculate_training_loss=True)

        # train the model on a sparse matrix of item/user/confidence weights
        self.model.fit(self.URM_train.T)
        self.USER_factors = self.model.user_factors
        self.ITEM_factors = self.model.item_factors


    '''def recommend(self, user_id, cutoff = None, remove_seen_flag=True, remove_top_pop_flag = False, remove_custom_items_flag = False,
                  return_scores=False):

        if cutoff is None:
            cutoff= self.URM_train.shape[1] - 1

        start_pos = self.URM_recommendations.indptr[user_id]
        end_pos = self.URM_recommendations.indptr[user_id+1]

        recommendation_list = self.URM_recommendations.data[start_pos:end_pos]

        if len(recommendation_list[:cutoff]) == 0:
            pass

        return recommendation_list[:cutoff]'''

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        '''# recommend items for a user
        rec_list = []
        for el in user_id_array:
            recommendations = self.model.recommend(el, self.URM_train, N=25975)
            rec_list.append([x[1] for x in recommendations])
        arr = np.array(rec_list)
        return arr'''

        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1], \
            "{}: User and Item factors have inconsistent shape".format(self.RECOMMENDER_NAME)

        assert self.USER_factors.shape[0] > np.max(user_id_array), \
            "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
                self.RECOMMENDER_NAME, self.USER_factors.shape[0], np.max(user_id_array))

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.ITEM_factors.shape[0]), dtype=np.float32) * np.inf
            item_scores[:, items_to_compute] = np.dot(self.USER_factors[user_id_array],
                                                      self.ITEM_factors[items_to_compute, :].T)

        else:
            item_scores = np.dot(self.USER_factors[user_id_array], self.ITEM_factors.T)

        return item_scores

    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)


        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]
            #print(scores_batch[user_index])

            #if remove_seen_flag:
            #    scores_batch[user_index] = self._remove_seen_on_scores(user_id, scores_batch[user_index)

            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
            # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
            # ranking = relevant_items_partition[relevant_items_partition_sorting]
            #
            # ranking_list.append(ranking)


        if remove_top_pop_flag:
            scores_batch = self._remove_TopPop_on_scores(scores_batch)

        if remove_custom_items_flag:
            scores_batch = self._remove_custom_items_on_scores(scores_batch)

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:,0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = [None] * ranking.shape[0]

        # Remove from the recommendation list any item that has a -inf score
        # Since -inf is a flag to indicate an item to remove
        for user_index in range(len(user_id_array)):
            user_recommendation_list = ranking[user_index]
            user_item_scores = scores_batch[user_index, user_recommendation_list]

            not_inf_scores_mask = np.logical_not(np.isinf(user_item_scores))

            user_recommendation_list = user_recommendation_list[not_inf_scores_mask]
            ranking_list[user_index] = user_recommendation_list.tolist()



        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]


        if return_scores:
            return ranking_list, scores_batch

        else:
            return ranking_list



    def __str__(self):
        return "PredefinedListRecommender"



