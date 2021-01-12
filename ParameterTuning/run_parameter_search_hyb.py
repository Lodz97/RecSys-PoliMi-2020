#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""



######################################################################
##########                                                  ##########
##########                      HYBRID                      ##########
##########                                                  ##########
######################################################################


from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
#Hybrid
from ScoresHybridP3alphaKNNCBF import ScoresHybridP3alphaKNNCBF
from ScoresHybridRP3betaKNNCBF import ScoresHybridRP3betaKNNCBF
from ScoresHybridP3alphaPureSVD import ScoresHybridP3alphaPureSVD
from ScoresHybridSpecialized import ScoresHybridSpecialized
from ScoresHybridSpecializedCold import ScoresHybridSpecializedCold
from ScoresHybridSpecializedV2Cold import ScoresHybridSpecializedV2Cold
from ScoresHybridSpecializedV2Mid import ScoresHybridSpecializedV2Mid
from ScoresHybridSpecializedV2Warm import ScoresHybridSpecializedV2Warm
from ScoresHybridSpecializedV2Mid12 import ScoresHybridSpecializedV2Mid12
from ScoresHybridSpecializedV2Warm12 import ScoresHybridSpecializedV2Warm12
from ScoresHybridSpecializedV3Cold import ScoresHybridSpecializedV3Cold
from ScoresHybridSpecializedV3Warm import ScoresHybridSpecializedV3Warm
from ScoresHybridSpecializedAdaptive import ScoresHybridSpecializedAdaptive
from ScoresHybridKNNCFKNNCBF import ScoresHybridKNNCFKNNCBF
from ScoresHybridUserKNNCFKNNCBF import ScoresHybridUserKNNCFKNNCBF
from FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg


######################################################################
from skopt.space import Real, Integer, Categorical
import traceback
from Utils.PoolWithSubprocess import PoolWithSubprocess


from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs


def runParameterSearch_Hybrid(recommender_class, URM_train, ICM_train, W_sparse_CF = None, URM_train_last_test = None, metric_to_optimize = "MAP",
                                     evaluator_validation = None, evaluator_test = None, evaluator_validation_earlystopping = None,
                                     output_folder_path ="result_experiments/",
                                     n_cases = 35, n_random_starts = 5, resume_from_saved = False, save_model = "best",
                                     allow_weighting = True,
                                     similarity_type_list = None):


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    URM_train = URM_train.copy()
    ICM_train = ICM_train.copy()
    # W_sparse_CF = W_sparse_CF.copy()

    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()

    try:

        output_file_name_root = recommender_class.RECOMMENDER_NAME

        parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)


       ##########################################################################################################

        if recommender_class in [ScoresHybridP3alphaKNNCBF, ScoresHybridRP3betaKNNCBF, ScoresHybridSpecialized,
                                 ScoresHybridSpecializedCold, ScoresHybridSpecializedV2Cold,
                                 ScoresHybridSpecializedV2Mid, ScoresHybridSpecializedV2Warm,
                                 ScoresHybridSpecializedV2Mid12, ScoresHybridSpecializedV2Warm12,
                                 ScoresHybridSpecializedV3Cold, ScoresHybridSpecializedV3Warm]:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK_P"] = Integer(5, 3000)
            hyperparameters_range_dictionary["alpha_P"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparameters_range_dictionary["normalize_similarity_P"] = Categorical([False])
            hyperparameters_range_dictionary["topK"] = Integer(5, 3000)
            hyperparameters_range_dictionary["shrink"] = Integer(0, 5000)
            hyperparameters_range_dictionary["similarity"] = Categorical(["tversky", "tanimoto", 'cosine', 'asymmetric'])
            hyperparameters_range_dictionary["normalize"] = Categorical([True, False])
            hyperparameters_range_dictionary["alpha"] = Real(low = 0, high = 2, prior = 'uniform')
            if recommender_class is ScoresHybridRP3betaKNNCBF:
                hyperparameters_range_dictionary["beta_P"] = Real(low = 0, high = 2, prior = 'uniform')

            if allow_weighting:
                hyperparameters_range_dictionary["feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )

        ##########################################################################################################

        if recommender_class in [ScoresHybridKNNCFKNNCBF, ScoresHybridUserKNNCFKNNCBF]:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK_CF"] = Integer(5, 1500)
            hyperparameters_range_dictionary["shrink_CF"] = Integer(0, 1500)
            hyperparameters_range_dictionary["similarity_CF"] = Categorical(
                ["tversky", "tanimoto", 'cosine', 'asymmetric'])
            hyperparameters_range_dictionary["normalize_CF"] = Categorical([True, False])
            hyperparameters_range_dictionary["topK"] = Integer(5, 1500)
            hyperparameters_range_dictionary["shrink"] = Integer(0, 1500)
            hyperparameters_range_dictionary["similarity"] = Categorical(
                ["tversky", "tanimoto", 'cosine', 'asymmetric'])
            hyperparameters_range_dictionary["normalize"] = Categorical([True, False])
            hyperparameters_range_dictionary["alpha"] = Real(low=0, high=1, prior='uniform')

            if allow_weighting:
                hyperparameters_range_dictionary["feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, ICM_train],
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=[],
                FIT_KEYWORD_ARGS={}
            )
        ##########################################################################################################

        if recommender_class is ScoresHybridSpecializedAdaptive:

            hyperparameters_range_dictionary = {}
            # Cold users hybrid
            hyperparameters_range_dictionary["topK_P_C"] = Integer(5, 1500)
            hyperparameters_range_dictionary["alpha_P_C"] = Real(low=0, high=2, prior='uniform')
            hyperparameters_range_dictionary["beta_P_C"] = Real(low=0, high=2, prior='uniform')
            hyperparameters_range_dictionary["normalize_similarity_P_C"] = Categorical([False])
            hyperparameters_range_dictionary["topK_C"] = Integer(5, 1500)
            hyperparameters_range_dictionary["shrink_C"] = Integer(0, 1500)
            hyperparameters_range_dictionary["similarity_C"] = Categorical(
                ["tversky", "tanimoto", 'cosine', 'asymmetric'])
            hyperparameters_range_dictionary["normalize_C"] = Categorical([True, False])
            # hyperparameters_range_dictionary["alpha_C"] = Real(low=0, high=1, prior='uniform')
            if allow_weighting:
                hyperparameters_range_dictionary["feature_weighting_C"] = Categorical(["none", "BM25", "TF-IDF"])

            # Warm users hybrid
            hyperparameters_range_dictionary["topK_P"] = Integer(5, 1500)
            hyperparameters_range_dictionary["alpha_P"] = Real(low=0, high=2, prior='uniform')
            hyperparameters_range_dictionary["beta_P"] = Real(low=0, high=2, prior='uniform')
            hyperparameters_range_dictionary["normalize_similarity_P"] = Categorical([False])
            hyperparameters_range_dictionary["topK"] = Integer(5, 1500)
            hyperparameters_range_dictionary["shrink"] = Integer(0, 1500)
            hyperparameters_range_dictionary["similarity"] = Categorical(
                ["tversky", "tanimoto", 'cosine', 'asymmetric'])
            hyperparameters_range_dictionary["normalize"] = Categorical([True, False])
            # hyperparameters_range_dictionary["alpha"] = Real(low=0, high=1, prior='uniform')
            if allow_weighting:
                hyperparameters_range_dictionary["feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])

            hyperparameters_range_dictionary["threshold"] = Integer(1, 30)

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, ICM_train],
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=[],
                FIT_KEYWORD_ARGS={}
            )

        ##########################################################################################################

        if recommender_class is ScoresHybridP3alphaPureSVD:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK_P"] = Integer(5, 1000)
            hyperparameters_range_dictionary["alpha_P"] = Real(low=0, high=2, prior='uniform')
            hyperparameters_range_dictionary["normalize_similarity_P"] = Categorical([False])
            hyperparameters_range_dictionary["num_factors"] = Integer(1, 500)


            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=[],
                FIT_KEYWORD_ARGS={}
            )

        ##########################################################################################################

        if recommender_class is CFW_D_Similarity_Linalg:
            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
            hyperparameters_range_dictionary["add_zeros_quota"] = Real(low=0, high=1, prior='uniform')
            hyperparameters_range_dictionary["normalize_similarity"] = Categorical([True, False])

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, ICM_train, W_sparse_CF],
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=[],
                FIT_KEYWORD_ARGS={}
            )

       #########################################################################################################

        if URM_train_last_test is not None:
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
        else:
            recommender_input_args_last_test = None

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        parameterSearch.search(recommender_input_args,
                               parameter_search_space = hyperparameters_range_dictionary,
                               n_cases = n_cases,
                               n_random_starts = n_random_starts,
                               resume_from_saved = resume_from_saved,
                               save_model = save_model,
                               output_folder_path = output_folder_path,
                               output_file_name_root = output_file_name_root,
                               metric_to_optimize = metric_to_optimize,
                               recommender_input_args_last_test = recommender_input_args_last_test)


    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()


import os, multiprocessing
from functools import partial

from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample



def read_data_split_and_search():
    from Data_manager.RecSys2020 import RecSys2020Reader
    from datetime import datetime
    from scipy import sparse as sps

    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """


    #URM_train, URM_test = split_train_in_two_percentage_global_sample(dataset.get_URM_all(), train_percentage = 0.80)
    #URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.80)

    URM_all, user_id_unique, item_id_unique = RecSys2020Reader.load_urm()
    ICM_all = RecSys2020Reader.load_icm_asset()
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.95)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.90)
    ICM_train, ICM_test = split_train_in_two_percentage_global_sample(ICM_all, train_percentage=0.95)
    ICM_train, ICM_validation = split_train_in_two_percentage_global_sample(ICM_train, train_percentage=0.90)

    URM_ICM_train = sps.vstack([URM_train, ICM_all.T])
    URM_ICM_train = URM_ICM_train.tocsr()


    output_folder_path = "ParamResultsExperiments/SKOPT_ScoresHybridP3alphaKNNCBF_specialized_extend_param"
    output_folder_path += datetime.now().strftime('%b%d_%H-%M-%S/')


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    hybrid_algorithm_list = [
        #ScoresHybridP3alphaKNNCBF,
        ScoresHybridRP3betaKNNCBF,
        #ScoresHybridP3alphaPureSVD,
        #ScoresHybridSpecialized,
        #ScoresHybridSpecializedCold,
        #ScoresHybridSpecializedV2Cold,
        #ScoresHybridSpecializedV3Cold,
        #ScoresHybridSpecializedV2Mid,
        #ScoresHybridSpecializedV2Warm,
        #ScoresHybridSpecializedV3Warm,
        #ScoresHybridSpecializedV2Mid12,
        #ScoresHybridSpecializedV2Warm12,
        #ScoresHybridSpecializedAdaptive,
        #ScoresHybridKNNCFKNNCBF,
        #ScoresHybridUserKNNCFKNNCBF,
        #CFW_D_Similarity_Linalg
    ]

    from Base.Evaluation.Evaluator import EvaluatorHoldout

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10])

    #cf = ItemKNNCFRecommender(URM_ICM_train)
    #cf.fit(**{"topK": 259, "shrink": 24, "similarity": "cosine", "normalize": True})
    #W_sparse_CF = cf.W_sparse

    runParameterSearch_Hybrid_partial = partial(runParameterSearch_Hybrid,
                                                       URM_train = URM_ICM_train,
                                                       ICM_train = URM_ICM_train.T,
                                                       #W_sparse_CF = W_sparse_CF,
                                                       metric_to_optimize = "MAP",
                                                       n_cases = 100,
                                                       n_random_starts=20,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = output_folder_path)




    from Utils.PoolWithSubprocess import PoolWithSubprocess


    pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()-1), maxtasksperchild=1)
    resultList = pool.map_async(runParameterSearch_Hybrid_partial, hybrid_algorithm_list)
    pool.close()
    pool.join()

    for recommender_class in hybrid_algorithm_list:

        try:

            runParameterSearch_Hybrid_partial(recommender_class)

        except Exception as e:

            print("On recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()


if __name__ == '__main__':

    read_data_split_and_search()
