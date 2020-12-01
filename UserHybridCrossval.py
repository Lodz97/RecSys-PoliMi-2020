from Data_manager.RecSys2020 import RecSys2020Reader
from Notebooks_utils.data_splitter import train_test_holdout
import matplotlib.pyplot as pyplot
import numpy as np
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from GraphBased import P3alphaRecommender, RP3betaRecommender
from SLIM_ElasticNet import SLIMElasticNetRecommender
from Base.Evaluation.Evaluator import EvaluatorHoldout
from MatrixFactorization.Cython import MatrixFactorization_Cython
from MatrixFactorization.PyTorch import MF_MSE_PyTorch
from MatrixFactorization import IALSRecommender, NMFRecommender, PureSVDRecommender
from KNN import ItemKNNCBFRecommender, ItemKNNCFRecommender, ItemKNNCustomSimilarityRecommender,\
                ItemKNNSimilarityHybridRecommender, UserKNNCFRecommender
from EASE_R import EASE_R_Recommender
from FeatureWeighting import CFW_D_Similarity_Linalg
import ItemKNNScoresHybridRecommender
import ScoresHybrid3Recommender
import ScoresHybridP3alphaKNNCBF
import CreateCSV
import multiprocessing
from Utils.PoolWithSubprocess import PoolWithSubprocess
from functools import partial

# https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi/blob/master/Practice%2009%20-%20SLIM%20BPR.ipynb
# https://github.com/nicolo-felicioni/recsys-polimi-2019/tree/master/Hybrid


def crossval(URM_all, ICM_all, k):

    seed = 1234 + k
    np.random.seed(seed)
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.97)
    ICM_train, ICM_test = train_test_holdout(ICM_all, train_perc=0.97)
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)

    p3alpha = P3alphaRecommender.P3alphaRecommender(URM_train)
    p3alpha.fit(**{"topK": 991, "alpha": 0.4705816992313091, "normalize_similarity": False})

    itemKNNCF = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    itemKNNCF.fit(**{"topK": 1000, "shrink": 732, "similarity": "cosine", "normalize": True,
                     "feature_weighting": "TF-IDF"})

    userKNNCF = UserKNNCFRecommender.UserKNNCFRecommender(URM_train)
    userKNNCF.fit(**{"topK": 131, "shrink": 2, "similarity": "cosine", "normalize": True})

    itemKNNCBF = ItemKNNCBFRecommender.ItemKNNCBFRecommender(URM_train, ICM_train)
    itemKNNCBF.fit(topK=700, shrink=200, similarity='jaccard', normalize=True, feature_weighting = "TF-IDF")


    hyb = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, p3alpha, itemKNNCF)
    hyb.fit(alpha=0.5)

    # Kaggle MAP 0.084
    hyb2 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, p3alpha, itemKNNCBF)
    hyb2.fit(alpha=0.5)

    # Kaggle MAP 0.08667
    #hyb3 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb, hyb2)
    #hyb3.fit(alpha=0.5)
    hyb3 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, p3alpha, userKNNCF)
    hyb3.fit(alpha=0.5)

    hyb5 = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_train, ICM_train)
    hyb6 = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_train, ICM_train)
    #hyb5.fit(**{"topK_P": 316, "alpha_P": 0.682309490338903, "normalize_similarity_P": False, "topK": 760, "shrink": 33,
    #           "similarity": "asymmetric", "normalize": True, "feature_weighting": "BM25"})
    #hyb5.fit(**{"topK_P": 715, "alpha_P": 0.47489080414690943, "normalize_similarity_P": True, "topK": 994, "shrink": 179,
    #           "similarity": "tanimoto", "normalize": True, "alpha": 0.5666004103014026, "feature_weighting": "TF-IDF"})
    #hyb5.fit(**{"topK_P": 531, "alpha_P": 0.5783052607747065, "normalize_similarity_P": False, "topK": 162, "shrink": 723,
    #            "similarity": "tversky", "normalize": True, "alpha": 0.5444830213942248, "feature_weighting": "TF-IDF"})
    hyb5.fit(**{"topK_P": 903, "alpha_P": 0.4108657561671193, "normalize_similarity_P": True, "topK": 448, "shrink": 20,
                "similarity": "tversky", "normalize": True, "alpha": 0.6290871066510789, "feature_weighting": "TF-IDF"})
    hyb6.fit(**{"topK_P": 1000, "alpha_P": 0.5432601071314623, "normalize_similarity_P": True, "topK": 620, "shrink": 0,
                "similarity": "tversky", "normalize": False, "alpha": 0.5707347522847057, "feature_weighting": "BM25"})

    v0 = evaluator_validation.evaluateRecommender(hyb)[0][10]["MAP"]
    v1 = evaluator_validation.evaluateRecommender(hyb2)[0][10]["MAP"]
    v2 = evaluator_validation.evaluateRecommender(hyb3)[0][10]["MAP"]
    v3 = evaluator_validation.evaluateRecommender(hyb5)[0][10]["MAP"]
    v4 = evaluator_validation.evaluateRecommender(hyb6)[0][10]["MAP"]

    return [v0, v1, v2, v3, v4]



if __name__ == '__main__':

    URM_all, user_id_unique, item_id_unique = RecSys2020Reader.load_urm()
    ICM_all = RecSys2020Reader.load_icm_asset()
    target_ids = RecSys2020Reader.load_target()

    # seed = 12341
    k_fold = 6
    n_models = 5

    ar = [[0 for x in range(n_models)] for y in range(k_fold)]
    cross_partial = partial(crossval, URM_all, ICM_all)
    ks = [x for x in range(0, k_fold)]

    pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()-1), maxtasksperchild=1)
    resultList = pool.map(cross_partial, ks)
    pool.close()
    pool.join()

    print("Hyb1 P3_CF: " + str(np.mean(resultList, axis=0)[0]))
    print("Hyb2 P3_CBF: " + str(np.mean(resultList, axis=0)[1]))
    print("Hyb3 Hyb1_Hyb2: " + str(np.mean(resultList, axis=0)[2]))
    print("Hyb4 P3_CBF_tuned1: " + str(np.mean(resultList, axis=0)[3]))
    print("Hyb5 P3_CBF_tuned2: " + str(np.mean(resultList, axis=0)[4]))

