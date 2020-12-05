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
import ScoresHybridRP3betaKNNCBF
import CreateCSV
import multiprocessing
from Utils.PoolWithSubprocess import PoolWithSubprocess
from functools import partial
import RankingHybrid
import time

# https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi/blob/master/Practice%2009%20-%20SLIM%20BPR.ipynb
# https://github.com/nicolo-felicioni/recsys-polimi-2019/tree/master/Hybrid


def crossval(URM_all, ICM_all, target_ids, k):

    seed = 1234 + k #+ int(time.time())
    np.random.seed()
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.90)
    ICM_train, ICM_test = train_test_holdout(ICM_all, train_perc=0.95)
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)
    args = {}

    p3alpha = P3alphaRecommender.P3alphaRecommender(URM_train)
    args = {"topK": 991, "alpha": 0.4705816992313091, "normalize_similarity": False}
    p3alpha.fit(**args)

    #p3alpha2 = P3alphaRecommender.P3alphaRecommender(URM_train)
    #args = {"topK": 400, "alpha": 0.5305816992313091, "normalize_similarity": False}
    #p3alpha2.fit(**args)

    #rp3beta = RP3betaRecommender.RP3betaRecommender(URM_train)
    #args = {"topK": 991, "alpha": 0.4705816992313091, "beta": 0.15, "normalize_similarity": False}
    #rp3beta.fit(**args)

    itemKNNCF = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    args = {"topK": 1000, "shrink": 732, "similarity": "cosine", "normalize": True, "feature_weighting": "TF-IDF"}
    itemKNNCF.fit(**args)

    userKNNCF = UserKNNCFRecommender.UserKNNCFRecommender(URM_train)
    args = {"topK": 131, "shrink": 2, "similarity": "cosine", "normalize": True}
    userKNNCF.fit(**args)

    itemKNNCBF = ItemKNNCBFRecommender.ItemKNNCBFRecommender(URM_train, ICM_all)
    args = {"topK": 700, "shrink": 100, "similarity": 'jaccard', "normalize": True, "feature_weighting": "TF-IDF"}
    itemKNNCBF.fit(**args)

    itemKNNCBF2 = ItemKNNCBFRecommender.ItemKNNCBFRecommender(URM_train, ICM_all)
    args = {"topK": 200, "shrink": 15, "similarity": 'jaccard', "normalize": True, "feature_weighting": "TF-IDF"}
    itemKNNCBF2.fit(**args)

    #cfw = CFW_D_Similarity_Linalg.CFW_D_Similarity_Linalg(URM_train, ICM_train, itemKNNCF.W_sparse)
    #cfw.fit(show_max_performance=False, logFile=None, loss_tolerance=1e-6,
    #        iteration_limit=500000, damp_coeff=0.5, topK=900, add_zeros_quota=0.5, normalize_similarity=True)

    # Need to change bpr code to avoid memory error, useless since it's bad
    #bpr = SLIM_BPR_Cython(URM_train, recompile_cython=False)
    #bpr.fit(**{"topK": 1000, "epochs": 130, "symmetric": False, "sgd_mode": "adagrad", "lambda_i": 1e-05,
    #          "lambda_j": 0.01, "learning_rate": 0.0001})

    pureSVD = PureSVDRecommender.PureSVDRecommender(URM_train)
    pureSVD.fit(num_factors=340)

    #hyb = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, p3alpha, itemKNNCBF)
    #hyb.fit(alpha=0.5)
    hyb = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, itemKNNCBF, pureSVD)
    hyb.fit(alpha=0.5)

    # Kaggle MAP 0.084 rp3beta, itemKNNCBF
    #hyb2 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, p3alpha, itemKNNCBF)
    #hyb2.fit(alpha=0.5)
    hyb2 = ItemKNNSimilarityHybridRecommender.ItemKNNSimilarityHybridRecommender(URM_train, itemKNNCBF.W_sparse, itemKNNCF.W_sparse)
    hyb2.fit(topK=1600)

    # Kaggle MAP 0.08667
    hyb3 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb, hyb2)
    hyb3.fit(alpha=0.5)
    #hyb3 = RankingHybrid.RankingHybrid(URM_train, hyb, hyb2)

    #hyb3 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, p3alpha, userKNNCF)
    #hyb3.fit(alpha=0.5)

    hyb5 = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_train, ICM_all)
    # Kaggle MAP 0.08856
    args = {"topK_P": 903, "alpha_P": 0.4108657561671193, "normalize_similarity_P": False, "topK": 448, "shrink": 20,
            "similarity": "tversky", "normalize": True, "alpha": 0.6290871066510789, "feature_weighting": "TF-IDF"}
    hyb5.fit(**args)

    # hyb5.fit(**{"topK_P": 1000, "alpha_P": 0.5432601071314623, "normalize_similarity_P": True, "topK": 620, "shrink": 0,
    #             "similarity": "tversky", "normalize": False, "alpha": 0.5707347522847057, "feature_weighting": "BM25"})

    # Kaggle MAP 0.086 :(
    #hyb6 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb3, hyb5)
    #hyb6.fit()
    hyb6 = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_train, ICM_all)
    args = {"topK_P": 756, "alpha_P": 0.5292654015790155, "normalize_similarity_P": False, "topK": 1000, "shrink": 47,
            "similarity": "tversky", "normalize": False, "alpha": 0.5207647439152092, "feature_weighting": "none"}
    hyb6.fit(**args)

    '''hyb6 = ScoresHybridRP3betaKNNCBF.ScoresHybridRP3betaKNNCBF(URM_train, ICM_all)
    args = {"topK_P": 623, "alpha_P": 0.5081918012150626, "normalize_similarity_P": False, "topK": 1000,
            "shrink": 1000, "similarity": "tversky", "normalize": True, "alpha": 0.44740093610861603, "beta_P": 0.0,
            "feature_weighting": "TF-IDF"}
    hyb6.fit(**args)'''

    hyb7 = RankingHybrid.RankingHybrid(URM_train, hyb6, hyb3)

    v0 = evaluator_validation.evaluateRecommender(hyb)[0][10]["MAP"]
    v1 = evaluator_validation.evaluateRecommender(hyb2)[0][10]["MAP"]
    v2 = evaluator_validation.evaluateRecommender(hyb3)[0][10]["MAP"]
    #v2 = 0
    v3 = evaluator_validation.evaluateRecommender(hyb5)[0][10]["MAP"]
    v4 = evaluator_validation.evaluateRecommender(hyb6)[0][10]["MAP"]
    #v4 = 0
    v5 = evaluator_validation.evaluateRecommender(hyb7)[0][10]["MAP"]

    #item_list = hyb6.recommend(target_ids, cutoff=10)
    #CreateCSV.create_csv(target_ids, item_list, 'HybPureSVD')

    return [v0, v1, v2, v3, v4, v5]


if __name__ == '__main__':

    URM_all, user_id_unique, item_id_unique = RecSys2020Reader.load_urm()
    ICM_all = RecSys2020Reader.load_icm_asset()
    target_ids = RecSys2020Reader.load_target()

    # seed = 12341
    k_fold = 2
    n_models = 6

    ar = [[0 for x in range(n_models)] for y in range(k_fold)]
    cross_partial = partial(crossval, URM_all, ICM_all, target_ids)
    ks = [x for x in range(0, k_fold)]

    pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()-1), maxtasksperchild=1)
    resultList = pool.map(cross_partial, ks)
    pool.close()
    pool.join()

    print("Hyb: " + str(np.mean(resultList, axis=0)[0]))
    print("Hyb2: " + str(np.mean(resultList, axis=0)[1]))
    print("Hyb3: " + str(np.mean(resultList, axis=0)[2]))
    print("Hyb5 P3_CBF_tuned1: " + str(np.mean(resultList, axis=0)[3]))
    print("Hyb6: " + str(np.mean(resultList, axis=0)[4]))
    print("Hyb7: " + str(np.mean(resultList, axis=0)[5]))

