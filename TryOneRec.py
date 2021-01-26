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
import ScoresHybridP3alphaKNNCBF, ScoresHybridSpecializedV2Mid
import CreateCSV
from scipy import sparse as sps

# https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi/blob/master/Practice%2009%20-%20SLIM%20BPR.ipynb
# https://github.com/nicolo-felicioni/recsys-polimi-2019/tree/master/Hybrid


if __name__ == '__main__':
    URM_all, user_id_unique, item_id_unique = RecSys2020Reader.load_urm()
    ICM_all = RecSys2020Reader.load_icm_asset()
    target_ids = RecSys2020Reader.load_target()

    URM_ICM_all = RecSys2020Reader.load_urm_icm()

    '''item_popularity = np.ediff1d(URM_all.tocsc().indptr)
    print(item_popularity)
    item_popularity = np.sort(item_popularity)
    pyplot.plot(item_popularity, 'ro')
    pyplot.ylabel('Num Interactions ')
    pyplot.xlabel('Sorted Item')
    pyplot.show()

    user_activity = np.ediff1d(URM_all.indptr)
    user_activity = np.sort(user_activity)

    pyplot.plot(user_activity, 'ro')
    pyplot.ylabel('Num Interactions ')
    pyplot.xlabel('Sorted User')
    pyplot.show()'''

    #np.random.seed(1234)
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.90)
    ICM_train, ICM_test = train_test_holdout(ICM_all, train_perc=0.9)
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)

    URM_ICM_train = sps.vstack([URM_train, ICM_all.T])
    URM_ICM_train = URM_ICM_train.tocsr()
    URM_ICM_train2 = sps.hstack([ICM_all, URM_train.T])
    URM_ICM_train2 = URM_ICM_train2.tocsr()

    earlystopping_keywargs = {"validation_every_n": 10,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator_validation,
                              "lower_validations_allowed": 5,
                              "validation_metric": "MAP",
                              }

    # MAP 0.07 Kaggle "topK": 131, "shrink": 2, "similarity": "cosine", "normalize": true}
    #recommender = UserKNNCFRecommender.UserKNNCFRecommender(URM_train)
    #recommender.fit(**{"topK": 131, "shrink": 2, "similarity": "cosine", "normalize": True})

    '''itemKNNCF = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    itemKNNCF.fit(**{"topK": 1000, "shrink": 732, "similarity": "cosine", "normalize": True,
                     "feature_weighting": "TF-IDF"})
    itemKNNCF2 = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_ICM_train)
    itemKNNCF2.fit(**{"topK": 1000, "shrink": 732, "similarity": "cosine", "normalize": True,
                     "feature_weighting": "TF-IDF"})
    itemKNNCF3 = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_ICM_train2.T)
    itemKNNCF3.fit(**{"topK": 1000, "shrink": 732, "similarity": "cosine", "normalize": True,
                     "feature_weighting": "TF-IDF"})'''
    itemKNNCBF = ItemKNNCBFRecommender.ItemKNNCBFRecommender(URM_train, URM_train.T)
    itemKNNCBF.fit(topK=700, shrink=200, similarity='jaccard', normalize=True, feature_weighting="TF-IDF")
    itemKNNCBF2 = ItemKNNCBFRecommender.ItemKNNCBFRecommender(URM_train, URM_ICM_train.T)
    itemKNNCBF2.fit(topK=700, shrink=200, similarity='jaccard', normalize=True, feature_weighting="TF-IDF")
    itemKNNCBF3 = ItemKNNCBFRecommender.ItemKNNCBFRecommender(URM_train, URM_ICM_train2)
    itemKNNCBF3.fit(topK=700, shrink=200, similarity='jaccard', normalize=True, feature_weighting="TF-IDF")

    #cfw = CFW_D_Similarity_Linalg.CFW_D_Similarity_Linalg(URM_train, ICM_train, itemKNNCF.W_sparse)
    #cfw.fit(show_max_performance=False, logFile=None, loss_tolerance=1e-6,
    #        iteration_limit=500000, damp_coeff=0.5, topK=900, add_zeros_quota=0.5, normalize_similarity=True)

    hyb5 = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_train, ICM_all)
    # Kaggle MAP 0.08856
    args = {"topK_P": 903, "alpha_P": 0.4108657561671193, "normalize_similarity_P": False, "topK": 448, "shrink": 20,
            "similarity": "tversky", "normalize": True, "alpha": 0.6290871066510789, "feature_weighting": "TF-IDF"}
    hyb5.fit(**args)

    hyb6 = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_train, ICM_all)
    #args = {"topK_P": 1303, "alpha_P": 0.4808657561671193, "normalize_similarity_P": False, "topK": 848, "shrink": 1,
    #        "similarity": "tversky", "normalize": False, "alpha": 0.5790871066510789, "feature_weighting": "TF-IDF"}
    args = {"topK_P": 756, "alpha_P": 0.5292654015790155, "normalize_similarity_P": False, "topK": 1000, "shrink": 47,
            "similarity": "tversky", "normalize": False, "alpha": 0.5207647439152092, "feature_weighting": "none"}
    hyb6.fit(**args)

    svd = PureSVDRecommender.PureSVDRecommender(URM_train)
    svd.fit()

    #cf = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_ICM_train)
    #cf.fit(**{"topK": 259, "shrink": 24, "similarity": "cosine", "normalize": True})
    #W_sparse_CF = cf.W_sparse
    #hyb7 = CFW_D_Similarity_Linalg.CFW_D_Similarity_Linalg(URM_train, ICM_all, W_sparse_CF)
    #hyb7.fit(**{"topK": 575, "add_zeros_quota": 0.6070346405411541, "normalize_similarity": False})

    hyb7 = ScoresHybridSpecializedV2Mid.ScoresHybridSpecializedV2Mid(URM_ICM_train, URM_ICM_train.T)
    hyb7.fit(**{"topK_P": 516, "alpha_P": 0.4753488773601332, "normalize_similarity_P": False, "topK": 258, "shrink": 136,
             "similarity": "asymmetric", "normalize": False, "alpha": 0.48907705969537585, "feature_weighting": "BM25"})

    print(evaluator_validation.evaluateRecommender(svd))
    print(evaluator_validation.evaluateRecommender(itemKNNCBF))
    print(evaluator_validation.evaluateRecommender(itemKNNCBF2))
    print(evaluator_validation.evaluateRecommender(itemKNNCBF3))
    print(evaluator_validation.evaluateRecommender(hyb7))
    print(evaluator_validation.evaluateRecommender(hyb5))
    print(evaluator_validation.evaluateRecommender(hyb6))

    #item_list = recommender.recommend(target_ids, cutoff=10)
    #CreateCSV.create_csv(target_ids, item_list, 'MyRec')
