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
import time

# https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi/blob/master/Practice%2009%20-%20SLIM%20BPR.ipynb
# https://github.com/nicolo-felicioni/recsys-polimi-2019/tree/master/Hybrid


def crossval(URM_all, ICM_all, target_ids, k):

    seed = 1234 + k #+ int(time.time())
    np.random.seed(seed)
    tp = 0.75
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=tp)
    ICM_train, ICM_test = train_test_holdout(ICM_all, train_perc=0.95)
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)
    args = {}

    p3alpha = P3alphaRecommender.P3alphaRecommender(URM_train)
    try:
        args = {"topK": 991, "alpha": 0.4705816992313091, "normalize_similarity": False}
        p3alpha.load_model('SavedModels\\', p3alpha.RECOMMENDER_NAME + toFileName(args) + ",s=" + str(seed) + ",tp=" + str(tp) + ".zip")
    except:
        print("Saved model not found. Fitting a new one...")
        p3alpha.fit(**args)
        p3alpha.save_model('SavedModels\\', p3alpha.RECOMMENDER_NAME + toFileName(args) + ",s=" + str(seed) + ",tp=" + str(tp))

    rp3beta = RP3betaRecommender.RP3betaRecommender(URM_train)
    try:
        args = {"topK": 991, "alpha": 0.4705816992313091, "beta": 0.37, "normalize_similarity": False}
        rp3beta.load_model('SavedModels\\', rp3beta.RECOMMENDER_NAME + toFileName(args) + ",s=" + str(seed) + ",tp=" + str(tp) + ".zip")
    except:
        print("Saved model not found. Fitting a new one...")
        rp3beta.fit(**args)
        rp3beta.save_model('SavedModels\\', rp3beta.RECOMMENDER_NAME + toFileName(args) + ",s=" + str(seed) + ",tp=" + str(tp))

    itemKNNCF = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    try:
        args = {"topK": 1000, "shrink": 732, "similarity": "cosine", "normalize": True, "feature_weighting": "TF-IDF"}
        itemKNNCF.load_model('SavedModels\\', itemKNNCF.RECOMMENDER_NAME + toFileName(args) + ",s=" + str(seed) + ",tp=" + str(tp) + ".zip")
    except:
        print("Saved model not found. Fitting a new one...")
        itemKNNCF.fit(**args)
        itemKNNCF.save_model('SavedModels\\', itemKNNCF.RECOMMENDER_NAME + toFileName(args) + ",s=" + str(seed) + ",tp=" + str(tp))

    userKNNCF = UserKNNCFRecommender.UserKNNCFRecommender(URM_train)
    try:
        args = {"topK": 131, "shrink": 2, "similarity": "cosine", "normalize": True}
        userKNNCF.load_model('SavedModels\\', userKNNCF.RECOMMENDER_NAME + toFileName(args) + ",s=" + str(seed) + ",tp=" + str(tp) + ".zip")
    except:
        print("Saved model not found. Fitting a new one...")
        userKNNCF.fit(**args)
        userKNNCF.save_model('SavedModels\\', userKNNCF.RECOMMENDER_NAME + toFileName(args) + ",s=" + str(seed) + ",tp=" + str(tp))

    itemKNNCBF = ItemKNNCBFRecommender.ItemKNNCBFRecommender(URM_train, ICM_all)
    try:
        args = {"topK": 700, "shrink": 100, "similarity": 'jaccard', "normalize": True, "feature_weighting": "TF-IDF"}
        itemKNNCBF.load_model('SavedModels\\', itemKNNCBF.RECOMMENDER_NAME + toFileName(args) + ",s=" + str(seed) + ",tp=" + str(tp) + ".zip")
    except:
        print("Saved model not found. Fitting a new one...")
        itemKNNCBF.fit(**args)
        itemKNNCBF.save_model('SavedModels\\', itemKNNCBF.RECOMMENDER_NAME + toFileName(args) + ",s=" + str(seed) + ",tp=" + str(tp))

    #cfw = CFW_D_Similarity_Linalg.CFW_D_Similarity_Linalg(URM_train, ICM_train, itemKNNCF.W_sparse)
    #cfw.fit(show_max_performance=False, logFile=None, loss_tolerance=1e-6,
    #        iteration_limit=500000, damp_coeff=0.5, topK=900, add_zeros_quota=0.5, normalize_similarity=True)

    # Need to change bpr code to avoid memory error, useless since it's bad
    # bpr = SLIM_BPR_Cython(URM_train, recompile_cython=False)
    # bpr.fit(**{"topK": 1000, "epochs": 130, "symmetric": False, "sgd_mode": "adagrad", "lambda_i": 1e-05,
    #          "lambda_j": 0.01, "learning_rate": 0.0001})

    pureSVD = PureSVDRecommender.PureSVDRecommender(URM_train)
    pureSVD.fit(num_factors=1000)

    hyb = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, p3alpha, itemKNNCBF)
    hyb.fit(alpha=0.5)

    # Kaggle MAP 0.084 rp3beta, itemKNNCBF
    hyb2 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, pureSVD, itemKNNCBF)
    hyb2.fit(alpha=0.5)

    # Kaggle MAP 0.08667
    hyb3 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb, hyb2)
    hyb3.fit(alpha=0.5)

    #hyb3 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, p3alpha, userKNNCF)
    #hyb3.fit(alpha=0.5)

    hyb5 = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_train, ICM_all)
    # Kaggle MAP 0.08856
    try:
        # Full values: "alpha_P": 0.4108657561671193, "alpha": 0.6290871066510789
        args = {"topK_P": 903, "alpha_P": 0.41086575, "normalize_similarity_P": False, "topK": 448, "shrink": 20,
                "similarity": "tversky", "normalize": True, "alpha": 0.6290871, "feature_weighting": "TF-IDF"}
        hyb5.load_model('SavedModels\\', hyb5.RECOMMENDER_NAME + toFileName(args) + ",s=" + str(seed) + ",tp=" + str(tp) + ".zip")
    except:
        print("Saved model not found. Fitting a new one...")
        hyb5.fit(**args)
        hyb5.save_model('SavedModels\\', hyb5.RECOMMENDER_NAME + toFileName(args) + ",s=" + str(seed) + ",tp=" + str(tp))

    # hyb5.fit(**{"topK_P": 1000, "alpha_P": 0.5432601071314623, "normalize_similarity_P": True, "topK": 620, "shrink": 0,
    #             "similarity": "tversky", "normalize": False, "alpha": 0.5707347522847057, "feature_weighting": "BM25"})

    # Kaggle MAP 0.086 :(
    #hyb6 = ScoresHybrid3Recommender.ScoresHybrid3Recommender(URM_train, rp3beta, itemKNNCBF, p3alpha)
    #hyb6.fit()

    hyb6 = ScoresHybridRP3betaKNNCBF.ScoresHybridRP3betaKNNCBF(URM_train, ICM_all)
    try:
        # Full values: "alpha_P": 0.5081918012150626, "alpha": 0.44740093610861603
        args = {"topK_P": 623, "alpha_P": 0.5081918, "normalize_similarity_P": False, "topK": 1000,
                "shrink": 1000, "similarity": "tversky", "normalize": True, "alpha": 0.4474009, "beta_P": 0.0,
                "feature_weighting": "TF-IDF"}
        hyb6.load_model('SavedModels\\', hyb6.RECOMMENDER_NAME + toFileName(args) + ",s=" + str(seed) + ",tp=" + str(tp) + ".zip")
    except:
        print("Saved model not found. Fitting a new one...")
        hyb6.fit(**args)
        hyb6.save_model('SavedModels\\', hyb6.RECOMMENDER_NAME + toFileName(args) + ",s=" + str(seed) + ",tp=" + str(tp))

    v0 = evaluator_validation.evaluateRecommender(hyb)[0][10]["MAP"]
    v1 = evaluator_validation.evaluateRecommender(hyb2)[0][10]["MAP"]
    v2 = evaluator_validation.evaluateRecommender(hyb3)[0][10]["MAP"]
    v3 = evaluator_validation.evaluateRecommender(hyb5)[0][10]["MAP"]
    v4 = evaluator_validation.evaluateRecommender(hyb6)[0][10]["MAP"]

    #item_list = hyb3.recommend(target_ids, cutoff=10)
    #CreateCSV.create_csv(target_ids, item_list, 'ItemKNNCBF__RP3beta')

    return [v0, v1, v2, v3, v4]


def toFileName(args):
    # Modify arguments to make them fit in file name
    return str(args).replace("'", "").replace(":", "=").replace("normalize_similarity", "n_sim")\
                    .replace("feature_weighting", "f_w").replace("normalize", "nor").replace("similarity", "sim")\
                    .replace("alpha", "a").replace("beta", "b").replace(" ", "").replace("False", "F")\
                    .replace("True", "T").replace("topK", "K").replace("shrink", "sh").replace("tversky", "tv")


if __name__ == '__main__':

    URM_all, user_id_unique, item_id_unique = RecSys2020Reader.load_urm()
    ICM_all = RecSys2020Reader.load_icm_asset()
    target_ids = RecSys2020Reader.load_target()

    # seed = 12341
    k_fold = 7
    n_models = 5

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

