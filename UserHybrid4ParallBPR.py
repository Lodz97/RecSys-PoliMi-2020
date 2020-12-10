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
import ScoresHybridP3alphaPureSVD
import RankingHybrid
import CreateCSV
from sklearn.preprocessing import normalize
from scipy import sparse as sps
from Utils.PoolWithSubprocess import PoolWithSubprocess
import multiprocessing
from functools import partial
import time


def fitRec(rec_args_name):
    rec = rec_args_name[0]
    args = rec_args_name[1]
    name = rec_args_name[2]
    rec.fit(**args)
    return [rec, name]


def compute_group_MAP(args, group_id):
    block_size = args["block_size"]
    profile_length = args["profile_length"]
    sorted_users = args["sorted_users"]
    cutoff = args["cutoff"]
    URM_test = args["URM_test"]
    hyb = args["hyb"]
    hyb2 = args["hyb2"]
    hyb3 = args["hyb3"]
    hyb5 = args["hyb5"]
    hyb6 = args["hyb6"]
    hyb7 = args["hyb7"]

    MAP_hyb_per_group = []
    MAP_hyb2_per_group = []
    MAP_hyb3_per_group = []
    MAP_hyb5_per_group = []
    MAP_hyb6_per_group = []
    MAP_hyb7_per_group = []

    start_pos = group_id * block_size
    end_pos = min((group_id + 1) * block_size, len(profile_length))

    users_in_group = sorted_users[start_pos:end_pos]

    users_in_group_p_len = profile_length[users_in_group]

    print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                  users_in_group_p_len.mean(),
                                                                  users_in_group_p_len.min(),
                                                                  users_in_group_p_len.max()))

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)

    '''results, _ = evaluator_test.evaluateRecommender(p3alpha)
    MAP_p3alpha_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(itemKNNCF)
    MAP_itemKNNCF_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(itemKNNCBF)
    MAP_itemKNNCBF_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(pureSVD)
    MAP_pureSVD_per_group.append(results[cutoff]["MAP"])'''

    results, _ = evaluator_test.evaluateRecommender(hyb)
    MAP_hyb_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(hyb2)
    MAP_hyb2_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(hyb3)
    MAP_hyb3_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(hyb5)
    MAP_hyb5_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(hyb6)
    MAP_hyb6_per_group.append(results[cutoff]["MAP"])

    results, _ = evaluator_test.evaluateRecommender(hyb7)
    MAP_hyb7_per_group.append(results[cutoff]["MAP"])

    return [MAP_hyb_per_group, MAP_hyb2_per_group, MAP_hyb3_per_group, MAP_hyb5_per_group, MAP_hyb6_per_group,
            MAP_hyb7_per_group]



if __name__ == '__main__':
    start_time = time.time()

    URM_all, user_id_unique, item_id_unique = RecSys2020Reader.load_urm()
    ICM_all = RecSys2020Reader.load_icm_asset()
    target_ids = RecSys2020Reader.load_target()

    #np.random.seed(12341)
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.80)
    # ICM_train, ICM_test = train_test_holdout(ICM_all, train_perc=0.995)
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)
    ICM_train = ICM_all

    URM_ICM_train = sps.vstack([URM_train, ICM_all.T])
    URM_ICM_train = URM_ICM_train.tocsr()

    l_list = []
    profile_length = np.ediff1d(URM_train.indptr)
    block_size = int(len(profile_length) * 0.2)
    sorted_users = np.argsort(profile_length)
    groups = 5
    rec_list = []
    arg_list = []
    name_list = []

    for group_id in range(0, groups):
        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, len(profile_length))

        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = profile_length[users_in_group]
        l_list.append(len(users_in_group))

        print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                      users_in_group_p_len.mean(),
                                                                      users_in_group_p_len.min(),
                                                                      users_in_group_p_len.max()))

    p3alpha = P3alphaRecommender.P3alphaRecommender(URM_train)
    p3alpha_args = {"topK": 991, "alpha": 0.4705816992313091, "normalize_similarity": False}
    rec_list.append(p3alpha)
    arg_list.append(p3alpha_args)
    name_list.append("p3alpha")

    itemKNNCF = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    itemKNNCF_args = {"topK": 1000, "shrink": 732, "similarity": "cosine", "normalize": True,
                      "feature_weighting": "TF-IDF"}
    rec_list.append(itemKNNCF)
    arg_list.append(itemKNNCF_args)
    name_list.append("itemKNNCF")

    itemKNNCBF = ItemKNNCBFRecommender.ItemKNNCBFRecommender(URM_train, ICM_train)
    itemKNNCBF_args = {"topK": 700, "shrink": 200, "similarity": 'jaccard', "normalize": True,
                       "feature_weighting": "TF-IDF"}
    rec_list.append(itemKNNCBF)
    arg_list.append(itemKNNCBF_args)
    name_list.append("itemKNNCBF")

    hyb5 = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_train, ICM_train)
    hyb6x = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_ICM_train, ICM_train)
    hyb6y = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_ICM_train, ICM_train)
    # Kaggle MAP 0.08856 and 0.09159
    hyb5_args = {"topK_P": 903, "alpha_P": 0.4108657561671193, "normalize_similarity_P": False, "topK": 448,
                 "shrink": 20,
                 "similarity": "tversky", "normalize": True, "alpha": 0.6290871066510789, "feature_weighting": "TF-IDF"}
    # hyb6x(v1) of Kaggle MAP 0.08954
    # hyb6x.fit(
    #    **{"topK_P": 903, "alpha_P": 0.4108657561671193, "normalize_similarity_P": False, "topK": 448, "shrink": 20,
    #       "similarity": "tversky", "normalize": True, "alpha": 0.6290871066510789, "feature_weighting": "TF-IDF"})
    # hyb6y.fit(
    #    **{"topK_P": 903, "alpha_P": 0.4108657561671193, "normalize_similarity_P": False, "topK": 448, "shrink": 20,
    #       "similarity": "tversky", "normalize": True, "alpha": 0.6290871066510789, "feature_weighting": "TF-IDF"})
    # hyb6x_args = {"topK_P": 954, "alpha_P": 0.43832289495670274, "normalize_similarity_P": False, "topK": 628,
    #              "shrink": 941, "similarity": "cosine", "normalize": True, "alpha": 0.9643109232916722,
    #              "feature_weighting": "BM25"}
    # Small MAP improvement over previous on param search, confirmed, # hyb6x(v2) of Kaggle MAP 0.09159
    hyb6x_args = {"topK_P": 717, "alpha_P": 0.4864819897070818, "normalize_similarity_P": False, "topK": 707,
                  "shrink": 0,
                  "similarity": "tversky", "normalize": False, "alpha": 0.8708660354719004,
                  "feature_weighting": "TF-IDF"}
    # Small MAP improvement over previous on param search, but in various trials it overfits
    # hyb6y_args = {"topK_P": 1000, "alpha_P": 0.3971126273470195, "normalize_similarity_P": False, "topK": 1000,
    #             "shrink": 0, "similarity": "tversky", "normalize": True, "alpha": 0.6441988996826603,
    #             "feature_weighting": "TF-IDF"}
    hyb6y_args = {"topK_P": 903, "alpha_P": 0.4108657561671193, "normalize_similarity_P": False, "topK": 448, "shrink": 20,
                  "similarity": "tversky", "normalize": True, "alpha": 0.6290871066510789, "feature_weighting": "TF-IDF"}
    rec_list.append(hyb5)
    arg_list.append(hyb5_args)
    name_list.append("hyb5")
    rec_list.append(hyb6x)
    arg_list.append(hyb6x_args)
    name_list.append("hyb6x")
    rec_list.append(hyb6y)
    arg_list.append(hyb6y_args)
    name_list.append("hyb6y")

    tot_args = zip(rec_list, arg_list, name_list)
    pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()-1), maxtasksperchild=1)
    resultList = pool.map(fitRec, tot_args)
    pool.close()
    pool.join()

    for el in resultList:
        if el[1] == "p3alpha":
            p3alpha = el[0]
        elif el[1] == "itemKNNCF":
            itemKNNCF = el[0]
        elif el[1] == "itemKNNCBF":
            itemKNNCBF = el[0]
        elif el[1] == "hyb5":
            hyb5 = el[0]
        elif el[1] == "hyb6x":
            hyb6x = el[0]
        elif el[1] == "hyb6y":
            hyb6y = el[0]

    pureSVD = PureSVDRecommender.PureSVDRecommender(URM_train)
    pureSVD.fit(num_factors=500)

    bpr = SLIM_BPR_Cython(URM_train, recompile_cython=False)
    bpr.fit(**{"topK": 1000, "epochs": 130, "symmetric": False, "sgd_mode": "adagrad", "lambda_i": 1e-05,
               "lambda_j": 0.01, "learning_rate": 0.0001})

    hyb = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, bpr, itemKNNCBF)
    hyb.fit(alpha=0.5)

    #hyb2 = ItemKNNSimilarityHybridRecommender.ItemKNNSimilarityHybridRecommender(URM_train, itemKNNCBF.W_sparse,
    #                                                                             p3alpha.W_sparse)
    #hyb2.fit(topK=1000)
    hyb2 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, itemKNNCBF, p3alpha)
    hyb2.fit(alpha=0.5)

    # Kaggle MAP 0.08667
    hyb3 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb, hyb2)
    hyb3.fit(alpha=0.5)

    #hyb5 = ScoresHybrid3Recommender.ScoresHybrid3Recommender(URM_train, itemKNNCF, userKNNCF, itemKNNCBF)
    #hyb5.fit(beta=0.3)

    # Kaggle MAP 0.09159 hyb6x(v2) + hyb5 (tried alpha 0.4 and 0.6, just small changes, test only as last resort)
    hyb6 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb6x, hyb5)
    hyb6.fit(alpha=0.5)
    #hyb7x = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb6y, hyb5)
    #hyb7x.fit(alpha=0.5)
    hyb7 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb6x, hyb5)
    hyb7.fit(alpha=0.6)


    MAP_p3alpha_per_group = []
    MAP_itemKNNCF_per_group = []
    MAP_itemKNNCBF_per_group = []
    MAP_pureSVD_per_group = []
    MAP_hyb_per_group = []
    MAP_hyb2_per_group = []
    MAP_hyb3_per_group = []
    MAP_hyb5_per_group = []
    MAP_hyb6_per_group = []
    MAP_hyb7_per_group = []
    cutoff = 10
    args = {"block_size": block_size, "profile_length": profile_length, "sorted_users": sorted_users, "cutoff": cutoff,
            "URM_test": URM_test, "hyb": hyb, "hyb2": hyb2, "hyb3": hyb3, "hyb5": hyb5, "hyb6": hyb6, "hyb7": hyb7}

    compute_group_MAP_partial = partial(compute_group_MAP, args)
    for i in range(0, groups):
        el = compute_group_MAP_partial(i)
        MAP_hyb_per_group.append(el[0])
        MAP_hyb2_per_group.append(el[1])
        MAP_hyb3_per_group.append(el[2])
        MAP_hyb5_per_group.append(el[3])
        MAP_hyb6_per_group.append(el[4])
        MAP_hyb7_per_group.append(el[5])


    import matplotlib.pyplot as pyplot

    '''pyplot.plot(MAP_p3alpha_per_group, label="p3alpha")
    pyplot.plot(MAP_itemKNNCF_per_group, label="itemKNNCF")
    pyplot.plot(MAP_itemKNNCBF_per_group, label="itemKNNCBF")
    pyplot.plot(MAP_pureSVD_per_group, label="pureSVD")'''
    pyplot.plot(MAP_hyb_per_group, label="hyb")
    pyplot.plot(MAP_hyb2_per_group, label="hyb2")
    pyplot.plot(MAP_hyb3_per_group, label="hyb3")
    pyplot.plot(MAP_hyb5_per_group, label="hyb5")
    pyplot.plot(MAP_hyb6_per_group, label="hyb6")
    pyplot.plot(MAP_hyb7_per_group, label="hyb7")
    pyplot.ylabel('MAP')
    pyplot.xlabel('User Group')
    pyplot.legend()
    pyplot.show()

    print(l_list)
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)
    print(evaluator_validation.evaluateRecommender(hyb))
    print(evaluator_validation.evaluateRecommender(hyb2))
    print(evaluator_validation.evaluateRecommender(hyb3))
    print(evaluator_validation.evaluateRecommender(hyb5))
    print(evaluator_validation.evaluateRecommender(hyb6))
    print(evaluator_validation.evaluateRecommender(hyb7))
    '''item_list = hyb6.recommend(target_ids, cutoff=10)
    CreateCSV.create_csv(target_ids, item_list, 'Hyb_URM_ICM_tuned')
    item_list = hyb2.recommend(target_ids, cutoff=10)
    CreateCSV.create_csv(target_ids, item_list, 'Hyb2')
    item_list = hyb6.recommend(target_ids, cutoff=10)
    CreateCSV.create_csv(target_ids, item_list, 'Hyb_URM_ICM')'''

    print("--- Execution time: %s seconds ---" % (time.time() - start_time))