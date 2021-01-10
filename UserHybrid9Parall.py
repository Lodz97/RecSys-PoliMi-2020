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
import ScoresHybridSpecialized
import ScoresHybridSpecializedCold
import ScoresHybridSpecializedFusion
import ScoresHybridSpecializedV2Cold, ScoresHybridSpecializedV2Mid, ScoresHybridSpecializedV2Warm
import ScoresHybridSpecializedV2Mid12, ScoresHybridSpecializedV2Warm12
import ScoresHybridSpecializedV3Cold
import ScoresHybridSpecializedV2Fusion
import ScoresHybridSpecializedAdaptive
import ScoresHybridKNNCFKNNCBF
import CreateCSV
from sklearn.preprocessing import normalize
from scipy import sparse as sps
from Utils.PoolWithSubprocess import PoolWithSubprocess
import multiprocessing
from functools import partial
import time


def augment_with_best_recommended_items(urm: sps.csr_matrix, rec, users, cutoff, value=0.5):
    augmented_urm = urm.tolil(copy=True).astype(np.float)
    for user in users:
        recommended_items = rec.recommend(user, cutoff=cutoff)
        for item in recommended_items:
            augmented_urm[user, item] += value

    # Return the augmented urm
    return augmented_urm.tocsr()


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

    if hyb7 is not None:
        results, _ = evaluator_test.evaluateRecommender(hyb7)
        MAP_hyb7_per_group.append(results[cutoff]["MAP"])

    if hyb7 is not None:
        return [MAP_hyb_per_group, MAP_hyb2_per_group, MAP_hyb3_per_group, MAP_hyb5_per_group, MAP_hyb6_per_group,
                MAP_hyb7_per_group]
    else:
        return [MAP_hyb_per_group, MAP_hyb2_per_group, MAP_hyb3_per_group, MAP_hyb5_per_group, MAP_hyb6_per_group]



if __name__ == '__main__':
    start_time = time.time()

    URM_all, user_id_unique, item_id_unique = RecSys2020Reader.load_urm()
    ICM_all = RecSys2020Reader.load_icm_asset()
    target_ids = RecSys2020Reader.load_target()

    #np.random.seed(123412)
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)
    # ICM_train, ICM_test = train_test_holdout(ICM_all, train_perc=0.995)
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)
    #URM_train = URM_all
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

    hyb_warm = ScoresHybridSpecialized.ScoresHybridSpecialized(URM_ICM_train, URM_ICM_train.T)
    hyb_warmV2 = ScoresHybridSpecializedV2Warm12.ScoresHybridSpecializedV2Warm12(URM_ICM_train, URM_ICM_train.T)
    # Warm of Kaggle MAP 0.09466
    hyb_warm_args = {"topK_P": 1000, "alpha_P": 0.587663346034695, "normalize_similarity_P": False, "topK": 1000,
                    "shrink": 1000, "similarity": "cosine", "normalize": True, "alpha": 0.5582200212368523,
                    "feature_weighting": "BM25"}
    hyb_warmV2_args = {"topK_P": 1238, "alpha_P": 0.580501466821829, "normalize_similarity_P": False, "topK": 1043,
                       "shrink": 163, "similarity": "asymmetric", "normalize": False, "alpha": 0.25081946305309705,
                       "feature_weighting": "BM25"}

    hyb_cold = ScoresHybridSpecializedCold.ScoresHybridSpecializedCold(URM_ICM_train, URM_ICM_train.T)
    # Cold of Kaggle MAP 0.09466
    hyb_cold_args = {"topK_P": 1000, "alpha_P": 0.3866334498207009, "normalize_similarity_P": False, "topK": 1000,
                    "shrink": 0, "similarity": "tanimoto", "normalize": False, "alpha": 0.5373872324033048,
                    "feature_weighting": "BM25"}

    rec_list.append(hyb_cold)
    arg_list.append(hyb_cold_args)
    name_list.append("hyb_cold")
    rec_list.append(hyb_warm)
    arg_list.append(hyb_warm_args)
    name_list.append("hyb_warm")
    rec_list.append(hyb_warmV2)
    arg_list.append(hyb_warmV2_args)
    name_list.append("hyb_warmV2")

    hyb5 = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_train, ICM_train)
    hyb5_args = {"topK_P": 520, "alpha_P": 0.35484705634053443, "normalize_similarity_P": False, "topK": 1018,
                       "shrink": 945, "similarity": "cosine", "normalize": False, "alpha": 0.7084453011155973, "feature_weighting": "BM25"}
    # hyb5 best args {"topK_P": 903, "alpha_P": 0.4108657561671193, "normalize_similarity_P": False, "topK": 448,
    #                  "shrink": 5,
    #                  "similarity": "tversky", "normalize": True, "alpha": 0.6290871066510789, "feature_weighting": "TF-IDF"}
    rec_list.append(hyb5)
    arg_list.append(hyb5_args)
    name_list.append("hyb5")

    tot_args = zip(rec_list, arg_list, name_list)
    pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()-1), maxtasksperchild=1)
    resultList = pool.map(fitRec, tot_args)
    pool.close()
    pool.join()

    for el in resultList:
        if el[1] == "hyb_cold":
            hyb_cold = el[0]
        elif el[1] == "hyb_warm":
            hyb_warm = el[0]
        elif el[1] == "hyb_coldV2":
            hyb_coldV2 = el[0]
        elif el[1] == "hyb_midV2":
            hyb_midV2 = el[0]
        elif el[1] == "hyb_warmV2":
            hyb_warmV2 = el[0]
        elif el[1] == "hyb5":
            hyb5 = el[0]
        elif el[1] == "hyb6x":
            hyb6x = el[0]

    # Kaggle MAP 0.09159 hyb6x(v2) + hyb5 (tried alpha 0.4 and 0.6, just small changes, test only as last resort)
    hyb6 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb_cold, hyb5)
    hyb6.fit(alpha=0.5)
    #hyb7x = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb6y, hyb5)
    #hyb7x.fit(alpha=0.5)

    ''' Enrich URM wirh best scores, fail
    URM_train = augment_with_best_recommended_items(URM_train, hyb6, user_id_unique, 1, 0.01)
    URM_ICM_train = sps.vstack([URM_train, ICM_all.T])
    URM_ICM_train = URM_ICM_train.tocsr()
    hyb6z = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_ICM_train, ICM_train)
    hyb6z.fit(**{"topK_P": 717, "alpha_P": 0.4864819897070818, "normalize_similarity_P": False, "topK": 707,
                 "shrink": 0, "similarity": "tversky", "normalize": False, "alpha": 0.8708660354719004,
                 "feature_weighting": "TF-IDF"})'''
    '''hyb5 = ScoresHybridSpecializedAdaptive.ScoresHybridSpecializedAdaptive(URM_ICM_train, URM_ICM_train.T)
    hyb5.fit(**{"topK_P_C": 1255, "alpha_P_C": 0.6476785243900337, "beta_P_C": 0.23324299829959988,
                 "normalize_similarity_P_C": False, "topK_C": 127, "shrink_C": 122, "similarity_C": "tversky",
                 "normalize_C": False, "feature_weighting_C": "none", "topK_P": 523, "alpha_P": 1.0956220670878771,
                 "beta_P": 0.30392479983094206, "normalize_similarity_P": False, "topK": 486, "shrink": 950,
                 "similarity": "tanimoto", "normalize": True, "feature_weighting": "BM25", "threshold": 9})'''

    #hyb7 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb6x, hyb5)
    #hyb7.fit(alpha=0.5)
    hyb7x = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb_warm, hyb5)
    hyb7x.fit(alpha=0.5)
    # Kaggle MAP 0.09466
    hyb = ScoresHybridSpecializedFusion.ScoresHybridSpecializedFusion(URM_train, hyb6, hyb7x, 6)

    hyb2 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb6, hyb7x)
    hyb2.fit(alpha=0.5)
    # Kaggle MAP 0.9483
    hyb3x = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb_warmV2, hyb5)
    hyb3x.fit(alpha=0.5)
    #hyb2 = ScoresHybridKNNCFKNNCBF.ScoresHybridKNNCFKNNCBF(URM_ICM_train, URM_ICM_train.T)
    #hyb2.fit(**{"topK_CF": 488, "shrink_CF": 1500, "similarity_CF": "tversky", "normalize_CF": True, "topK": 1500,
    #            "shrink": 1500, "similarity": "asymmetric", "normalize": False, "alpha": 0.23233349150222427,
    #            "feature_weighting": "BM25"})
    #hyb2 = ScoresHybridSpecializedV2Mid.ScoresHybridSpecializedV2Mid(URM_ICM_train, URM_ICM_train.T)
    #hyb2.fit(**{"topK_P": 516, "alpha_P": 0.4753488773601332, "normalize_similarity_P": False, "topK": 258, "shrink": 136,
    #         "similarity": "asymmetric", "normalize": False, "alpha": 0.48907705969537585, "feature_weighting": "BM25"})
    #hyb2 = ScoresHybridSpecializedV3Cold.ScoresHybridSpecializedV3Cold(URM_ICM_train, URM_ICM_train.T)
    #hyb2.fit(**{"topK_P": 536, "alpha_P": 0.041143148090634137, "normalize_similarity_P": False, "topK": 1259,
    #            "shrink": 1310, "similarity": "tversky", "normalize": True, "alpha": 0.30170377380902696, "feature_weighting": "TF-IDF"})
    # Param tune with different random pred
    #hyb2 = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_ICM_train, URM_ICM_train.T)
    #hyb2.fit(**{"topK_P": 348, "alpha_P": 0.6442245707882884, "normalize_similarity_P": False, "topK": 91, "shrink": 1418,
    # "similarity": "cosine", "normalize": False, "alpha": 0.6572797618019163, "feature_weighting": "BM25"})
    # Using hyb5 param with URM_ICM_train, best in group 1
    #hyb2 = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_ICM_train, URM_ICM_train.T)
    #hyb2.fit(**{"topK_P": 903, "alpha_P": 0.4108657561671193, "normalize_similarity_P": False, "topK": 448,
    #             "shrink": 5,
    #             "similarity": "tversky", "normalize": True, "alpha": 0.6290871066510789, "feature_weighting": "TF-IDF"})
    #hyb7 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb2, hyb3)
    #hyb7.fit(alpha=0.5)
    # Kaggle MAP 0.09487, thereshold 6.1
    # Kaggle MAP 0.09509, thereshold 5.9
    hyb3 = ScoresHybridSpecializedFusion.ScoresHybridSpecializedFusion(URM_ICM_train, hyb, hyb3x, 5.9)
    hyb7 = ScoresHybridSpecializedFusion.ScoresHybridSpecializedFusion(URM_ICM_train, hyb2, hyb3x, 5.9)

    #hyb7 = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_ICM_train, URM_ICM_train.T)
    #hyb7.fit(**{"topK_P": 516, "alpha_P": 0.4753488773601332, "normalize_similarity_P": False, "topK": 258, "shrink": 136,
    #          "similarity": "asymmetric", "normalize": False, "alpha": 0.48907705969537585, "feature_weighting": "BM25"})
    # Like hyb3x, but with hyb7 instead of hyb5
    #hyb2 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb3, hyb7)
    #hyb2.fit(alpha=0.5)
    #hyb2 = ScoresHybridSpecializedFusion.ScoresHybridSpecializedFusion(URM_ICM_train, hyb3, hyb2x, 5.9)
    #hyb7 = ScoresHybridSpecializedFusion.ScoresHybridSpecializedFusion(URM_ICM_train, hyb2, hyb3x, 3.1)
    #hyb7 = ScoresHybridSpecializedFusion.ScoresHybridSpecializedFusion(URM_ICM_train, hyb3, hyb7y, 5.9)



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

    pool = PoolWithSubprocess(processes=multiprocessing.cpu_count()-1, maxtasksperchild=1)
    compute_group_MAP_partial = partial(compute_group_MAP, args)
    resultList = pool.map(compute_group_MAP_partial, range(0, groups))
    pool.close()
    pool.join()
    for el in resultList:
        MAP_hyb_per_group.append(el[0])
        MAP_hyb2_per_group.append(el[1])
        MAP_hyb3_per_group.append(el[2])
        MAP_hyb5_per_group.append(el[3])
        MAP_hyb6_per_group.append(el[4])
        if hyb7 is not None:
            MAP_hyb7_per_group.append(el[5])

    # Needed because of memory error
    '''for group_id in range(0, groups):
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

        results, _ = evaluator_test.evaluateRecommender(hyb7)
        MAP_hyb7_per_group.append(results[cutoff]["MAP"])'''


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
    if hyb7 is not None:
        pyplot.plot(MAP_hyb7_per_group, label="hyb7")
    pyplot.ylabel('MAP')
    pyplot.xlabel('User Group')
    pyplot.legend()
    pyplot.show()

    print(l_list)
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)
    pool = PoolWithSubprocess(processes=multiprocessing.cpu_count()-1, maxtasksperchild=1)
    if hyb7 is not None:
        hyb_list = [hyb, hyb2, hyb3, hyb5, hyb6, hyb7]
    else:
        hyb_list = [hyb, hyb2, hyb3, hyb5, hyb6]
    resultList = pool.map(evaluator_validation.evaluateRecommender, hyb_list)
    pool.close()
    pool.join()
    for el in resultList:
        print(el)
    '''item_list = hyb3.recommend(target_ids, cutoff=10)
    CreateCSV.create_csv(target_ids, item_list, 'Hyb_URM_ICM_cold_warm_V2_more_mix_newhyb5')
    item_list = hyb2.recommend(target_ids, cutoff=10)
    CreateCSV.create_csv(target_ids, item_list, 'Hyb2')
    item_list = hyb6.recommend(target_ids, cutoff=10)
    CreateCSV.create_csv(target_ids, item_list, 'Hyb_URM_ICM')'''

    print("--- Execution time: %s seconds ---" % (time.time() - start_time))