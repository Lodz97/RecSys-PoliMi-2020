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
import ItemKNNScoresHybridRecommender
import CreateCSV

# https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi/blob/master/Practice%2009%20-%20SLIM%20BPR.ipynb
# https://github.com/nicolo-felicioni/recsys-polimi-2019/tree/master/Hybrid


if __name__ == '__main__':
    URM_all, user_id_unique, item_id_unique = RecSys2020Reader.load_urm()
    ICM_all = RecSys2020Reader.load_icm_asset()
    target_ids = RecSys2020Reader.load_target()

    #np.random.seed(12341)
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.97)
    ICM_train, ICM_test = train_test_holdout(ICM_all, train_perc=0.97)
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)

    earlystopping_keywargs = {"validation_every_n": 10,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator_validation,
                              "lower_validations_allowed": 5,
                              "validation_metric": "MAP",
                              }

    profile_length = np.ediff1d(URM_train.indptr)
    block_size = int(len(profile_length) * 0.1)
    sorted_users = np.argsort(profile_length)
    for group_id in range(0, 11):
        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, len(profile_length))

        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = profile_length[users_in_group]

        print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                      users_in_group_p_len.mean(),
                                                                      users_in_group_p_len.min(),
                                                                      users_in_group_p_len.max()))

    from Base.NonPersonalizedRecommender import TopPop
    topPop = TopPop(URM_train)
    topPop.fit()


    p3alpha = P3alphaRecommender.P3alphaRecommender(URM_train)
    p3alpha.fit(**{"topK": 991, "alpha": 0.4705816992313091, "normalize_similarity": False})

    itemKNNCF = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    itemKNNCF.fit(**{"topK": 1000, "shrink": 732, "similarity": "cosine", "normalize": True,
                     "feature_weighting": "TF-IDF"})

    userKNNCF = UserKNNCFRecommender.UserKNNCFRecommender(URM_train)
    userKNNCF.fit(**{"topK": 131, "shrink": 2, "similarity": "cosine", "normalize": True})

    itemKNNCBF = ItemKNNCBFRecommender.ItemKNNCBFRecommender(URM_train, ICM_train)
    itemKNNCBF.fit(topK=700, shrink=200, similarity='jaccard', normalize=True, feature_weighting = "TF-IDF")

    slim = SLIMElasticNetRecommender.MultiThreadSLIM_ElasticNet(URM_train)
    slim.fit(**{"topK": 87, "l1_ratio": 0.000010002224923703737})
    #slim.load_model('C:\\Users\\Giacomo\\PycharmProjects\\RecSys-PoliMi-2020\\ParameterTuning\\ParamResultsExperiments\\SKOPT_SLIMElasticNet_Nov28_19-42-57\\',
    #                'SLIMElasticNetRecommender_best_model.zip')

    #bpr = SLIM_BPR_Cython(URM_train, recompile_cython=False)
    #bpr.fit(**{"topK": 1000, "epochs": 130, "symmetric": False, "sgd_mode": "adagrad", "lambda_i": 1e-05,
    #                   "lambda_j": 0.01, "learning_rate": 0.0001})

    pureSVD = PureSVDRecommender.PureSVDRecommender(URM_train)
    pureSVD.fit()

    hyb = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, itemKNNCBF, p3alpha)
    hyb.fit(alpha=0.5)

    # Kaggle MAP 0.081
    hyb2 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb, slim)
    hyb2.fit(alpha=0.5)

    hyb3 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb, hyb2)
    hyb3.fit(alpha=0.5)

    MAP_p3alpha_per_group = []
    MAP_itemKNNCF_per_group = []
    MAP_userKNNCF_per_group = []
    MAP_itemKNNCBF_per_group = []
    MAP_slim_per_group = []
    MAP_pureSVD_per_group = []
    #MAP_bpr_per_group = []
    MAP_topPop_per_group = []
    MAP_hyb_per_group = []
    MAP_hyb2_per_group = []
    MAP_hyb3_per_group = []
    cutoff = 10
    l_list = []

    for group_id in range(0, 11):
        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, len(profile_length))

        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = profile_length[users_in_group]
        l_list.append(len(users_in_group))

        print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                      users_in_group_p_len.mean(),
                                                                      users_in_group_p_len.min(),
                                                                      users_in_group_p_len.max()))

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group = sorted_users[users_not_in_group_flag]

        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)

        results, _ = evaluator_test.evaluateRecommender(p3alpha)
        MAP_p3alpha_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(itemKNNCF)
        MAP_itemKNNCF_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(userKNNCF)
        MAP_userKNNCF_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(itemKNNCBF)
        MAP_itemKNNCBF_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(slim)
        MAP_slim_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(pureSVD)
        MAP_pureSVD_per_group.append(results[cutoff]["MAP"])

        #results, _ = evaluator_test.evaluateRecommender(bpr)
        #MAP_bpr_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(topPop)
        MAP_topPop_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(hyb)
        MAP_hyb_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(hyb2)
        MAP_hyb2_per_group.append(results[cutoff]["MAP"])

        results, _ = evaluator_test.evaluateRecommender(hyb3)
        MAP_hyb3_per_group.append(results[cutoff]["MAP"])

    import matplotlib.pyplot as pyplot

    pyplot.plot(MAP_p3alpha_per_group, label="p3alpha")
    pyplot.plot(MAP_itemKNNCF_per_group, label="itemKNNCF")
    pyplot.plot(MAP_userKNNCF_per_group, label="userKNNCF")
    pyplot.plot(MAP_itemKNNCBF_per_group, label="itemKNNCBF")
    pyplot.plot(MAP_slim_per_group, label="slim")
    pyplot.plot(MAP_pureSVD_per_group, label="pureSVD")
    #pyplot.plot(MAP_bpr_per_group, label="bpr")
    pyplot.plot(MAP_topPop_per_group, label="topPop")
    pyplot.plot(MAP_hyb_per_group, label="hyb")
    pyplot.plot(MAP_hyb2_per_group, label="hyb2")
    pyplot.plot(MAP_hyb3_per_group, label="hyb3")
    pyplot.ylabel('MAP')
    pyplot.xlabel('User Group')
    pyplot.legend()
    pyplot.show()

    print(l_list)
    print([np.mean(MAP_itemKNNCF_per_group), np.mean(MAP_userKNNCF_per_group), np.mean(MAP_hyb_per_group)])
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)
    print(evaluator_validation.evaluateRecommender(userKNNCF))
    print(evaluator_validation.evaluateRecommender(hyb))
    print(evaluator_validation.evaluateRecommender(hyb2))
    print(evaluator_validation.evaluateRecommender(hyb3))
    item_list = hyb.recommend(target_ids, cutoff=10)
    CreateCSV.create_csv(target_ids, item_list, 'Hyb1')
    item_list = hyb2.recommend(target_ids, cutoff=10)
    CreateCSV.create_csv(target_ids, item_list, 'Hyb2')
    item_list = hyb3.recommend(target_ids, cutoff=10)
    CreateCSV.create_csv(target_ids, item_list, 'Hyb3')