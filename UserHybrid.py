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

# https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi/blob/master/Practice%2009%20-%20SLIM%20BPR.ipynb
# https://github.com/nicolo-felicioni/recsys-polimi-2019/tree/master/Hybrid


if __name__ == '__main__':
    URM_all, user_id_unique, item_id_unique = RecSys2020Reader.load_urm()
    ICM_all = RecSys2020Reader.load_icm_asset()
    target_ids = RecSys2020Reader.load_target()

    #np.random.seed(12341)
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.97)
    ICM_train, ICM_test = train_test_holdout(ICM_all, train_perc=0.995)
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

    #p3beta = RP3betaRecommender.RP3betaRecommender(URM_train)
    #p3beta.fit(**{"topK": 1000, "alpha": 2.0, "beta": 0.0, "normalize_similarity": True})

    p3alpha = P3alphaRecommender.P3alphaRecommender(URM_train)
    p3alpha.fit(**{"topK": 991, "alpha": 0.4705816992313091, "normalize_similarity": False})

    itemKNNCF = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    itemKNNCF.fit(**{"topK": 1000, "shrink": 732, "similarity": "cosine", "normalize": True,
                     "feature_weighting": "TF-IDF"})

    userKNNCF = UserKNNCFRecommender.UserKNNCFRecommender(URM_train)
    userKNNCF.fit(**{"topK": 131, "shrink": 2, "similarity": "cosine", "normalize": True})

    itemKNNCBF = ItemKNNCBFRecommender.ItemKNNCBFRecommender(URM_train, ICM_train)
    itemKNNCBF.fit(topK=700, shrink=200, similarity='jaccard', normalize=True, feature_weighting = "TF-IDF")

    #cfw = CFW_D_Similarity_Linalg.CFW_D_Similarity_Linalg(URM_train, ICM_train, itemKNNCF.W_sparse)
    #cfw.fit(show_max_performance = False, logFile = None, loss_tolerance = 1e-6,
    #        iteration_limit = 50000, damp_coeff=0.5, topK = 300, add_zeros_quota = 0.0, normalize_similarity = False)

    #slim = SLIMElasticNetRecommender.MultiThreadSLIM_ElasticNet(URM_train)
    #slim.load_model('SavedModels\\', 'SLIM.zip')
    #slim.fit(**{"topK": 87, "l1_ratio": 0.000010002224923703737})
    #slim.save_model('SavedModels\\', 'SLIM')
    #slim.load_model('C:\\Users\\Giacomo\\PycharmProjects\\RecSys-PoliMi-2020\\ParameterTuning\\ParamResultsExperiments\\SKOPT_SLIMElasticNet_Nov28_19-42-57\\',
    #                'SLIMElasticNetRecommender_best_model.zip')

    #bpr = SLIM_BPR_Cython(URM_train, recompile_cython=False)
    #bpr.fit(**{"topK": 1000, "epochs": 130, "symmetric": False, "sgd_mode": "adagrad", "lambda_i": 1e-05,
    #                   "lambda_j": 0.01, "learning_rate": 0.0001})

    #ease = EASE_R_Recommender.EASE_R_Recommender(URM_train)
    #ease.fit(topK=None, l2_norm = 3 * 1e3, normalize_matrix = False, verbose = True)
    #ease.load_model('SavedModels\\', 'Ease.zip')

    pureSVD = PureSVDRecommender.PureSVDRecommender(URM_train)
    pureSVD.fit()

    #hyb0 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, p3alpha, userKNNCF)
    #hyb0.fit(alpha=0.5)

    hyb = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, p3alpha, itemKNNCBF)
    hyb.fit(alpha=0.5)

    # Kaggle MAP 0.084
    hyb2 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, p3alpha, itemKNNCF)
    hyb2.fit(alpha=0.5)

    # Kaggle MAP 0.08667
    hyb3 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb, hyb2)
    hyb3.fit(alpha=0.5)

    #hyb4 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, itemKNNCBF, p3alpha)
    #hyb4.fit(alpha=0.5)

    #hyb5 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb4, hyb3)
    #hyb5.fit(alpha=0.35)

    #hyb5 = ScoresHybrid3Recommender.ScoresHybrid3Recommender(URM_train, itemKNNCF, userKNNCF, itemKNNCBF)
    #hyb5.fit(beta=0.3)

    hyb5 = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_train, ICM_train)
    hyb6 = ScoresHybridP3alphaKNNCBF.ScoresHybridP3alphaKNNCBF(URM_train, ICM_train)
    #hyb5.fit(**{"topK_P": 316, "alpha_P": 0.682309490338903, "normalize_similarity_P": False, "topK": 760, "shrink": 33,
    #           "similarity": "asymmetric", "normalize": True, "feature_weighting": "BM25"})
    #hyb5.fit(**{"topK_P": 715, "alpha_P": 0.47489080414690943, "normalize_similarity_P": True, "topK": 994, "shrink": 179,
    #           "similarity": "tanimoto", "normalize": True, "alpha": 0.5666004103014026, "feature_weighting": "TF-IDF"})
    #hyb5.fit(**{"topK_P": 531, "alpha_P": 0.5783052607747065, "normalize_similarity_P": False, "topK": 162, "shrink": 723,
    #            "similarity": "tversky", "normalize": True, "alpha": 0.5444830213942248, "feature_weighting": "TF-IDF"})
    # Kaggle MAP 0.08856
    hyb5.fit(**{"topK_P": 903, "alpha_P": 0.4108657561671193, "normalize_similarity_P": True, "topK": 448, "shrink": 20,
                "similarity": "tversky", "normalize": True, "alpha": 0.6290871066510789, "feature_weighting": "TF-IDF"})
    #hyb6.fit(**{"topK_P": 1000, "alpha_P": 0.5432601071314623, "normalize_similarity_P": True, "topK": 620, "shrink": 0,
    #           "similarity": "tversky", "normalize": False, "alpha": 0.5707347522847057, "feature_weighting": "BM25"})
    hyb6.fit(**{"topK_P": 817, "alpha_P": 0.4055117493348105, "normalize_similarity_P": False, "topK": 573, "shrink": 870,
                "similarity": "cosine", "normalize": False, "alpha": 0.806611463440025, "feature_weighting": "BM25"})




    '''MAP_p3alpha_per_group = []
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
    MAP_hyb5_per_group = []
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

        #results, _ = evaluator_test.evaluateRecommender(slim)
        #MAP_slim_per_group.append(results[cutoff]["MAP"])f

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

        results, _ = evaluator_test.evaluateRecommender(hyb5)
        MAP_hyb5_per_group.append(results[cutoff]["MAP"])

    import matplotlib.pyplot as pyplot

    pyplot.plot(MAP_p3alpha_per_group, label="p3alpha")
    pyplot.plot(MAP_itemKNNCF_per_group, label="itemKNNCF")
    pyplot.plot(MAP_userKNNCF_per_group, label="userKNNCF")
    pyplot.plot(MAP_itemKNNCBF_per_group, label="itemKNNCBF")
    #pyplot.plot(MAP_slim_per_group, label="slim")
    pyplot.plot(MAP_pureSVD_per_group, label="pureSVD")
    #pyplot.plot(MAP_bpr_per_group, label="bpr")
    pyplot.plot(MAP_topPop_per_group, label="topPop")
    pyplot.plot(MAP_hyb_per_group, label="hyb")
    pyplot.plot(MAP_hyb2_per_group, label="hyb2")
    pyplot.plot(MAP_hyb3_per_group, label="hyb3")
    pyplot.plot(MAP_hyb5_per_group, label="hyb5")
    pyplot.ylabel('MAP')
    pyplot.xlabel('User Group')
    pyplot.legend()
    pyplot.show()

    print(l_list)
    print([np.mean(MAP_itemKNNCF_per_group), np.mean(MAP_userKNNCF_per_group), np.mean(MAP_hyb_per_group)])'''
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)
    #print(evaluator_validation.evaluateRecommender(cfw))
    #print(evaluator_validation.evaluateRecommender(hyb0))
    print(evaluator_validation.evaluateRecommender(hyb))
    print(evaluator_validation.evaluateRecommender(hyb2))
    print(evaluator_validation.evaluateRecommender(hyb3))
    print(evaluator_validation.evaluateRecommender(hyb5))
    print(evaluator_validation.evaluateRecommender(hyb6))
    item_list = hyb.recommend(target_ids, cutoff=10)
    CreateCSV.create_csv(target_ids, item_list, 'Hyb')
    item_list = hyb2.recommend(target_ids, cutoff=10)
    CreateCSV.create_csv(target_ids, item_list, 'Hyb2')
    item_list = hyb3.recommend(target_ids, cutoff=10)
    CreateCSV.create_csv(target_ids, item_list, 'Hyb3')
