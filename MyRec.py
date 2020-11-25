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
import CreateCSV

# https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi/blob/master/Practice%2009%20-%20SLIM%20BPR.ipynb
# https://github.com/nicolo-felicioni/recsys-polimi-2019/tree/master/Hybrid


if __name__ == '__main__':
    URM_all, user_id_unique, item_id_unique = RecSys2020Reader.load_urm()
    ICM_all = RecSys2020Reader.load_icm_asset()
    target_ids = RecSys2020Reader.load_target()

    item_popularity = np.ediff1d(URM_all.tocsc().indptr)
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
    pyplot.show()

    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)
    ICM_train, ICM_test = train_test_holdout(ICM_all, train_perc=0.8)
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)

    earlystopping_keywargs = {"validation_every_n": 10,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator_validation,
                              "lower_validations_allowed": 5,
                              "validation_metric": "MAP",
                              }

    # MAP 0.057, kaggle MAP 0.054
    # recommender = SLIM_BPR_Cython(URM_train, recompile_cython=False)
    # recommender.fit(**{"topK": 665, "epochs": 2000, "symmetric": False, "sgd_mode": "adagrad", "lambda_i": 0.01,
    #                    "lambda_j": 1e-05, "learning_rate": 0.0001}, **earlystopping_keywargs)

    # MAP 0.052
    # recommender = P3alphaRecommender.P3alphaRecommender(URM_train)
    # recommender.fit(**{"topK": 998, "alpha": 0.08643815887780361, "normalize_similarity": False})

    # Bad MAP 0.035
    # recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    # recommender.fit(**{"topK": 1000, "alpha": 2.0, "beta": 0.0, "normalize_similarity": True})

    # Bad MAP 0.0092
    # recommender = MatrixFactorization_Cython.MatrixFactorization_BPR_Cython(URM_train, recompile_cython=False)
    # recommender.fit(**{"sgd_mode": "adam", "epochs": 520, "num_factors": 89, "batch_size": 1,
    #                   "positive_reg": 0.000496729692408822, "negative_reg": 2.0095466515941112e-05,
    #                   "learning_rate": 0.00036482393449734817})

    # Bad MAP 0.0092??
    # recommender = MatrixFactorization_Cython.MatrixFactorization_FunkSVD_Cython(URM_train, recompile_cython=False)
    # recommender.fit(**{"sgd_mode": "adagrad", "epochs": 500, "use_bias": True, "batch_size": 16, "num_factors": 126,
    #                   "item_reg": 0.0013469014228160933, "user_reg": 0.001684005069940341,
    #                   "learning_rate": 0.04299371225427124, "negative_interactions_quota": 0.49217247466651953})

    # Bad MAP 0.016 (40 epochs, 126 factors)
    # recommender = MatrixFactorization_Cython.MatrixFactorization_AsySVD_Cython(URM_train, recompile_cython=False)
    # recommender.fit(**{"sgd_mode": "adagrad", "epochs": 200, "use_bias": True, "batch_size": 1, "num_factors": 126,
    #                   "item_reg": 0.0013469014228160933, "user_reg": 0.001684005069940341,
    #                   "learning_rate": 0.04299371225427124, "negative_interactions_quota": 0.49217247466651953},
    #                **earlystopping_keywargs)

    # Bad MAP 0.0001
    # recommender = MF_MSE_PyTorch.MF_MSE_PyTorch(URM_train)
    # recommender.fit(epochs=100, batch_size = 256, num_factors=100, learning_rate = 0.001, use_cuda = True,
    #                 **earlystopping_keywargs)

    # Bad MAP 0.029
    # recommender = IALSRecommender.IALSRecommender(URM_train)
    # recommender.fit(epochs = 100, num_factors = 50, confidence_scaling = "log", alpha = 1.0,
    #                 epsilon = 1.0, reg = 1e-3, init_mean=0.0, init_std=0.1, **earlystopping_keywargs)

    # Not working
    # recommender = NMFRecommender.NMFRecommender(URM_train)
    # recommender.fit(num_factors=100, l1_ratio = 0.5, solver = "multiplicative_update", init_type = "random",
    #                 beta_loss = "frobenius")

    # Bad MAP 0.032 (1000 factors)
    # recommender = PureSVDRecommender.PureSVDRecommender(URM_train)
    # recommender.fit(num_factors=2000)

    # MAP 0.026 (topK=700, shrink=300, similarity='jaccard', normalize=True, feature_weighting = "TF-IDF")
    # recommender = ItemKNNCBFRecommender.ItemKNNCBFRecommender(URM_train, ICM_train)
    # recommender.fit(topK=700, shrink=200, similarity='jaccard', normalize=True, feature_weighting = "TF-IDF")

    # MAP 0.0563 (**{"topK": 1000, "shrink": 732, "similarity": "cosine", "normalize": True,
    #               "feature_weighting": "TF-IDF"})
    # recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    # recommender.fit(**{"topK": 1000, "shrink": 732, "similarity": "cosine", "normalize": True,
    #                    "feature_weighting": "TF-IDF"})

    # MAP 0.058 (**{"topK": 305, "shrink": 0, "similarity": "cosine", "normalize": True,
    #               "feature_weighting": "TF-IDF"})
    recommender = UserKNNCFRecommender.UserKNNCFRecommender(URM_train)
    recommender.fit(**{"topK": 305, "shrink": 0, "similarity": "cosine", "normalize": True,
                       "feature_weighting": "TF-IDF"})

    print(evaluator_validation.evaluateRecommender(recommender))

    item_list = recommender.recommend(target_ids, cutoff=10)
    CreateCSV.create_csv(target_ids, item_list, 'MyRec')
