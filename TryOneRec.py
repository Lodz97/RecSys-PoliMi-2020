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

    #np.random.seed(1234)
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.97)
    ICM_train, ICM_test = train_test_holdout(ICM_all, train_perc=0.8)
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)

    earlystopping_keywargs = {"validation_every_n": 10,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator_validation,
                              "lower_validations_allowed": 5,
                              "validation_metric": "MAP",
                              }

    # MAP 0.07 Kaggle "topK": 131, "shrink": 2, "similarity": "cosine", "normalize": true}

    recommender = UserKNNCFRecommender.UserKNNCFRecommender(URM_train)
    recommender.fit(**{"topK": 131, "shrink": 2, "similarity": "cosine", "normalize": True})


    print(evaluator_validation.evaluateRecommender(recommender))

    item_list = recommender.recommend(target_ids, cutoff=10)
    CreateCSV.create_csv(target_ids, item_list, 'MyRec')
