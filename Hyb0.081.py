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

    np.random.seed(12341)
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.97)
    ICM_train, ICM_test = train_test_holdout(ICM_all, train_perc=0.97)
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)

    itemKNNCF = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    itemKNNCF.fit(**{"topK": 1000, "shrink": 732, "similarity": "cosine", "normalize": True,
                     "feature_weighting": "TF-IDF"})

    userKNNCF = UserKNNCFRecommender.UserKNNCFRecommender(URM_train)
    userKNNCF.fit(**{"topK": 131, "shrink": 2, "similarity": "cosine", "normalize": True})

    itemKNNCBF = ItemKNNCBFRecommender.ItemKNNCBFRecommender(URM_train, ICM_train)
    itemKNNCBF.fit(topK=700, shrink=200, similarity='jaccard', normalize=True, feature_weighting = "TF-IDF")

    hyb = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, itemKNNCBF, userKNNCF)
    hyb.fit(alpha=0.5)

    # Kaggle MAP 0.081
    hyb2 = ItemKNNScoresHybridRecommender.ItemKNNScoresHybridRecommender(URM_train, hyb, itemKNNCF)
    hyb2.fit(alpha=0.5)


    print(evaluator_validation.evaluateRecommender(userKNNCF))
    print(evaluator_validation.evaluateRecommender(hyb))
    print(evaluator_validation.evaluateRecommender(hyb2))

    item_list = hyb.recommend(target_ids, cutoff=10)
    CreateCSV.create_csv(target_ids, item_list, 'Hyb_User_Item_KNNCF')