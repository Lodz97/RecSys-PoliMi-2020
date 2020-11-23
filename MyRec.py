from Data_manager.RecSys2020 import RecSys2020Reader
from Notebooks_utils.data_splitter import train_test_holdout
import matplotlib.pyplot as pyplot
import numpy as np
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Base.Evaluation.Evaluator import EvaluatorHoldout
import os
from datetime import datetime

# https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi/blob/master/Practice%2009%20-%20SLIM%20BPR.ipynb
# https://github.com/nicolo-felicioni/recsys-polimi-2019/tree/master/Hybrid


res_dir = 'Results/MyRec'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)


def create_csv(target_ids, results, results_dir=res_dir):

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        f.write('user_id,item_list\n')

        for target_id, result in zip(target_ids, results):
            f.write(str(target_id) + ', ' + ' '.join(map(str, result)) + '\n')



if __name__ == '__main__':
    URM_all, user_id_unique, item_id_unique = RecSys2020Reader.load_urm()
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

    recommender = SLIM_BPR_Cython(URM_train, recompile_cython=False)
    recommender.fit(epochs=1000, batch_size=100, sgd_mode='adagrad', learning_rate=1e-4, positive_threshold_BPR=1)

    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)
    print(evaluator_validation.evaluateRecommender(recommender))

    item_list = recommender.recommend(target_ids, cutoff=10)
    create_csv(target_ids, item_list)

