import pandas as pd
import scipy.sparse as sps
import numpy as np

urm_path = "C:/Users/Giacomo/PycharmProjects/RecSys-PoliMi-2020/Data/data_train.csv"
icm_asset_path = "C:/Users/Giacomo/PycharmProjects/RecSys-PoliMi-2020/Data/data_ICM_title_abstract.csv"
target_path = "C:/Users/Giacomo/PycharmProjects/RecSys-PoliMi-2020/Data/data_target_users_test.csv"




def load_urm():
    df_original = pd.read_csv(filepath_or_buffer=urm_path, sep=',', header=0,
                              dtype={'row': int, 'col': int, 'rating': float})

    df_original.columns = ['user', 'item', 'rating']

    user_id_list = df_original['user'].values
    item_id_list = df_original['item'].values
    rating_id_list = df_original['rating'].values

    user_id_unique = np.unique(user_id_list)
    item_id_unique = np.unique(item_id_list)

    csr_matrix = sps.csr_matrix((rating_id_list, (user_id_list, item_id_list)))
    csr_matrix = csr_matrix.astype(dtype=np.float)
    # print("DataReader:")
    # print("\tLoading the URM:")
    # print("\t\tURM size:" + str(csr_matrix.shape))
    # print("\t\tURM unique users:" + str(user_id_unique.size))
    # print("\t\tURM unique items:" + str(item_id_unique.size))
    # print("\tURM loaded.")

    return csr_matrix, user_id_unique, item_id_unique

def load_target():
    df_original = pd.read_csv(filepath_or_buffer=target_path, sep=',', header=0,
                              dtype={'user': int})

    df_original.columns = ['user']

    user_id_list = df_original['user'].values

    user_id_unique = np.unique(user_id_list)

    # print("DataReader:")
    # print("\tLoading the target users:")
    # print("\t\tTarget size:" + str(user_id_unique.shape))
    # print("\tTarget users loaded.")

    return user_id_unique

def load_icm_asset():
    df_original = pd.read_csv(filepath_or_buffer=icm_asset_path, sep=',', header=0,
                              dtype={'item': int, 'feature': int, 'data': float})

    df_original.columns = ['item', 'feature', 'data']

    item_id_list = df_original['item'].values
    feature_id_list = df_original['feature'].values
    data_id_list = df_original['data'].values * 2

    csr_matrix = sps.csr_matrix((data_id_list, (item_id_list, feature_id_list)))

    # print("DataReader:")
    # print("\tLoading the asset ICM: " + icm_asset_path)
    # print("\t\tAsset ICM size:" + str(csr_matrix.shape))
    # print("\tAsset ICM loaded.")

    return csr_matrix

def load_urm_icm():
    urm, _, _ = load_urm()
    icm = load_icm_asset()
    urm_icm = sps.vstack([urm, icm.T])
    urm_icm = urm_icm.tocsr()

    return urm_icm
