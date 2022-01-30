import os
import shutil
import numpy as np
import pandas as pd


def create_training_sets_from_source(name: str, source_data: pd.DataFrame, column_names: list, train_share=0.8,
                                     share_entries=None):
    source_data.columns = column_names
    source_data.sort_values('Time', inplace=True)

    if share_entries is not None:
        n_entries = len(source_data)
        source_data = source_data.iloc[:int(n_entries * share_entries)]

    entries_to_consider = source_data
    user_counter = entries_to_consider.groupby('User').count()['Time']
    item_counter = entries_to_consider.groupby('Item').count()['Time']
    users_to_drop = list(user_counter[user_counter < min_n_interactions[name]].index)
    items_to_drop = list(item_counter[item_counter < min_n_interactions[name]].index)
    while len(users_to_drop) > 0 or len(items_to_drop) > 0:
        entries_to_consider.drop(entries_to_consider[entries_to_consider.User.isin(users_to_drop)].index, inplace=True)
        entries_to_consider.drop(entries_to_consider[entries_to_consider.Item.isin(items_to_drop)].index, inplace=True)
        user_counter = entries_to_consider.groupby('User').count()['Time']
        item_counter = entries_to_consider.groupby('Item').count()['Time']
        users_to_drop = list(user_counter[user_counter < min_n_interactions[name]].index)
        items_to_drop = list(item_counter[item_counter < min_n_interactions[name]].index)

    entries_to_consider.sort_values(['User', 'Time'], inplace=True)
    users_to_consider = list(set(entries_to_consider.User))
    stats = get_statistics_from_dataset(entries_to_consider)

    entries_to_consider = entries_to_consider[['User', 'Item']]
    entries_to_consider['IsRated'] = [1] * len(entries_to_consider)

    train_indices = []
    test_indices = []

    for u in users_to_consider:
        user_data = entries_to_consider[entries_to_consider.User == u]
        n_entries = len(user_data)
        train_size = int(np.round(n_entries * train_share))
        train_indices += list(user_data.iloc[:train_size].index)
        test_indices += list(user_data.iloc[train_size:].index)

    train_data = entries_to_consider.loc[train_indices]
    test_data = entries_to_consider.loc[test_indices]
    if share_entries is None:
        train_data.to_csv(os.path.join(reproduced_data_folder_name, name, 'test', 'train.txt'), sep=' ', header=None,
                          index=None)
        test_data.to_csv(os.path.join(reproduced_data_folder_name, name, 'test', 'test.txt'), sep=' ', header=None,
                         index=None)
    else:
        train_data.to_csv(os.path.join(reproduced_data_folder_name, '%s_small' % name, 'test', 'train.txt'), sep=' ',
                          header=None, index=None)
        test_data.to_csv(os.path.join(reproduced_data_folder_name, '%s_small' % name, 'test', 'test.txt'), sep=' ',
                         header=None, index=None)

    return train_data, test_data, stats


def get_statistics_from_dataset(data: pd.DataFrame, print_statistics=True):
    users_to_consider = list(set(data.User))
    n_user = len(users_to_consider)
    n_items = len(set(data.Item))
    avg_act_user = np.mean(data.groupby('User').count().Item)
    avg_act_item = np.mean(data.groupby('Item').count().User)
    n_actions = len(data)
    stats = [n_user, n_items, avg_act_user, avg_act_item, n_actions]

    if print_statistics:
        for sk, s in zip(stat_keys, stats):
            print('%s: %1.2f' % (sk, s))

    return stats


# Settings
gowalla = False
ml1m = True
min_n_interactions = {'gowalla': 15, 'ml1m': 5}
stat_keys = ['n_user', 'n_items', 'avg_act_user', 'avg_act_item', 'n_actions']
reproduced_data_folder_name = 'reproduced_data'

# Create Result folder structure
if os.path.exists(reproduced_data_folder_name):
    shutil.rmtree(reproduced_data_folder_name)
    os.mkdir(reproduced_data_folder_name)
    for folder_name in ['gowalla', 'gowalla_small', 'ml1m']:
        os.mkdir(os.path.join(reproduced_data_folder_name, folder_name))
        os.mkdir(os.path.join(reproduced_data_folder_name, folder_name, 'test'))

if ml1m:
    os_ml1m = pd.read_csv('data/ml/ratings.dat', delimiter='::', header=None)
    ml1m_columns = ['User', 'Item', 'Rating', 'Time']
    paper_train_ml = pd.read_csv('data/ml/train.txt', delimiter=' ', header=None)
    paper_test_ml = pd.read_csv('data/ml/test.txt', delimiter=' ', header=None)
    paper_joined_ml = pd.concat([paper_train_ml, paper_test_ml])
    paper_joined_ml.columns = ['User', 'Item', 'IsRated']
    print('\nStatistics for reproduced ML-1M Data:')
    ml_train, ml_test, ml_stats = create_training_sets_from_source('ml1m', os_ml1m, ml1m_columns)
    print('\nStatistics for Repository ML-1M Data:')
    get_statistics_from_dataset(paper_joined_ml)
if gowalla:
    os_gowalla = pd.read_csv('data/gowalla/original_source.txt', delimiter='\t', header=None)
    gowalla_columns = ['User', 'Time', 'Lat', 'Lon', 'Item']
    paper_train_gowalla = pd.read_csv('data/gowalla/train.txt', delimiter=' ', header=None)
    paper_test_gowalla = pd.read_csv('data/gowalla/test.txt', delimiter=' ', header=None)
    paper_joined_gowalla = pd.concat([paper_train_gowalla, paper_test_gowalla])
    paper_joined_gowalla.columns = ['User', 'Item', 'IsRated']
    print('\nStatistics for reproduced Gowalla Data:')
    g_train, g_test, g_stats = create_training_sets_from_source('gowalla', os_gowalla, gowalla_columns)
    print('\nStatistics for small reproduced Gowalla Data:')
    g_train_n, g_test_n, g_stats_n = create_training_sets_from_source('gowalla', os_gowalla, gowalla_columns,
                                                                      share_entries=0.318)
    print('\nStatistics for Repository Gowalla Data:')
    get_statistics_from_dataset(paper_joined_gowalla)

print('Done')
