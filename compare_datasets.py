import numpy as np
import pandas as pd

gowalla = True
ml = False
# INFO: os stands for original source

min_n_interactions = {'gowalla': 15, 'ml1m': 5}
train_share = 0.8

# Gowalla
if gowalla:
    os_gowalla = pd.read_csv('data/gowalla/original_source.txt', delimiter='\t', header=None)
    os_gowalla.columns = ['User', 'Time', 'Lat', 'Lon', 'LocID']

    interaction_counter = os_gowalla.groupby('User').count()['Time']
    users_to_consider = list(interaction_counter[interaction_counter > 14].index)
    entries_to_consider = os_gowalla[os_gowalla.User.isin(users_to_consider)]
    entries_to_consider.sort_values(['User', 'Time'], inplace=True)

    paper_train_gowalla = pd.read_csv('data/gowalla/train.txt', delimiter=' ', header=None)

    entries_to_consider = entries_to_consider[['User', 'LocID']]
    entries_to_consider['IsRated'] =[1] * len(entries_to_consider)

    train_indices = []
    test_indices = []

    for u in users_to_consider:
        user_data = entries_to_consider[entries_to_consider.User == u]
        n_entries = len(user_data)
        train_size = int(np.round(n_entries * train_share))
        train_indices += list(user_data.iloc[:train_size].index)
        test_indices += list(user_data.iloc[train_size:].index)

    train_data = entries_to_consider.loc[train_indices]
    train_data.to_csv('reproduced_data/gowalla/test/train.txt', sep=' ', header=None, index=None)
    test_data = entries_to_consider.loc[test_indices]
    test_data.to_csv('reproduced_data/gowalla/test/test.txt', sep=' ', header=None, index=None)

# ML
if ml:
    os_ml = pd.read_csv('data/ml/ratings.dat', delimiter='::', header=None)
    os_ml.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    paper_train_ml = pd.read_csv('data/ml/train.txt', delimiter=' ', header=None)
    paper_test_ml = pd.read_csv('data/ml/test.txt', delimiter=' ', header=None)
    paper_train_ml.columns = ['UserID', 'MovieID', 'IsRated']  # I assume ...
    paper_test_ml.columns = ['UserID', 'MovieID', 'IsRated']

    os_ml.sort_values(['UserID', 'Timestamp'], inplace=True)
    user_counts = os_ml.groupby('UserID').count().MovieID
    users_to_consider = user_counts[user_counts >= min_n_interactions['ml1m']].index
    relevant_data = os_ml[os_ml.UserID.isin(users_to_consider)][['UserID', 'MovieID']]
    # Nothing discarded, as smallest number of interactions is 20
    relevant_data['IsRated'] = [1] * len(relevant_data)

    train_indices = []
    test_indices = []

    for u in users_to_consider:
        user_data = relevant_data[relevant_data.UserID == u]
        n_entries = len(user_data)
        train_size = int(np.round(n_entries * train_share))
        train_indices += list(user_data.iloc[:train_size].index)
        test_indices += list(user_data.iloc[train_size:].index)

    train_data = relevant_data.loc[train_indices]
    train_data.to_csv('reproduced_data/ml-1m/test/train.txt', sep=' ', header=None, index=None)
    test_data = relevant_data.loc[test_indices]
    test_data.to_csv('reproduced_data/ml-1m/test/test.txt', sep=' ', header=None, index=None)

print('Done')
