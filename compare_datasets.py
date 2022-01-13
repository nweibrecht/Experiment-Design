import pandas as pd

gowalla = True
ml = True
print('Start!')
# INFO: os stands for original source

# Gowalla
if gowalla:
    os_gowalla = pd.read_csv('data/gowalla/original_source.txt', delimiter='\t', header=None)
    os_gowalla.columns = ['User', 'Time', 'Lat', 'Lon', 'LocID']

    interaction_counter = os_gowalla.groupby('User').count()['Time']
    users_to_consider = list(interaction_counter[interaction_counter > 14].index)
    entries_to_consider = os_gowalla[os_gowalla.User.isin(users_to_consider)]
    entries_to_consider.sort_values(['User', 'Time'], inplace=True)

    paper_train_gowalla = pd.read_csv('data/gowalla/train.txt', delimiter=' ', header=None)

# ML
if ml:
    os_ml = pd.read_csv('data/ml/ratings.dat', delimiter='::', header=None)
    os_ml.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    paper_train_ml = pd.read_csv('data/ml/train.txt', delimiter=' ', header=None)

print('Done')
