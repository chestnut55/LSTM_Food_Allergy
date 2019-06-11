from collections import defaultdict

import numpy as np
import pandas as pd


def metadata(file_path):
    meta = pd.read_csv(file_path, sep=",", index_col=False)
    meta = meta.dropna(subset=['gid_wgs'])

    meta = meta.dropna(subset=['allergy_milk','allergy_egg','allergy_peanut'])

    allergy_milk = meta.allergy_milk.values.tolist()
    allergy_egg = meta.allergy_egg.values.tolist()
    allergy_peanut = meta.allergy_peanut.values.tolist()

    allergy = []
    for milk, egg, peanut in zip(allergy_milk, allergy_egg, allergy_peanut):
        if milk or egg or peanut:
            allergy.append(True)
        else:
            allergy.append(False)

    meta['allergy'] = allergy
    return meta


def time_points_data(df_meta, df_data):
    timepoints = defaultdict(list)
    groups = df_meta.groupby('subjectID')
    for name, group in groups:
        gid_wgs = group['gid_wgs'].values.tolist()
        if len(gid_wgs) > 2:
            timepoints.setdefault(name, []).extend(gid_wgs)
        else:
            df_meta = df_meta.drop(df_meta[df_meta['subjectID'] == name].index)
            df_data = df_data.drop(columns = gid_wgs)
    return timepoints, df_meta, df_data


def df_genus_features(file_path):
    df = pd.read_csv(file_path, sep="\t", index_col=0)
    # df = (df - df.min()) / (df.max() - df.min())

    feature_list = list(df.index)
    results = []
    for feature in feature_list:
        fs = feature.split('|')
        if len(fs) == 6:
            results.append(feature)

    return df.loc[results]


def lstm_input(meta_file_name, data_file_name):
    meta_file = metadata(meta_file_name)
    data_file = df_genus_features(data_file_name)

    w_ids = set(data_file.columns.values).intersection(set(meta_file['gid_wgs'].values))
    meta_file = meta_file[meta_file['gid_wgs'].isin(w_ids)]
    data_file = data_file.loc[:, w_ids]

    time_points, meta_file, data_file = time_points_data(meta_file, data_file)
    subjects = list(time_points.keys())

    _, counts = np.unique(meta_file['subjectID'], return_counts=True)
    maxLen = max(counts)
    print(np.sum(counts))

    numFeatures = len(data_file.index)

    print("samples FIN="+ str(len(meta_file[meta_file['country'] == 'FIN'])))
    print("samples RUS=" + str(len(meta_file[meta_file['country'] == 'RUS'])))
    print("samples EST=" + str(len(meta_file[meta_file['country'] == 'EST'])))

    groups = meta_file.groupby('country')
    for name, group in groups:
        subjectIDs = set(group['subjectID'].values.tolist())
        print("subjects " +name +"=" + str(len(subjectIDs)))

    return maxLen, numFeatures, subjects, meta_file, time_points, data_file


if __name__ == '__main__':
    meta = '../data/metadata.csv'
    data = '../data/diabimmune_karelia_metaphlan_table.txt'
    lstm_input(meta, data)
