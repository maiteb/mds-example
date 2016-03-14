import argparse

import numpy as np
import pandas as pd
from sklearn.manifold import MDS

from scipy.spatial.distance import squareform,pdist


def _parse_file_argument():
    parser = argparse.ArgumentParser("MDS Commandline Tool")
    parser.add_argument('--csv',
                        help='csv filename')
    parser.add_argument('--label_prefix',
                        help='csv\'s label column prefix')
    parser.add_argument('--label_sufix',
                        help='csv\'s label column sufix')
    args = parser.parse_args()
    return args



RS = 20160101

if __name__ == '__main__':
    args = _parse_file_argument()
    data = pd.read_csv(args.csv)
    data.fillna(0, inplace=True)

    label_column = args.label_prefix
    label_prefix = data[label_column].values
    data.drop(label_column, axis=1, inplace=True)

    label_column = args.label_sufix
    label_sufix = data[label_column].values
    data.drop(label_column, axis=1, inplace=True)

    id_column = 'id'
    data.drop(id_column, axis=1, inplace=True)

    df_norm = (data - data.mean()) / (data.max() - data.min())

    print "Limpou as colunas"

    similarities = squareform(pdist(df_norm,'euclidean'))

    print "Calculou similaridades"

    mds = MDS(n_components=2, metric=False, max_iter=100, eps=1e-6, verbose=2, random_state=RS, dissimilarity="precomputed", n_jobs=4)
    mds_result = mds.fit_transform(similarities)

    print "Calculou o MDS"

    labels = []
    for x in range(len(label_prefix)):
        label_to_add = str(label_prefix[x])
        if label_sufix[x] is not None:
            label_to_add = label_to_add + '-' + str(label_sufix[x])
        labels.append(label_to_add)

    data_frame = pd.DataFrame(index=labels,data=mds_result,columns=['x','y'])
    data_frame.to_csv('mds.csv', index_label='label')

