import ast
import os.path
import pickle

import numpy as np
import pandas as pd
import tsfresh
from sklearn.ensemble import RandomForestClassifier

from .externals import fslutils


def _last_line(file):
    with open(file, 'r') as f:
        for line in f:
            x = line
    return x


def _has_melodic_mix(directory):
    return os.path.exists(os.path.join(directory, 'melodic_mix'))


def _has_classification(directory, labels_file):
    return os.path.exists(os.path.join(directory, labels_file))


def save_prediction(prediction, filename):
    labels = [["Unclassified Noise"] if not x else ["Signal"]
              for x in prediction]
    fslutils.saveLabelFile(labels, filename)


def load_fsl(directories, labels_file='hand_classification.txt'):
    """Given one or many MELODIC directories, extract the melodic mix
    and the hand classification.
    """
    if isinstance(directories, str):
        directories = [directories]

    for directory in directories:
        assert _has_melodic_mix(directory), f'Missing melodic mix in {directory}'

    mixing_matrices = [np.loadtxt(x)
                       for x in [os.path.join(dir_, 'melodic_mix')
                                 for dir_ in directories]]

    data = [pd.DataFrame(mix).T.stack().reset_index()
            for mix in mixing_matrices]
    n_components = [x.shape[1] for x in mixing_matrices]

    if len(data) > 1:
        for index, (n_comps, dset) in enumerate(zip(n_components[:-1],
                                                    data[1:])):
            dset['level_0'] = dset['level_0'] + n_comps

    yac_data = pd.concat(data, ignore_index=True)

    if labels_file is not None:
        for directory in directories:
            assert _has_classification(directory, labels_file), f'Missing labels in {directory}'

        raw_labels = [[c-1 for c in ast.literal_eval(_last_line(x))]
                      for x in [os.path.join(dir_, labels_file)
                            for dir_ in directories]]
        labels = [[x not in lst for x in range(n_components[index])]
                  for index, lst in enumerate(raw_labels)]

        yac_labels = pd.Series(sum(labels, []))
    else:
        yac_labels = None

    return yac_data, yac_labels


class YetAnotherClassifier():
    """The API is different from that in scikit-learn in that we pass in the
    data in the format that TSFRESH expects it to be, that is, data is a pandas
    DataFrame, and labels are Pandas Series
    """

    def_settings = tsfresh.feature_extraction.EfficientFCParameters()

    def __init__(self, architecture=None):
        self.trained = False
        self.architecture = architecture

    def fit(self, data, labels):
        feats = tsfresh.extract_features(data,
                                         column_id='level_0',
                                         column_sort='level_1',
                                         default_fc_parameters=self.def_settings)

        tsfresh.utilities.dataframe_functions.impute(feats) # Remove NaNs, if any:
        relevant_feats = tsfresh.select_features(feats,
                                                 labels,
                                                 fdr_level=1e-18)

        self.relevant_features = relevant_feats.columns
        self.settings = tsfresh.feature_extraction.settings.from_columns(relevant_feats)

        clf = RandomForestClassifier(n_estimators=40)
        clf.fit(relevant_feats, labels)
        self.classifier = clf
        self.trained = True

    def predict(self, data):
        if not self.trained:
            assert self.architecture is not None, 'No classifier selected and no fit performed.'
            filename = os.path.join(__file__,
                                    'classifiers', self.architecture + '.pkl')
            with open(filename, 'rb') as f:
                arch = pickle.load(f)
            clf = arch['clf']
            settings = arch['settings']
            relevant_features = arch['relevant_features']
        else:
            clf = self.classifier
            settings = self.settings
            relevant_features = self.relevant_features

        features = tsfresh.extract_features(data,
                                            column_id='level_0',
                                            column_sort='level_1',
                                            default_fc_parameters=settings['0'])

        return clf.predict(features[relevant_features])

    def dump(self, name):
        dumpdata = {'clf': self.classifier,
                    'settings': self.settings,
                    'relevant_features': self.relevant_features}
        filename = os.path.join(__file__, 'classifiers', name + '.pkl')
        with open(filename, 'wb+') as f:
            pickle.dump(dumpdata, f)
