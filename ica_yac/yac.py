#!/usr/bin/env python
"""
ICA-YAC (Yet Another Classifier)

ICA-YAC generates a labels file for your ICA components. It does a binary
classification (Signal / Noise) and saves the results to a file in a format
compatible with FSLEYES.
"""
import argparse
import ast
import os.path
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .externals import fslutils

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tsfresh


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

    # Before concatenating all the data, we change level_0, which will
    # correspond to tsfresh's COLUMN_ID so that components of different
    # scans get different numbers.
    if len(data) > 1:
        for index, (n_comps, dset) in enumerate(zip(np.cumsum(n_components),
                                                    data[1:])):
            dset['level_0'] = dset['level_0'] + n_comps

    # Now, because each dset has different component numbers, we can concatenate
    # them without problem.
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
        # We should have as many component IDs as we have labels.

        assert np.unique(yac_data['level_0']).shape == yac_labels.shape
    else:
        yac_labels = None

    return yac_data, yac_labels


def available_architectures():
    return list(map(lambda x: x[:-4],
                    os.listdir(os.path.join(os.path.dirname(__file__),
                                            'classifiers'))))


class YetAnotherClassifier():
    """The API is different from that in scikit-learn in that we pass in the
    data in the format that TSFRESH expects it to be, that is, data is a pandas
    DataFrame, and labels are Pandas Series
    """

    def_settings = tsfresh.feature_extraction.EfficientFCParameters()

    def __init__(self, architecture=None, distributor=None):
        self.trained = False
        self.architecture = architecture
        self.distributor = distributor

    def fit(self, data, labels):
        feats = tsfresh.extract_features(data,
                                         column_id='level_0',
                                         column_sort='level_1',
                                         default_fc_parameters=self.def_settings,
                                         distributor=self.distributor)

        tsfresh.utilities.dataframe_functions.impute(feats) # Remove NaNs, if any
        relevant_feats = tsfresh.select_features(feats,
                                                 labels,
                                                 fdr_level=1e-15)

        self.relevant_features = relevant_feats.columns
        self.settings = tsfresh.feature_extraction.settings.from_columns(relevant_feats)

        clf = RandomForestClassifier(n_estimators=40)
        clf.fit(relevant_feats, labels)
        self.classifier = clf
        self.trained = True

    def predict(self, data):
        if not self.trained:
            assert self.architecture is not None, 'No classifier selected and no fit performed.'
            filename = os.path.join(os.path.dirname(__file__),
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
        filename = os.path.join(os.path.dirname(__file__),
                                'classifiers', name + '.pkl')
        with open(filename, 'wb+') as f:
            pickle.dump(dumpdata, f)


def train(args):

    if os.path.exists(args.name) and not args.force:
        msg = 'Attempt to overwrite already existing architecture file. '
        msg += 'If this is the intended behaviour, try again with --force'
        raise FileExistsError(msg)

    data, labels = load_fsl(args.inputdir, labels_file=args.labelfile)
    yac = YetAnotherClassifier()
    yac.fit(data, labels)

    yac.dump(args.name)


def predict(args):

    if os.path.isabs(args.labelfile):
        target_file = args.labelfile
    else:
        target_file = os.path.join(args.inputdir, args.labelfile)

    if os.path.exists(target_file) and not args.force:
        msg = 'Attempt to overwrite already existing label file. '
        msg += 'If this is the intended behaviour, try again with --force'
        raise FileExistsError(msg)

    data, _l = load_fsl(args.inputdir, labels_file=None)
    yac = YetAnotherClassifier(args.architecture)
    prediction = yac.predict(data)

    save_prediction(prediction, target_file)


def visualize(args):
    data, labels = load_fsl(args.inputdir, labels_file=args.labelfile)
    visdata = np.vstack(data.groupby('level_0')[0].apply(np.hstack).values).T
    fig, axes = plt.subplots(nrows=15, ncols=10, figsize=plt.figaspect(0.5))
    for index, item in enumerate(axes.flatten()):
        color = 'g' if index in np.nonzero(labels)[0] else 'r'
        try:
            item.plot(visdata[:200, index], color, linewidth=1)
        except IndexError:
            # TODO Better handling of n_components != 150
            pass
        for k, v in item.spines.items():
            v.set_visible(False)
        item.set_xticks([])
        item.set_yticks([])
    fig.suptitle('YAC Classification', fontsize=18)
    plt.show()


def _cli_parser():

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(help='sub-command help')

    # Create the parser for the training command
    parser_t = subparsers.add_parser('train', help='Train YAC on new datasets')

    parser_t.add_argument('inputdir', type=str, nargs='+',
                          help='Input directory(ies) with melodic_mix and hand_classification')

    parser_t.add_argument('name', type=str,
                          help='Target name for classifier \"architecture\"')

    parser_t.add_argument('-l', '--labelfile', type=str,
                          default='hand_classification.txt',
                          help='Name of classification file. Default hand_classification.txt')

    parser_t.add_argument('--force', action='store_true',
                          help='Overwrite existing classifier architecture.')

    parser_t.set_defaults(func=train)

    # Create the parser for the predict command
    parser_p = subparsers.add_parser('predict', help='Generate prediction')

    parser_p.add_argument('inputdir', type=str,
                          help='Input directory with melodic_mix')

    parser_p.add_argument('-l', '--labelfile', type=str,
                          default='yac_classification.txt',
                          help='Name of classification file. Default yac_classification.txt')

    parser_p.add_argument('--force', action='store_true',
                          help='Overwrite existing hand label.')

    parser_p.add_argument('-a', '--architecture', type=str,
                          default='MESH', choices=available_architectures(),
                          help='Name of classifier \"architecture\"')
    parser_p.set_defaults(func=predict)

    # Create a parser for visualization
    parser_v = subparsers.add_parser('visualize', help='Plot component classification')

    parser_v.add_argument('inputdir', type=str,
                          help='Input directory with melodic_mix and hand_classification')

    parser_v.add_argument('-l', '--labelfile', type=str,
                          default='yac_classification.txt',
                          help='Name of classification file. Default yac_classification.txt')

    parser_v.set_defaults(func=visualize)

    return parser


def run_yac():

    parser = _cli_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    run_yac()
