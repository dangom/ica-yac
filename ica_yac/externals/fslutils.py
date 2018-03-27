"""
Code taken from fsl.py https://git.fmrib.ox.ac.uk/fsl/fslpy
Copyright 2016-2017 University of Oxford, Oxford, UK.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


def isNoisyComponent(labels, signalLabels=None):
    """Given a set of component labels, returns ``True`` if the component
    is ultimately classified as noise, ``False`` otherwise.

    :arg signalLabels: Labels which are deemed signal. If a component has
                       no labels in this list, it is deemed noise. Defaults
                       to ``['Signal', 'Unknown']``.
    """
    if signalLabels is None:
        signalLabels = ['signal', 'unknown']

    signalLabels = [l.lower() for l in signalLabels]
    labels       = [l.lower() for l in labels]
    noise        = not any([sl in labels for sl in signalLabels])

    return noise


def saveLabelFile(allLabels,
                  filename,
                  dirname=None,
                  listBad=True,
                  signalLabels=None):
    """Saves the given classification labels to the specified file. The
    classifications are saved in the format described in the
    :func:`loadLabelFile` method.

    :arg allLabels:    A list of lists, one list for each component, where
                       each list contains the labels for the corresponding
                       component.

    :arg filename:     Name of the file to which the labels should be saved.

    :arg dirname:      If provided, is output as the first line of the file.
                       Intended to be a relative path to the MELODIC analysis
                       directory with which this label file is associated. If
                       not provided, a ``'.'`` is output as the first line.

    :arg listBad:      If ``True`` (the default), the last line of the file
                       will contain a comma separated list of components which
                       are deemed 'noisy' (see :func:`isNoisyComponent`).

    :arg signalLabels: Labels which should be deemed 'signal' - see the
                       :func:`isNoisyComponent` function.
    """

    lines      = []
    noisyComps = []

    # The first line - the melodic directory name
    if dirname is None:
        dirname = '.'

    lines.append(dirname)

    for i, labels in enumerate(allLabels):

        comp   = i + 1
        noise  = isNoisyComponent(labels, signalLabels)
        labels = [l.replace(',', '_') for l in labels]
        tokens = [str(comp)] + labels + [str(noise)]
        lines.append(', '.join(tokens))

        if noise:
            noisyComps.append(comp)

    if listBad:
        lines.append('[' + ', '.join([str(c) for c in noisyComps]) + ']')

    with open(filename, 'wt') as f:
        f.write('\n'.join(lines) + '\n')
