ICA-YAC
=======

-----

ICA-YAC (Yet Another Classifier) is a ICA artefact removal tool that automatically classifies components into either "Signal" or "Noise".
It is similar in spirit to ICA-FIX and ICA-AROMA, but it only requires the timeseries of the components for the classification procedure. This makes ICA-YAC both fast, flexible and easy to use as it does not require motion parameter estimates, does not rely on any spatial features nor atlas, which allows training and predicting steps to be performed in different spaces, and yields results in seconds.

ICA-YAC requires minimal training if your dataset does not fit any of the pretrained classes that come with this package.

.. contents:: **Table of Contents**
    :backlinks: none

Provided Classifiers
--------------------

ICA-YAC comes with a set of pretrained classifiers that can be used out-of-the-box to clean your data. These are listed below:

+------+------------+---------+-------+
| Name | Resolution | TE (ms) | TR (s)|
+=====================================+
| MESH |  4.5 iso   |   48.6  | 0.158 |
+------+------------+---------+-------+

If you train your own datasets and obtain good results, we welcome your classifier as a contribution.

Usage
-----

After installing ICA-yac, run :code:`ica-yac --help` for information on syntax and usage.


Installation
------------

ICA-YAC supports python 3.5 and 3.6.
Either clone the project and install it by running

.. code-block:: bash

    $ python setup.py install


or install it directly with pip by calling:

.. code-block:: bash

    $ pip install git+https://github.com/dangom/ica-yac.git

License
-------

ICA-YAC is distributed under the terms of

- `Apache License, Version 2.0 <https://choosealicense.com/licenses/apache-2.0>`_
