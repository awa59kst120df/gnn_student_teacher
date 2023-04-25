"""
This string will be saved to the experiment's archive folder as the "experiment description"
"""
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from pycomex.experiment import Experiment
from pycomex.util import Skippable


SHORT_DESCRIPTION = (
    'This will be the short description when listing the experiments from the command line'
)

# Every variable which is defined on this upmost module level and whose name is all uppercase is
# automatically detected as a so-called "experiment parameter". These parameters will be saved as
# additional artifacts as part of the experiment.
BIN_COUNT = 5

with Skippable(), (e := Experiment(base_path=os.getcwd(), namespace='template', glob=globals())):
    # "e.info" should be used instead of "print". It will use python's "logging" module to not only
    # print the content ot the console but also write it into a log file at the same time.
    e.info('starting experiment...')

    for index in range(BIN_COUNT):
        value = random.randint(1, 100)
        # Directly index the experiment manager as if it was a dict to permanently commit values obtained
        # during the experiment to be saved in a persistent file within the experiment's archive folder.
        # Use the slash character "/" to define nested dictionary structures. And even if a given
        # structure does not yet exist, the complete nesting tree will automatically be created first!
        e[f'values/{index}'] = value

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    xs = list(range(BIN_COUNT))
    ys = list(e['values'].values())
    ax.plot(xs, ys)
    # Directly commit matplotlib figures as additional experiment artifacts to be saved in the artifact
    # folder!
    e.commit_fig('plot.pdf', fig)


# Use this separate analysis context manager to define any code which represents any sort of post-processing
# or analysis which should be performed after the experiment is over and which should potentially be
# repeated.
# All the code within this context manager will automatically be copied into a "analysis.py" file within the
# archive folder of the experiment run and from that point on it can be run independently of the experiment
# again and again, in case adjustments have to be made to the analysis.
with Skippable(), e.analysis:
    e.info('starting analysis...')

    # All of this code will still work in the "analysis.py" file! This is because a snapshot copy of this
    # very file will be imported within that analysis module and the experiment context manager will
    # automatically load the persisted experiment values from the JSON file which they were saved in.
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    xs = list([int(key) for key in e['values'].keys()])
    ys = list(e['values'].values())
    ax.bar(xs, ys)
    e.commit_fig('bar.pdf', fig)


# NOTE: This file can be safely imported. Importing this module will NOT cause the experiment code to be
# executed. The experiment context automatically detects whenever it is being imported vs. directly
# executed and will skip the entire content of the context manager in the former case!
