# -*- coding: utf-8 -*-
"""
This module defines default signals supported by NiftyNet engine.

By design, all connected functions (event handlers) have access
to TF session and graph
by ``tf.get_default_session()`` and ``tf.get_default_graph()``.
"""

from __future__ import print_function
from __future__ import unicode_literals

from blinker import Namespace

# possible phases of the engine, used throughout the project.
TRAIN = 'training'
VALID = 'validation'
INFER = 'inference'
EVAL = 'evaluation'
ALL = 'all'

#: namespace of NiftyNet's default signals.
NIFTYNET = Namespace()

#: Signal emitted immediately after the application's graph is created
GRAPH_CREATED = NIFTYNET.signal(
    'graph_started',
    doc="emitted when application's graph is created")

#: Signal emitted at the beginning of a training/inference process
#: (after the creation of both graph and session.)
SESS_STARTED = NIFTYNET.signal(
    'session_started',
    doc='signal emitted at the beginning of the training/inference loop.')

#: Signal emitted before the end of a training/inference process
#: (after the creation of both graph and session.)
SESS_FINISHED = NIFTYNET.signal(
    'session_finished',
    doc='signal emitted at the end of the training/inference loop.')

#: Signal emitted immediately when each iteration starts
#: (after the creation of both graph and session.)
ITER_STARTED = NIFTYNET.signal(
    'iteration_started',
    doc='emitted when every iteration starts, before ``tf.session.run(...)``.')

#: Signal emitted before the end of each iteration
#: (after the creation of both graph and session.)
ITER_FINISHED = NIFTYNET.signal(
    'iteration_finished',
    doc='emitted at the end of each iteration, after ``tf.session.run(...)``.')


# EPOCH_STARTED = NIFTYNET.signal(
#     'epoch_started',
#     doc='emitted at the beginning of each training epoch')
# EPOCH_FINISHED = NIFTYNET.signal(
#     'epoch_finished',
#     doc='emitted at the end of each training epoch')
