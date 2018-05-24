# -*- coding: utf-8 -*-
"""
This module defines default signals supported by NiftyNet engine.

By design, all connected functions (event handlers) have access
to TF session and graph
by ``tf.get_default_session()`` and ``tf.get_default_graph()``
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

# namespace of NiftyNet's default signals.
NIFTYNET = Namespace()

ITER_STARTED = NIFTYNET.signal(
    'iteration_started',
    doc='emitted when every iteration starts, before ``tf.session.run(...)``.')
ITER_FINISHED = NIFTYNET.signal(
    'iteration_finished',
    doc='emitted at the end of each iteration, after ``tf.session.run(...)``.')

SESS_STARTED = NIFTYNET.signal(
    'session_started',
    doc='emitted at the beginning of the training/inference loop.')
SESS_FINISHED = NIFTYNET.signal(
    'session_finished',
    doc='signal emitted at the end of the training/inference loop.')

GRAPH_FINALISED = NIFTYNET.signal(
    'tf_graph_finalised',
    doc='signal emitted when the graph is finalised.')

# EPOCH_STARTED = NIFTYNET.signal(
#     'epoch_started',
#     doc='emitted at the beginning of each training epoch')
# EPOCH_FINISHED = NIFTYNET.signal(
#     'epoch_finished',
#     doc='emitted at the end of each training epoch')
