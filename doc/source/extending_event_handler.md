# Signals and event handlers

NiftyNet's engine features highly customisable interfaces by decoupling the
main loop iterator and the application-specific functions triggered at each
iteration.  This is implemented with the
[`blinker`](https://pythonhosted.org/blinker/) library.

The [available signals](niftynet.engine.signal.html) are:
- `GRAPH_CREATED`;
- `SESS_STARTED`;
- `SESS_FINISHED`;
- `ITER_STARTED`;
- `ITER_FINISHED`.

Event handler functions registered to these signals will be called by the
engine, along with NiftyNet application properties and iteration messages as
function parameters.


## The engine and signals
Engine emits signals at various stages of the `net_run` processes.  The pattern
is defined in
[niftynet/engine/application_driver.py](niftynet.engine.application_driver.html)
and is shared among all applications.

An abstraction of the engine and signals is illustrated as follow:
```python
event_handlers = "a list of event_handler class names (from user's config file) "
# initialise the event handler instance with the engine's properties
initialise_event_handlers(engine_properties)

with tf.Graph().as_default():
    ... adding application network layer ops.
    ... adding application network gradient ops.

    # Notifying event handlers that
    # are registered with the `GRAPH_CREATED` signal.
    GRAPH_CREATED.send(application, iter_msg=None)

with tf.Session() as sess:
    # Notifying event handlers that
    # are registered with the `SESS_STARTED` signal.
    SESS_STARTED.send(application, iter_msg=None)

    # the main loop of training/inference
    for iter_msg in iter_msg_generator(starting_iter, end_iter):

        # Notifying event handlers that
        # are registered with the `ITER_STARTED` signal.
        ITER_STARTED.send(application, iter_msg=iter_msg)

        # run tensorflow variables
        # optionally this command will also update the network params.
        sess.run(iter_msg.get_variables_to_run())

        # Notifying event handlers that
        # are registered with the `ITER_FINISHED` signal.
        ITER_FINISHED.send(application, iter_msg=iter_msg)

    # Notifying event handlers that
    # are registered with the `SESS_FINISHED` signal.
    SESS_FINISHED.send(application, iter_msg=None)
```


## Event handlers
Event handlers are customisable classes with their methods connected to the
relevant signals.  For example, [a model
saver](_modules/niftynet/engine/handler_model.html#ModelSaver) could save the
trainable parameters as TensorFlow checkpoints every 10 iterations during
training. The pseudo-code would be:
```python
# using ITER_FINISHED provided by NiftyNet's engine
from niftynet.engine.signal import ITER_FINISHED

class ModelSaver(object):
    def __init__(self, engine_property_1, ...):
        # The handlers are stateful
        self.model_dir = validate_model_file_directory(engine_property_1)
        self.save_every_n = 10

        # Register self.self.save_model_interval with the `ITER_FINISHED` signal.
        # self.save_model_interval will be called
        # before the end of every iteration
        ITER_FINISHED.connect(self.save_model_interval)

    def save_model_interval(self, app, **msg):
        # reading the current iteration number from the message from engine
        current_iteration = msg['iter_msg'].current_iter
        if current_iteration % self.save_every_n == 0:
          ... call saving model functions with self.model_dir...

```

## Customised event handlers
NiftyNet supports the mixture of [build-in](niftynet.engine.application_factory.html#niftynet.engine.application_factory.EventHandlerFactory) and customised event handlers, to use
a default handler (e.g.
[`tensorboard_logger`](niftynet.engine.handler_tensorboard.html))
and a `NewHandlerClass` in Python script located in
`my_extension/event_handlers.py`

```bash
net_segment ... --event_handler \
  tensorboard_logger,my_extension.event_handlers.NewHanderClass
```
