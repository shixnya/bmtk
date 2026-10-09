.. _dpointnet_training_standards--dpointnet-training-standards-and-configuration-migration:

DPointNet defaults and configuration migration
==============================================

This page explains the GLIF defaults in this source revision. For a first run,
start with :doc:`dpointnet_guide`; for objectives and update protocols, use
:doc:`dpointnet_training`. The migration section at the end is only needed
when updating an older configuration.

.. _dpointnet_training_standards--at-a-glance:
.. _dpointnet_training_standards--standard-defaults:

Current defaults
----------------

These are the defaults for new GLIF configurations. Explicit supported
settings override them; targets and the training protocol are chosen separately.

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Area
     - Default
     - What it means
   * - Dynamics / reset
     - ``nest`` / soft
     - DPointNet's NEST-compatible dynamics in TensorFlow; training uses subtractive reset.
   * - Spike derivative
     - Gaussian, width ``0.28``, gain ``0.05``
     - Smooth derivative used for learning, not a different forward spike threshold.
   * - Recurrent / voltage gradient scales
     - ``1.0`` / ``1.0``
     - Retain both learning paths; physical membrane decay still applies.
   * - Spike-triggered event derivatives
     - ``detach_reset=true``, ``detach_asc_reset=false``
     - Block the spike-to-voltage-reset derivative but retain the after-spike-current event derivative.
   * - Recurrent weight training
     - Individual edges when enabled
     - Each trainable recurrent edge has its own weight.
   * - High-level GLIF RNN precision
     - Mixed FP16, selective FP32 voltage/ASC
     - See the precision section below.
   * - Activation checkpointing
     - On, chunk size ``25`` steps
     - Save memory by recomputing intermediates, without truncating BPTT.
   * - Voltage output
     - Compact penalty tracking, not full voltage sequences
     - Tracking alone adds no loss and does not clamp voltage.
   * - Acceleration
     - ``auto``
     - Select compatible implementations without changing the requested precision or dynamics.
   * - Drifting-grating LGN generation
     - Device generation for supported seeded inputs
     - See :doc:`dpointnet_lgn_pipeline` if using this optional visual-input model.

``dynamics_mode="nest"`` selects DPointNet's NEST-compatible implementation,
not the NEST simulator. It is the default for new training and simulation.
Preserve explicit legacy dynamics when reproducing a
legacy experiment. The supplied examples explicitly preserve their original
scientific settings rather than silently adopting new ones.

Choose batch size, sequence length, seed, optimizer, learning rate, epochs,
condition layout, targets and loss coefficients for your experiment.
Visual-cortex (V1) recipe values are not automatically imposed on other models.
No activity-rescue loss is enabled automatically.

.. _dpointnet_training_standards--precision-switches:

Choose precision
----------------

**FP32** uses 32-bit floating point. **FP16** uses 16-bit floating point, saving
memory but representing a narrower range of values. Mixed precision uses FP16
for much of the computation while retaining FP32 trainable weights and
optimizer variables. The selective state policy also retains FP32 membrane
voltage and after-spike currents (ASC); other floating state remains compact.

Merge this **JSON fragment** into top-level ``run`` in a complete config:

.. code-block:: json

   {
     "run": {
       "precision": {
         "mixed_precision": true,
         "fp32_voltage_and_asc": true,
         "fp32_temporal_gradients": false
       }
     }
   }

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Switch
     - Default
     - Effect
   * - ``mixed_precision``
     - ``true``
     - FP16 compute with FP32 trainable weights/optimizer variables; false selects full FP32.
   * - ``fp32_voltage_and_asc``
     - ``true``
     - FP32 voltage/ASC under mixed precision.
   * - ``fp32_temporal_gradients``
     - ``false``
     - Wider derivatives across time, for checking accuracy under mixed precision.

.. list-table:: Precision profiles
   :header-rows: 1
   :widths: 35 15 20 30

   * - Profile
     - Compute
     - Voltage / ASC
     - Derivatives across time
   * - Default
     - FP16
     - FP32
     - FP16
   * - ``fp32_temporal_gradients=true``
     - FP16
     - FP32
     - FP32 replay path
   * - ``fp32_voltage_and_asc=false``
     - FP16
     - FP16
     - FP16
   * - ``mixed_precision=false``
     - FP32
     - FP32
     - FP32 compute path

For a simple CPU run, explicitly select ``mixed_precision=false`` as in the
small example. Precision is not automatically changed when no GPU is present.
Under mixed precision, FP32 temporal gradients require FP32 voltage/ASC.
All switches must be booleans; invalid combinations raise an error.

Full FP32 is never narrowed by the other switches. The mixed FP32-temporal
route preserves the original forward precision and requires a supported
temporal runner; see :doc:`dpointnet_performance` for its replay/storage costs.
A direct ``GLIF3Cell`` inherits its Keras dtype policy rather than installing
the high-level RNN's policy itself.

.. _dpointnet_training_standards--clear-names-for-gradient-controls:

Gradient controls and old names
-------------------------------

Put these fields in ``rnn_cell_params``. A surrogate is the smooth derivative
used to learn through spikes; a gradient scale adjusts a particular learning
path without changing the forward neuron equations.

.. list-table::
   :header-rows: 1
   :widths: 30 30 25 15

   * - Standard name
     - Accepted old name
     - Translation
     - Default
   * - ``spike_surrogate_gain``
     - ``dampening_factor``
     - Same value
     - ``0.05``
   * - ``recurrent_spike_gradient_scale``
     - ``recurrent_dampening_factor``
     - Same value
     - ``1.0``
   * - ``voltage_state_gradient_scale``
     - ``voltage_gradient_dampening``
     - Complement of the clipped old removed fraction
     - ``1.0``
   * - ``spike_surrogate_width``
     - ``gauss_std``
     - Same value
     - ``0.28``
   * - ``spike_surrogate``
     - ``pseudo_gauss``
     - ``true`` means Gaussian, ``false`` triangular
     - ``"gaussian"``

For recurrent and voltage scales, zero blocks that learning route and one
retains it. The recurrent scale does not block the direct weight gradient.
The voltage scale is not the biological membrane-decay constant.

.. important::

   The old voltage option describes the fraction **removed**, not retained:
   ``voltage_state_gradient_scale = 1 - clip(voltage_gradient_dampening, 0, 1)``.
   Old values 0, 0.5 and 1 therefore map to new values 1, 0.5 and 0.
   Supplying both an old name and its new name is an error, even if equivalent.

Surrogate gain is not a retention fraction. Width and gain jointly affect
learning; choose them together and record changes. Gaussian width does not
affect the fixed-width triangular surrogate.

.. _dpointnet_training_standards--automatic-acceleration:

Automatic acceleration
----------------------

Omitting ``rnn_cell_params.acceleration_profile`` selects ``"auto"``.
Leave individual kernel flags unspecified for ordinary use. Explicit flags
override automatic choices and are checked for compatibility; unsupported
explicit requests raise. Explicit ``acceleration_profile=null`` disables
profile expansion, not separately requested options.

CPU execution does not require custom CUDA operators. On GPUs, selection
depends on hardware, available operators, precision and network structure,
not just the CUDA version. See :doc:`dpointnet_performance` for setup and
:doc:`dpointnet_pascal` for older NVIDIA hardware limitations.

.. _dpointnet_training_standards--existing-examples-and-migration:

Updating an older configuration
-------------------------------

**Omitted settings now change behavior.** Accepting old parameter names does
not freeze the old defaults of other fields. If reproducing an old experiment,
make its settings explicit before updating BMTK.

.. list-table:: Changes from the previous GLIF defaults
   :header-rows: 1
   :widths: 40 30 30

   * - Setting
     - Previous
     - Current
   * - Dynamics
     - ``legacy``
     - **nest**
   * - Surrogate
     - Triangular
     - **Gaussian**
   * - Gaussian width / gain
     - ``0.5`` / ``0.3``
     - **0.28 / 0.05**
   * - Recurrent / voltage gradient retention
     - ``0.5`` / ``0.5``
     - **1.0 / 1.0**
   * - ASC-event derivative
     - Detached
     - **Attached**
   * - Individual recurrent weights in direct GLIF3Cell construction, when training is enabled
     - Not trainable by the old per-type flag
     - **Trainable**; high-level RNN already used individual edges
   * - High-level RNN compute / mixed state
     - FP32 / compute state
     - **Mixed FP16 / selective voltage-ASC state**
   * - Activation checkpointing
     - Off
     - **On**; interval remains 25 steps
   * - Full voltage output / penalty tracking
     - On / off
     - **Off / on**
   * - Acceleration profile
     - Opt-in
     - **Automatic**
   * - Drifting-grating LGN generation
     - Host
     - **Device** for supported seeded inputs

The historical ``train_recurrent_per_type=true`` flag does not implement
trainable type-shared recurrent weights; it disables individual recurrent
weight training. Do not use it to request weight tying.

The helper below returns a copy with previous scientific defaults made
explicit and automatic acceleration selected:

.. code-block:: python

   from bmtk.simulator.dpointnet.migration import preserve_previous_defaults

   migrated = preserve_previous_defaults(old_config_dict)

Review the result against the original experiment, including precision and
outputs. Explicit legacy ``run.dtype`` and cell precision fields remain
supported; conflicting requests with the new switches raise. To preserve a
historical all-FP16-state setup, specify ``state_precision="compute"`` as well
as its compute dtype.

Acceleration may change floating-point reduction order, so preserved scientific
settings do not imply bitwise-identical results. Check your application's
numerical tolerances and learning behavior rather than relaxing them to pass.

.. _dpointnet_training_standards--validation-and-scope:

Scope
-----

CPU/older-Keras compatibility and RTX3090 execution have been tested for this
defaults change. These tests do not cover every GPU family or establish
physical multi-GPU equivalence. Source/build validation procedures
belong in :doc:`dpointnet_development`; they are not additional settings needed
to run a model.
