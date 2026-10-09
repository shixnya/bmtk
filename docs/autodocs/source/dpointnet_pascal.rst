Pascal GPU compatibility
========================

Pascal includes GTX1080Ti and Titan Xp GPUs (compute capability 6.1).
Read this page only if deploying on those devices.

.. warning::

   General NEST-compatible execution on Pascal is not fully validated. Earlier
   complete regression testing failed, including fused neuron-state launch
   errors. Passing individual kernels or LGN input tests does not establish
   support for your full model or long training run.

Before choosing a Pascal GPU
----------------------------

* Confirm that TensorFlow, its CUDA dependencies and the driver support the
  device. A CUDA driver version alone is not enough.
* If using custom operators, include architecture 61 explicitly and verify
  both operator manifests; see :doc:`dpointnet_performance`. It is absent
  from the implicit build target list.
* Keep automatic acceleration rather than forcing native/packed flags from
  a newer GPU. Automatic selection is conservative on Pascal.
* Check the actual workload's initialization and update memory. Titan Xp
  has more memory than GTX1080Ti; fitting one does not establish fitting both.
* Test the configured dynamics, precision, inputs, updates and restoration.
  Do not change those scientific choices just to make a compatibility test pass.

NEST-compatible dynamics remain the default. Explicit legacy dynamics are
available for legacy reproduction, not as a scientifically equivalent
workaround for a NEST-compatible-mode device failure.

Visual inputs
-------------

DPointNet includes a pre-Volta GPU path for LGN spatial filtering that avoids
unsupported cuDNN convolution. It retains the original padding, normalization,
row order and temporal filters, and chunks large movies.
This is specific to LGN filtering, not a general convolution fallback.
Measure actual input shapes and memory before using it in a model.

If execution fails
------------------

Retain the first exception, model/config revision, precision report, operator
manifests, TensorFlow/CUDA versions and GPU name. Establish whether it is a
missing target, unsupported library operation, launch-resource or memory
failure before selecting a recovery.
Initial-state recovery notices are explained in :doc:`dpointnet_input_recovery`.

A successful build or another GPU family's test suite is not a complete
Pascal qualification. Kernel implementation details belong in
:doc:`dpointnet_development`, not in the ordinary configuration workflow.
