Pascal GPU compatibility
========================

Pascal includes GTX1080Ti and Titan Xp GPUs (compute capability 6.1).
Read this page only if deploying on those devices.

.. warning::

   Architecture coverage alone is not workload qualification. Automatic routing
   now admits compatible Pascal CSR currents and repaired NEST state kernels,
   but initialization, training and inference still need measured VRAM headroom
   for your batch, connectivity and inputs.

Before choosing a Pascal GPU
----------------------------

* Confirm that TensorFlow, its CUDA dependencies and the driver support the
  device. A CUDA driver version alone is not enough.
* If using custom operators, include architecture 61 explicitly and verify
  both operator manifests; see :doc:`dpointnet_performance`. It is absent
  from the implicit build target list.
* Keep automatic acceleration rather than forcing packed flags from a newer
  GPU. It selects SM61 CSR currents, active queues and eligible native
  accumulation, while retaining canonical-gradient boundaries.
* Rebuild both operators from this source. Generic NEST launches now use
  kernel-specific occupancy limits. Automatic state admission checks a
  defaulted ``launch_geometry_version`` capability attribute; an old unmarked
  library is not admitted even if its manifest advertises SM61.
* Check the actual workload's initialization and update memory. Titan Xp
  has more memory than GTX1080Ti; fitting one does not establish fitting both.
* Test the configured dynamics, precision, inputs, updates and restoration.
  Do not change those scientific choices just to make a compatibility test pass.

NEST-compatible dynamics remain the default. Explicit legacy dynamics are
available for legacy reproduction, not as a scientifically equivalent
workaround for a NEST-compatible-mode device failure.

Why the native current route matters
------------------------------------

Unfused input projection can materialize large temporary connection tensors.
In the measured 66,658-neuron batch-10 V1 inference, the old Pascal fallback
exceeded memory headroom for hard reset and ran out of memory for soft reset.
With the existing SM61 native CSR current route, the same paired protocol
passed on GTX1080Ti at 5.16/5.18 GiB peak TensorFlow allocation. Do not extrapolate
this to batch-32 BPTT, whose temporal tape and optimizer add memory.

cuDNN compatibility is a separate surface: some convolution shapes are not
supported on Pascal by recent cuDNN. The synchronization-loss regression uses
a CPU-placed independent convolution oracle while retaining the actual
DPointNet value/gradient calculation on GPU and unchanged strict tolerances.
This does not install a general CPU fallback in the model.

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
