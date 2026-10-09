# DPointNet training standards and configuration migration

**Status:** user-designated standard for new, unpinned GLIF engineering work
(2026-10-08), with the local qualification below.
Publication, installation and migration of existing experiments are separate actions.

## At a glance

DPointNet now has a consistent starting configuration for new GLIF training:
NEST dynamics, a Gaussian spike surrogate, undampened recurrent and membrane
credit, mixed precision, selective FP32 state, checkpointed BPTT and
hardware-aware acceleration. Project-specific experimental choices remain
explicit.

**This is an intentional change to omitted-setting behavior.** Explicit supported
settings remain authoritative. Existing configuration files using the old
parameter names are still accepted, but omitted fields receive the new defaults.
The repository examples explicitly preserve their previous scientific settings.
Installed environments, active experiments and archived execution snapshots are
not migrated automatically.

NEST execution and gradient validation do **not** establish scientific training
convergence. These settings are a documented V1 starting reference, not a claim
of a globally optimal surrogate, learning rate or objective.

## Standard defaults

| Area | Previous default | New standard | Interpretation |
|---|---|---|---|
| Dynamics | `legacy` | **`nest`** | NEST-aligned dynamics; explicit legacy remains supported |
| Training reset | Soft through the training wrapper | Soft | Direct cell construction also defaults to soft |
| Spike surrogate | Triangular | **Gaussian** | Explicit shape selection replaces a boolean |
| Gaussian width | `0.5` | **`0.28`** | Width and gain must be considered together |
| Surrogate gain | `0.3` | **`0.05`** | Backward spike derivative, not forward firing threshold |
| Recurrent spike credit | Retain `0.5` | **Retain `1.0`** | No additional recurrent-gradient attenuation |
| Voltage self-loop credit | Retain `0.5` | **Retain `1.0`** | Physical membrane decay still applies |
| Reset-event attachment | Detached | Detached | `detach_reset=true` |
| ASC-event attachment | Detached | **Attached** | `detach_asc_reset=false` |
| Direct cell recurrent sharing | Per type | **Per edge** | High-level RNN already selected per-edge training |
| RNN compute policy | FP32 | **Mixed FP16** | FP32 master variables and optimizer slots |
| Mixed-FP16 state policy | Compute precision | **Selective** | FP32 voltage/ASC; compact synaptic state |
| Temporal gradients | Compute precision | Compute precision | Ordinary FP16 temporal path under mixed FP16 |
| BPTT checkpointing | Off | **On** | Memory optimization, not truncation of credit |
| Checkpoint interval | 25 steps | 25 steps | 25 ms only when `dt=1 ms` |
| Acceleration profile | Opt-in | **Automatic** | Hardware/operator/topology-qualified routes |
| Full voltage sequences | Returned | **Not returned by default** | Request them explicitly when an objective needs them |
| Voltage-penalty tracking | Off | **On** | Tracking does not install a loss or its coefficient |
| Drifting-grating LGN generation | Host path | **Device path** | Requires a seed and supported strategy; seeded sample assignment is retained |

Direct GLIF cells inherit their surrounding Keras compute policy. When that
policy is not mixed FP16, omitted state precision resolves to `compute`, not an
invalid selective-state request. The high-level GLIF RNN selects mixed FP16 by
default. Other cell classes retain their previous FP32 RNN default.

### Values that remain project-specific

Batch size, sequence length, condition groups, epochs, steps per epoch, learning
rate and optimizer selection are not inferred from the V1 recipe. Neither are
scientific targets, input-weight training groups, loss coefficients, rescue
subsets, stimulus splits or seeds.

Batch32 and a 500-ms window are useful V1 reference choices, not imposed defaults.
The V1 reference uses expAdam with constant LR0.01; a project must select those
values explicitly. No activity-rescue loss is installed automatically.

## Clear names for gradient controls

| Standard name | Legacy alias accepted | Translation | Standard value |
|---|---|---|---|
| `spike_surrogate_gain` | `dampening_factor` | Same value | `0.05` |
| `recurrent_spike_gradient_scale` | `recurrent_dampening_factor` | Same value | `1.0` |
| `voltage_state_gradient_scale` | `voltage_gradient_dampening` | **Complement of the old removed fraction** | `1.0` |
| `spike_surrogate_width` | `gauss_std` | Same value | `0.28` |
| `spike_surrogate` | `pseudo_gauss` | `true` -> `gaussian`; `false` -> `triangular` | `gaussian` |

For the recurrent and voltage controls, zero blocks that route and one retains
the complete derivative. `recurrent_spike_gradient_scale` affects recurrent spike
credit, not the direct weight gradient. `voltage_state_gradient_scale` affects the
membrane-voltage self-loop, not the biological decay constant.

The old voltage convention was inverted:

```text
old voltage_gradient_dampening = 0.0 -> new voltage_state_gradient_scale = 1.0
old voltage_gradient_dampening = 0.5 -> new voltage_state_gradient_scale = 0.5
old voltage_gradient_dampening = 1.0 -> new voltage_state_gradient_scale = 0.0
```

Legacy voltage clipping semantics are retained. Supplying both a standard name
and its legacy alias is an error, even when the values appear equivalent.
The low-level kernel interfaces retain their existing names and ABI.

`spike_surrogate_gain` is a gain, not a retention fraction. Gaussian width and
gain jointly affect temporal Jacobians; changing either is a scientific change.
The width is ignored by the fixed-width triangular surrogate.

## Precision switches

Precision is explicit and is not silently lowered to fit a GPU.

```json
{
  "run": {
    "precision": {
      "mixed_precision": true,
      "fp32_voltage_and_asc": true,
      "fp32_temporal_gradients": false
    }
  }
}
```

| Switch | Default | Effect |
|---|---|---|
| `mixed_precision` | `true` | FP16 compute with FP32 masters/slots; false selects FP32 |
| `fp32_voltage_and_asc` | `true` | Selective FP32 voltage/ASC under mixed FP16 |
| `fp32_temporal_gradients` | `false` | True selects the FP32 temporal-reference path under mixed FP16 |

| Profile | Compute | Voltage/ASC | Temporal path | Replay |
|---|---|---|---|---|
| Standard | FP16 | FP32 | Ordinary FP16 | None |
| FP32 temporal reference | FP16 | FP32 | FP32 | Recorded by default |
| Compute-state alternative | FP16 | FP16 | Ordinary FP16 | None |
| `mixed_precision=false` | FP32 | FP32 | FP32 compute path | None |

Disabling mixed precision never downcasts a state or gradient because another
switch is false. Under mixed FP16, FP32 temporal gradients require FP32
voltage/ASC state. Invalid combinations and unknown/non-boolean switches raise
configuration errors before model construction.

Explicit legacy `run.dtype`, `state_precision` and
`temporal_gradient_precision` remain supported. If both interfaces are supplied,
conflicting requests raise an error. A legacy `dtype` selects the original
compute policy; supply `state_precision="compute"` too when preserving a
historical all-compute-state mixed-FP16 configuration.

## Automatic acceleration

Omitting `acceleration_profile` selects `"auto"`. It does not change the requested
scientific parameters or precision. Individual explicit accelerator settings
remain authoritative and are validated rather than silently overridden.
Explicit `acceleration_profile=null` disables profile expansion.

| Route | Selection rule |
|---|---|
| Fused currents/state and direct loop | Compatible loaded operators, supported compute/master dtype and basis width |
| Direct-CSR recurrent gradients | Trainable per-edge recurrence with a training-engine or FP32-temporal canonical-gradient boundary |
| Packed backward | Qualified architecture, FP16 temporal path, batch32 and supported metadata |
| Native recurrent accumulation | Qualified carrier route, batch/basis/topology and hardware |
| NEST compact coefficients/history/events | Compatible NEST state route and entry points |
| Online voltage penalty | Compatible state route with penalty tracking enabled |
| Device queues/aggregation | Compatible small-batch/four-basis route |

Direct-cell `GradientTape` calls retain canonical master-weight gradients;
automatic CSR-layout gradients and native carriers require the enclosing RNN
training or FP32-temporal boundary that restores canonical ordering.
Inference-only RNNs do not assume a training boundary. Other compatible direct-cell
accelerators remain enabled. Automatic type-indexed NEST dispatch rechecks live
coefficient identity at each rollout. Static dispatch requires explicit opt-in
and its existing immutable-coefficient assumption.

Inspect `rnn.acceleration_report`, `rnn.precision_report` and the runtime log.
Automatic selection accounts for architecture, operator availability and ABI,
not just the CUDA version. Unsupported explicit requests still fail clearly.
General NEST/Pascal execution and unrestricted cross-GPU equivalence are not
claimed by this standard. Fat-binary coverage is not execution qualification.

The device Poisson sampler remains off by default because switching it changes
seeded background realizations. Non-drifting-grating LGN inputs retain their
supported host route, as does initial-state warmup.
Drifting-grating device generation requires a stimulus/run seed and a local
single-worker strategy. Explicitly select `use_device_generation=false` for
unseeded host generation, unsupported strategies or host firing-rate output;
see [LGN input generation](dpointnet_lgn_pipeline.md).

## Existing examples and migration

The DPointNet example configurations now use the standard names while explicitly
retaining previous dynamics, surrogate/credit settings, state precision and output
behavior wherever omitted defaults would otherwise change them. Compatible
automatic acceleration is enabled. Existing explicit execution exceptions remain
visible in the configuration.

Preservation means the same scientific setup and established numerical-equivalence
checks, not guaranteed bit-for-bit equality after acceleration changes floating-point
evaluation order. No validation tolerance is relaxed for the migration.

The helper `bmtk.simulator.dpointnet.migration.preserve_previous_defaults` returns a
copy with pre-standard scientific defaults made explicit and automatic acceleration
selected. Review any migrated file before using it; this helper does not migrate
active jobs, install dependencies or rewrite model data.

For new configs, use standard names directly:

```json
{
  "run": {
    "batch_size": 32,
    "seq_len": 500,
    "dt": 1.0,
    "precision": {
      "mixed_precision": true,
      "fp32_voltage_and_asc": true,
      "fp32_temporal_gradients": false
    }
  },
  "rnn_cell_params": {
    "cell_model": "GLIF3Cell",
    "spike_surrogate": "gaussian",
    "spike_surrogate_width": 0.28,
    "spike_surrogate_gain": 0.05,
    "recurrent_spike_gradient_scale": 1.0,
    "voltage_state_gradient_scale": 1.0
  }
}
```

This fragment is not a complete network/training config. Batch/window values are
explicit examples; supply network components, input groups, objectives and training
duration/LR/optimizer for the actual project.

Use default BFC for standard launchers: unset `TF_GPU_ALLOCATOR` before importing
TensorFlow. The example entrypoints do this explicitly. Importing the library does
not overwrite an application's allocator environment. STG-private memory-growth-off
and NO_MEM_OPT settings are not global V1/library standards.

## Validation and scope

Validation must cover omitted defaults, both configuration interfaces, old/new
alias forward and VJP equivalence, the voltage complement, example preservation,
CPU/Keras compatibility, qualified GPU dispatch and actual accepted optimizer
updates. A bounded initial V1 smoke is sufficient for this migration's requested
training check; it is not a full training or convergence study.

The accompanying validation report records the exact source snapshot, environment,
hardware, test counts, initial updates and remaining qualification boundaries.
Do not interpret checks still in progress as passed.

| Check | Result | Scope |
|---|---|---|
| Complete CUDA-disabled TensorFlow 2.21 / Keras 3 suite | **1,387 passed; 794 skipped** | CPU semantics/orchestration and loaded-library shape contracts; native GPU execution skipped |
| Complete Python 3.8 / TensorFlow 2.13 / Keras 2 suite | **1,351 passed; 830 skipped** | Actual older runtime, not a simulated Keras API |
| RTX3090 native safety preflight | **163 passed; 2 skipped** | Alias/VJP equality, empty topology, canonical direct tapes, live coefficients and distributed masters |
| Frozen RTX3090 focused suite | **330 passed; 2 skipped** | Precision, acceleration and aliases with matching native operators |
| Example configuration audit | **20 preserved** | Exact previous inputs, objectives, protocol and seed-bearing sections |
| Previous-source versus migrated all-to-all example | **Passed** | 300 neurons, batch1, 500 steps; matched inputs, outputs, final state and masters on CPU |
| Precision-switch model checks | **Four profiles passed** | JSON parsing, real construction/forward execution, voltage/ASC and master dtypes |
| Initial full-V1 training updates and strict restore | **Passed** | 66,658 neurons, batch32, 500 steps; three accepted updates, zero scaling rejections |
| Independent checkpoint-finiteness audit | **16 variables passed** | Floating masters, compute shadows and optimizer variables read offline |
| Complete RTX3090 GPU suite | **2,153 passed; 28 skipped** | Full suite on the final immutable source snapshot and matching native operators |

The example rollout uses the existing numerical tolerances (`rtol=2e-5`,
`atol=3e-5`). No tolerance, clipping rule, target, scientific constraint or
precision was relaxed to make a check pass. Other GPU families, physical
multi-GPU execution and scientific convergence are not qualified by these checks.

The V1 smoke verified finite objectives and masters, actual master-weight changes
and an exact model/optimizer checkpoint roundtrip. It exercised omitted new
defaults with automatic acceleration. Its TensorFlow peak was approximately
15.4 GiB on RTX3090, including startup and tracing. Three initial updates establish startup feasibility, not
training convergence, long-run memory safety or a throughput benchmark.
