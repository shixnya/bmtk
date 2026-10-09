# NEST-compatible DPointNet: reset and precision

This small example compares the **same 300-neuron network and saved input
spikes** in DPointNet and actual NEST. It separates the effects of changing
reset policy and numerical precision instead of changing both at once.

| Configuration | Computation | Voltage/ASC | Reset | Purpose |
| --- | --- | --- | --- | --- |
| `config.nest.json` | NEST double precision | NEST double precision | Hard | Actual NEST reference through PointNet |
| `config.fp32.hard.json` | FP32 | FP32 | Hard | Closest DPointNet comparison to NEST |
| `config.fp32.soft.json` | FP32 | FP32 | Soft | Isolate reset-policy effects at FP32 |
| `config.mixed.hard.json` | Mixed FP16 | FP32 | Hard | Isolate precision effects at hard reset |
| `config.mixed.soft.json` | Mixed FP16 | FP32 | Soft | Standard training-like precision/reset combination |

All DPointNet profiles use `dynamics_mode="nest"` and
`acceleration_profile="auto"`. **DPointNet is still running in TensorFlow;
it is not invoking NEST.** The separate PointNet script runs NEST itself.
NEST uses internal double precision; the FP32 arm is not claiming NEST
also runs in FP32.

## What is held fixed

- 200 excitatory and 100 inhibitory GLIF3 neurons, with after-spike currents.
- One seeded network realization, recurrent weights and 100 virtual inputs.
- The same saved grid-aligned input events in every run, without resampling.
- A 1-ms timestep, 300-ms duration, grid-aligned refractory periods and delays.
- Exact alpha synapses with the same time constants in both implementations.
  DPointNet uses one-hot basis coefficients, so no waveform-fitting error is
  mixed into the precision/reset comparison.
- Initial rest state and fixed weights; no optimizer updates.

Membrane and ASC parameters are adapted from `../dpointnet_all2all`; the
refractory periods and synaptic time constants are explicitly chosen for this
comparison. This is a toy circuit, not a biological fitting recipe.
The saved inputs are seeded grid Bernoulli events, not a substituted Poisson
background generator.

Hard reset assigns the reset voltage after a spike and holds it during the
refractory interval. Soft reset subtracts the threshold-to-reset displacement
and retains the voltage overshoot; its subsequent refractory dynamics also
differ. It is a model change, not just a faster implementation.

## Build and run

Use a writable copy of this directory. As in other BMTK examples, run commands
with this directory as the working directory. Generated network, inputs and
outputs are ignored by Git.

First build the assets using Python with BMTK:

```bash
python build_network.py
```

Then run each DPointNet configuration in a **separate Python process**:

```bash
CUDA_VISIBLE_DEVICES="" python run_dpointnet.py config.fp32.hard.json
CUDA_VISIBLE_DEVICES="" python run_dpointnet.py config.fp32.soft.json
CUDA_VISIBLE_DEVICES="" python run_dpointnet.py config.mixed.hard.json
CUDA_VISIBLE_DEVICES="" python run_dpointnet.py config.mixed.soft.json
```

The four profile JSON files are **overrides of `config.base.json`**.
`run_dpointnet.py` recursively merges them before passing the complete
configuration to BMTK. They are not standalone configurations for another
generic launcher. The base includes the shared network, input and inference
settings; the profiles change only precision, reset and output directory.

Mixed FP16 can run on CPU for this small example. CPU auto routing uses
compatible TensorFlow implementations; it does not exercise CUDA kernels.
To run on a GPU, omit `CUDA_VISIBLE_DEVICES=""` in the intended GPU environment.
Matching optional custom operators are needed to exercise fused CUDA paths;
`auto` selects only compatible paths. Check the actual resolved report rather
than assuming a profile name proves GPU execution. Results can differ with
floating-point reduction order across devices.

If NEST and BMTK PointNet are installed, run the actual reference:

```bash
python run_pointnet.py config.nest.json
```

You may use a different Python environment for NEST and TensorFlow. Keep the
working directory and built assets identical. PointNet uses receptor-only
synaptic component files in `components/nest_synaptic_models`; DPointNet uses
the equivalent one-hot basis components in `components/synaptic_models`.

Finally, plot the runs:

```bash
MPLBACKEND=Agg python plot_output.py
```

Without a NEST installation, explicitly compare only DPointNet runs:

```bash
MPLBACKEND=Agg python plot_output.py --without-nest
```

That command uses FP32/hard as its reference; it is **not** a NEST validation.

## Outputs and interpretation

Each `output_<profile>/` contains spikes, DPointNet voltages where applicable,
logs and `resolved.json`. The report records dtypes, acceleration selection,
runtime version and hashes of the shared assets. The plotting script rejects
different inputs, timesteps, out-of-grid events and non-finite voltages.
Do not rebuild the network between arms.

`output_comparison/` contains:

- `comparison.png`: separate rasters, a representative neuron voltage trace
  and cumulative event disagreements.
- `comparison.csv`: total spikes, mean rates and neuron/time-bin agreement
  against NEST (or explicitly FP32/hard without NEST).
- `pairwise.csv`: reset effects at fixed precision and precision effects at
  fixed reset, including normalized-voltage RMS differences.
- `comparison.json`: the measurements and resolved runtime reports.

Spike agreement uses the original end-of-step timestamps, with **no time shift
or nearest-spike realignment**. Event F1 is twice the matched neuron/timestep
events divided by the total events in both runs. Small timing changes reduce
event agreement even when firing rates remain close. In recurrent networks,
an early threshold crossing difference can affect later activity.
Voltage plots are in DPointNet's normalized units: threshold is one, reset is
zero, and these are post-reset traces, not millivolts or pre-reset overshoots.

With seed 3000, NEST 3.8 and TensorFlow 2.21 on CPU, the measured example was:

| Profile | Spikes | Event F1 vs NEST |
| --- | ---: | ---: |
| NEST | 2397 | 1.0000 |
| FP32/hard | 2397 | 1.0000 |
| FP32/soft | 2493 | 0.1943 |
| Mixed/hard | 2380 | 0.7323 |
| Mixed/soft | 2513 | 0.1988 |

These are illustrative results for this seed and environment, not enforced
accuracy thresholds or a general spike-identity guarantee.
The pairs in `pairwise.csv`, rather than only comparison with NEST, show which
changes are due to reset and which are due to precision.

## Relation to training and inference

The mixed/soft arm uses the standard GLIF forward precision and reset policy,
but deliberately **does not train**. Holding weights fixed is necessary to
attribute differences to precision and reset rather than optimizer updates.
The spike surrogate and temporal-gradient settings affect training credit;
this forward-only comparison does not test them.

Training rejects hard reset. A model trained with soft reset should normally
retain soft reset for inference, even when FP32 inference is requested.
Switching its reset to hard changes the model and needs a separate evaluation;
closer agreement with NEST does not make it the same trained model.

The documented replay path uses a 1-ms grid. Do not simply change `dt` in
these configs: the current generic spike-file iterator does not propagate
the model timestep to the input generator. A different timestep needs an
explicitly timestep-aware input generator as well as regenerated input,
refractory and delay checks.
