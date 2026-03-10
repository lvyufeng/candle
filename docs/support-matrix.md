# Candle 0.1 Support Matrix (NPU-First)

This matrix defines the effective support scope for Candle `0.1.x`.

For NPU installation and runtime prerequisites, see [install-npu.md](install-npu.md).

## GA

- Ascend 910B single-card training core path on a host where the newest installed CANN toolkit is exposed through `/usr/local/Ascend/ascend-toolkit/latest` and sourced into the shell.
  - tensor creation,
  - forward ops used by baseline training,
  - autograd backward,
  - optimizer step,
  - checkpoint save/load on the critical path.
- CPU backend for development and CI baseline.

## Experimental

- Additional NPU ops outside the baseline training path.
- Transformers compatibility runner (`tests/run_test.py`) and related patches.
- Partial distributed/HCCL collectives where behavior is validated only in limited scenarios.

## Not Supported In 0.1 Scope

- Distributed high-level features:
  - DeviceMesh,
  - Tensor Parallel,
  - FSDP and composable FSDP.
- True JIT/compile acceleration backends.
- Full ONNX export compatibility guarantees.

## Validation Gates

0.1 release quality is evaluated by:

- local Ascend 910B NPU gate set:
  - `tests/npu/test_npu_golden_training_loop.py`,
  - `tests/npu/test_npu_training_checkpoint_continuity.py`,
  - `tests/npu/test_mul_scalar_regression_npu.py`,
  - `tests/npu/test_no_cpu_fallback_npu.py`.
- CPU + contract CI baseline.
- MPS CI baseline for cross-backend regression visibility.

## Runtime Rule

- NPU execution paths must remain on NPU.
- Runtime fallback from NPU kernels to CPU is not allowed.
