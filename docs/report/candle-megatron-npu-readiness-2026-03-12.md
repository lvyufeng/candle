# Candle → Megatron-LM (NPU, No Megatron Modifications) Readiness Plan

> Date: 2026-03-12
> Goal: Run Megatron-LM main on Ascend NPU with **no Megatron code changes** and **no external packages installed**.
> Assumptions: Megatron-LM main expects CUDA/NCCL semantics; Candle must emulate these on top of NPU/HCCL.

---

## 1. Blocking Gaps (Import & Initialization)

These must be fixed before Megatron even starts:

- **CUDA facade over NPU**
  - Provide a `torch.cuda`-compatible API that maps to NPU.
  - Required: `is_available`, `device_count`, `current_device`, `set_device`, `synchronize`, `Stream`, `Event`, RNG state, `amp` entrypoints.
  - Without this, Megatron fails at import and device setup.

- **NCCL backend semantics over HCCL**
  - `init_process_group(backend="nccl")` must succeed and map to HCCL.
  - Provide `ProcessGroupNCCL`-compatible behavior or a transparent adapter.

- **Tensor parallel / DTensor API must not be stubbed**
  - `torch.distributed.tensor.parallel` and `torch.distributed._tensor` are currently stubs.
  - Megatron imports these modules; stubs cause immediate failure.

---

## 2. Distributed & Parallelism (Full Training)

Megatron requires a full distributed surface beyond basic DDP:

- **Collectives**
  - Must support: `all_reduce`, `all_gather`, `all_gather_into_tensor`, `reduce_scatter`, `reduce_scatter_tensor`, `broadcast`, `barrier`, `send/recv`, `isend/irecv`, `batch_isend_irecv`.

- **Pipeline / tensor parallel utilities**
  - `torch.distributed.tensor.parallel.parallelize_module`
  - `SequenceParallel`
  - Minimal DTensor placement semantics so Megatron sharding logic runs.

- **ProcessGroup features**
  - Async work handles, error propagation, and timeouts compatible with Megatron's launch flow.

---

## 3. Checkpointing (Pretrain + Resume)

- **`torch.distributed.checkpoint.stateful` must be functional**
  - Current Candle implementation is stubbed.
  - Required to save/load sharded checkpoints at scale.

- **State dict compatibility**
  - Ensure `state_dict` / `load_state_dict` works under sharded and parallel contexts.

---

## 4. Kernel/Op Coverage (Megatron Training Path)

Megatron relies on a specific set of ops with correct autograd:

- **Core transformer ops**
  - `layer_norm` / `rms_norm`
  - `softmax` / `log_softmax`
  - `dropout`
  - `gelu` / `silu`
  - `matmul` / `bmm`
  - `view` / `transpose` / `reshape`

- **Attention stack**
  - scaled dot-product attention path
  - rotary embedding ops

- **Autograd completeness**
  - Backward for all training-critical ops must exist.

---

## 5. External Dependency Shims (No Packages Installed)

Megatron imports these by default; Candle must provide shims:

- `apex`
- `transformer_engine`
- `flash_attn`

Shims must provide minimal APIs so imports succeed and training runs with Candle ops.

---

## 6. Verification Gates

- **Import gate**: Megatron imports without error under Candle.
- **Init gate**: `init_process_group(backend="nccl")` succeeds and sets up multi-node.
- **Training gate**: single-iteration pretrain forward/backward + optimizer step.
- **Checkpoint gate**: save + reload succeeds in multi-node context.
- **Stability gate**: full pretrain run with deterministic loss (within tolerance).

---

## 7. Phase Plan (P0 / P1 / P2)

### P0 (Run Megatron without code changes)
- Implement CUDA facade over NPU.
- NCCL-to-HCCL backend adapter.
- Tensor-parallel/DTensor APIs (minimal but real, not stub).
- Import shims for `apex`, `transformer_engine`, `flash_attn`.

### P1 (Full Training Features)
- Distributed checkpoint implementation.
- Autograd completeness for transformer ops.
- Pipeline parallel + sequence parallel correctness.

### P2 (Performance & Stability)
- Host round-trip elimination in NPU hot paths.
- Kernel fusion and memory optimizations.
- Benchmark parity vs torch+npu.

---

## Summary

Running Megatron-LM main on NPU with **no Megatron modifications** requires Candle to emulate CUDA/NCCL at the API level, provide non-stub tensor-parallel APIs, supply checkpointing support, and shim key external dependencies. The plan above orders those gaps so Megatron can import, initialize, and train end-to-end.
