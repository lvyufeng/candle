# Install Candle On Ascend NPU

## Scope

This guide documents the supported Candle `0.1.x` NPU path:

- SoC: Ascend 910B
- Topology: single-card only
- Validation target: local golden training loop, checkpoint continuity, and no-CPU-fallback regression gates

Everything outside that path remains experimental unless called out explicitly in [support-matrix.md](support-matrix.md).

## Prerequisites

Before installing Candle for NPU, make sure all of the following are true:

- CANN is installed on the host.
- If multiple CANN versions are installed, `/usr/local/Ascend/ascend-toolkit/latest` points to the newest installed version.
- The newest toolkit contains:
  - `/usr/local/Ascend/ascend-toolkit/latest/lib64`
  - `/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64`
  - `/usr/local/Ascend/ascend-toolkit/latest/opp`
  - `/usr/local/Ascend/ascend-toolkit/latest/python/site-packages`
- Python can import Candle and the Ascend Python packages in the same environment.

Candle's NPU loader aligns itself to `/usr/local/Ascend/ascend-toolkit/latest`, but you should still source the toolkit environment explicitly so the shell, Python path, and runtime libraries agree on one CANN version.

## Environment Setup

Start from a fresh shell. If your CANN installation provides `set_env.sh`, source it first:

```bash
if [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi
```

Then make the runtime paths explicit:

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:${PWD}/src:${PYTHONPATH:-}
```

If `set_env.sh` is not present on your host, you still need the exports above before importing Candle's NPU backend.

## Install Candle

For the current source workflow:

```bash
git clone https://github.com/candle-org/candle.git
cd candle
pip install -e ".[test]"
```

This installs Candle plus the test dependencies used by the 0.1 NPU validation path.

## Verify NPU Availability

Run a direct availability probe before running model code:

```bash
python - <<'PY'
import candle as torch

print('npu available:', torch.npu.is_available(verbose=True))
if torch.npu.is_available():
    print('device count:', torch.npu.device_count())
    print('device name:', torch.npu.get_device_name())
PY
```

Expected:

- `npu available: True`
- `device count` is at least `1`
- `device name` reports an Ascend 910B device on the supported GA path

If availability is `False`, keep `verbose=True` and fix the missing ACL/CANN import or library path issue before continuing.

## Minimal NPU Quick Start

Once availability is green, run a minimal device computation:

```bash
python - <<'PY'
import candle as torch

assert torch.npu.is_available(verbose=True)
x = torch.randn((4, 8), device='npu')
w = torch.randn((8, 2), device='npu', requires_grad=True)
y = torch.matmul(x, w)
y.sum().backward()
print('device:', y.device)
print('soc:', torch.npu.get_device_name())
PY
```

Expected:

- tensors stay on `npu`
- backward completes on-device
- no implicit CPU fallback is required for this path

## Run The 0.1 Local NPU Gates

The declared 0.1 NPU validation set is:

```bash
PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:src \
pytest \
  tests/npu/test_npu_golden_training_loop.py \
  tests/npu/test_npu_training_checkpoint_continuity.py \
  tests/npu/test_mul_scalar_regression_npu.py \
  tests/npu/test_no_cpu_fallback_npu.py \
  -v --tb=short
```

Expected:

- all tests pass on a supported Ascend 910B host
- the golden training loop stays finite and decreases loss
- checkpoint save/load continues training after reload
- declared NPU release paths do not round-trip through CPU

## Troubleshooting

### `torch.npu.is_available(verbose=True)` reports missing `acl`

Your Ascend Python packages are not visible to the active interpreter. Re-check:

- `source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh`
- `PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:${PWD}/src:${PYTHONPATH:-}`

### Runtime libraries fail to load

Re-check:

- `LD_LIBRARY_PATH` includes both toolkit runtime directories
- `/usr/local/Ascend/ascend-toolkit/latest` points to the newest installed CANN version
- `ASCEND_OPP_PATH` points to `/usr/local/Ascend/ascend-toolkit/latest/opp`

### Tests pass on CPU but not on NPU

That is not an acceptable release workaround. Candle's NPU GA path must execute on NPU without runtime fallback to CPU.
