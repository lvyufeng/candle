"""4-card HCCL DDP training gate for sprint P0 validation."""

import os
import subprocess
import sys
import time

import pytest


SCRIPT = r'''
import os
import sys
import time

src_dir = os.environ.get("CANDLE_SRC")
if src_dir:
    sys.path.insert(0, src_dir)

import candle as torch
import candle.nn as nn
import candle.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

device = torch.Device(f"npu:{rank}")
time.sleep(0.05 * rank)
dist.init_process_group("hccl", device_id=device)

# Deterministic single-layer model and optimizer.
model = nn.Linear(8, 4).to(device)
with torch.no_grad():
    model.weight.fill_(0.01)
    model.bias.fill_(0.0)

ddp = nn.parallel.DistributedDataParallel(model)
optim = torch.optim.SGD(ddp.parameters(), lr=0.1)

for step in range(5):
    x = torch.ones((6, 8), device=device, dtype=torch.float32)
    y = ddp(x)
    loss = y.sum()

    optim.zero_grad(set_to_none=False)
    loss.backward()
    optim.step()

    # Rank-local scalar signal for step progress.
    marker = torch.tensor([float(step + 1)], device=device)
    dist.all_reduce(marker, op=dist.ReduceOp.SUM)
    assert float(marker.to("cpu").item()) == float(world_size * (step + 1))

# Validate final parameter consistency against rank0 snapshot.
w_ref = model.weight.detach().clone()
b_ref = model.bias.detach().clone()

dist.broadcast(w_ref, src=0)
dist.broadcast(b_ref, src=0)

w_diff = (model.weight.detach() - w_ref).abs().sum().to("cpu").item()
b_diff = (model.bias.detach() - b_ref).abs().sum().to("cpu").item()
assert w_diff < 1e-5, f"rank={rank} weight mismatch {w_diff}"
assert b_diff < 1e-5, f"rank={rank} bias mismatch {b_diff}"

dist.barrier()
dist.destroy_process_group()
print(f"[rank {rank}] HCCL 4-card DDP training gate PASS")
'''


def _run_once(master_port):
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(master_port)
    env["WORLD_SIZE"] = "4"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["CANDLE_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

    worker = f"/tmp/_hccl_ddp_4card_training_gate_{master_port}.py"
    with open(worker, "w") as f:
        f.write(SCRIPT)

    procs = []
    for rank in range(4):
        env_rank = {**env, "RANK": str(rank)}
        procs.append(
            subprocess.Popen(
                [sys.executable, worker],
                env=env_rank,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        )

    outputs = []
    failed = []
    for rank, proc in enumerate(procs):
        try:
            out, _ = proc.communicate(timeout=420)
            text = out.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            text = "TIMEOUT\n" + out.decode("utf-8", errors="replace")
        outputs.append((rank, proc.returncode, text))
        if proc.returncode != 0:
            failed.append(rank)

    return failed, outputs


def _run_case(master_port):
    retries = 2
    for attempt in range(1, retries + 1):
        failed, outputs = _run_once(master_port + attempt - 1)
        if not failed:
            return

        joined = "\n".join(text for _, _, text in outputs)
        transient = "resource unavailable" in joined.lower()
        if transient and attempt < retries:
            time.sleep(5)
            continue

        details = "\n\n".join(
            f"=== rank {rank} exit={code} ===\n{text}" for rank, code, text in outputs
        )
        raise AssertionError(f"4-card DDP gate failed on ranks {failed}:\n{details}")


def test_hccl_ddp_4card_training_gate():
    _run_case(master_port=29770)
