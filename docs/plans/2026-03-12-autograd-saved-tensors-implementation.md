# Autograd Saved-Tensor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement PyTorch-compatible saved-tensor access (`_saved_*`, `_raw_saved_*`, SavedTensor hooks) to satisfy `test_autograd.py` behaviors.

**Architecture:** Add a richer `SavedTensor` wrapper and `Node` saved-field mapping, wire global saved-tensor hook enable/disable in `autograd.graph`, expose shims under `torch._C._autograd`, and update autograd wrappers to populate common saved fields. Implement via TDD and iterate with `compat/pytorch/run.py --file test_autograd.py -x`.

**Tech Stack:** Python, pytest, candle autograd engine, compat/pytorch test harness.

---

### Task 1: SavedTensor hook semantics (basic)

**Files:**
- Modify: `tests/cpu/test_saved_tensor_hooks.py`
- Modify: `src/candle/autograd/node.py`

**Step 1: Write the failing test**

Add to `tests/cpu/test_saved_tensor_hooks.py`:

```python
def test_saved_tensor_register_hooks_requires_callables():
    x = torch.tensor([1.0], requires_grad=True)
    y = x * x
    raw = y.grad_fn._raw_saved_self
    with pytest.raises(TypeError):
        raw.register_hooks(lambda x: x)
    with pytest.raises(TypeError):
        raw.register_hooks(1, 1)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/cpu/test_saved_tensor_hooks.py::test_saved_tensor_register_hooks_requires_callables -v`
Expected: FAIL because `SavedTensor.register_hooks` missing or wrong validation.

**Step 3: Write minimal implementation**

In `src/candle/autograd/node.py`, implement `SavedTensor.register_hooks` with callable validation and one-time registration guard.

**Step 4: Run test to verify it passes**

Run: `pytest tests/cpu/test_saved_tensor_hooks.py::test_saved_tensor_register_hooks_requires_callables -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/cpu/test_saved_tensor_hooks.py src/candle/autograd/node.py
git commit -m "feat: validate saved tensor hook registration"
```

---

### Task 2: SavedTensor release errors and None forbidden

**Files:**
- Modify: `tests/cpu/test_saved_tensor_hooks.py`
- Modify: `src/candle/autograd/node.py`

**Step 1: Write the failing test**

```python
def test_saved_tensor_none_and_release_errors():
    class Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(None)
            return x
        @staticmethod
        def backward(ctx, g):
            return g

    x = torch.tensor([1.0], requires_grad=True)
    y = Fn.apply(x)
    raw = y.grad_fn._raw_saved_tensors[0]
    with pytest.raises(RuntimeError, match="None is forbidden"):
        raw.register_hooks(lambda x: x, lambda x: x)
    y.sum().backward()
    with pytest.raises(RuntimeError, match="after they have already been freed"):
        _ = y.grad_fn._saved_tensors
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/cpu/test_saved_tensor_hooks.py::test_saved_tensor_none_and_release_errors -v`
Expected: FAIL due to missing error behavior.

**Step 3: Write minimal implementation**

- `SavedTensor.materialize` should raise after release.
- `register_hooks` should error if tensor is None.
- `Node.release_saved_tensors` marks saved tensors as released.

**Step 4: Run test to verify it passes**

Run: `pytest tests/cpu/test_saved_tensor_hooks.py::test_saved_tensor_none_and_release_errors -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/cpu/test_saved_tensor_hooks.py src/candle/autograd/node.py
git commit -m "feat: enforce saved tensor release and None rules"
```

---

### Task 3: Pack hook in-place modification detection

**Files:**
- Modify: `tests/cpu/test_saved_tensor_hooks.py`
- Modify: `src/candle/autograd/node.py`

**Step 1: Write the failing test**

```python
def test_saved_tensor_pack_hook_inplace_modification_raises():
    def pack(x):
        x += 1
        return x

    def unpack(x):
        return x

    x = torch.tensor([1.0], requires_grad=True)
    y = x * x
    raw = y.grad_fn._raw_saved_self
    with pytest.raises(RuntimeError, match="pack hook is modifying"):
        raw.register_hooks(pack, unpack)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/cpu/test_saved_tensor_hooks.py::test_saved_tensor_pack_hook_inplace_modification_raises -v`
Expected: FAIL, missing version check.

**Step 3: Write minimal implementation**

In `SavedTensor.register_hooks`, capture version counter before `pack`, compare after. If changed, raise.

**Step 4: Run test to verify it passes**

Run: `pytest tests/cpu/test_saved_tensor_hooks.py::test_saved_tensor_pack_hook_inplace_modification_raises -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/cpu/test_saved_tensor_hooks.py src/candle/autograd/node.py
git commit -m "feat: detect in-place modifications in pack hooks"
```

---

### Task 4: `_saved_*` and `_raw_saved_*` attribute access

**Files:**
- Modify: `tests/cpu/test_saved_tensor_hooks.py`
- Modify: `src/candle/autograd/node.py`

**Step 1: Write the failing test**

```python
def test_saved_field_accessors():
    x = torch.tensor([1.0], requires_grad=True)
    y = x * x
    self_saved = y.grad_fn._saved_self
    raw_saved = y.grad_fn._raw_saved_self
    assert isinstance(raw_saved, torch._C._autograd.SavedTensor)
    assert torch.allclose(self_saved, x)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/cpu/test_saved_tensor_hooks.py::test_saved_field_accessors -v`
Expected: FAIL because `_saved_self`/`_raw_saved_self` not implemented.

**Step 3: Write minimal implementation**

- Add `_saved_fields` to `Node` and implement `__getattr__` to resolve `_saved_*` and `_raw_saved_*`.
- Populate `_saved_fields` in autograd wrappers for binary ops with `self`/`other` keys.

**Step 4: Run test to verify it passes**

Run: `pytest tests/cpu/test_saved_tensor_hooks.py::test_saved_field_accessors -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/cpu/test_saved_tensor_hooks.py src/candle/autograd/node.py src/candle/_backends/autograd.py
git commit -m "feat: expose saved tensor fields via grad_fn"
```

---

### Task 5: Global saved_tensors_hooks enable/disable

**Files:**
- Modify: `tests/cpu/test_saved_tensor_hooks.py`
- Modify: `src/candle/autograd/graph.py`
- Modify: `src/candle/_C.py`

**Step 1: Write the failing test**

```python
def test_disable_saved_tensors_hooks_blocks_registration():
    with torch.autograd.graph.disable_saved_tensors_hooks("blocked"):
        with pytest.raises(RuntimeError, match="blocked"):
            with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
                pass
    assert torch._C._autograd._saved_tensors_hooks_is_enabled()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/cpu/test_saved_tensor_hooks.py::test_disable_saved_tensors_hooks_blocks_registration -v`
Expected: FAIL due to missing API.

**Step 3: Write minimal implementation**

- Add `disable_saved_tensors_hooks` context in `autograd.graph`.
- Track enabled flag; make `saved_tensors_hooks` raise when disabled.
- Expose `_saved_tensors_hooks_is_enabled()` under `torch._C._autograd`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/cpu/test_saved_tensor_hooks.py::test_disable_saved_tensors_hooks_blocks_registration -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/cpu/test_saved_tensor_hooks.py src/candle/autograd/graph.py src/candle/_C.py
git commit -m "feat: add saved tensor hooks enable/disable shims"
```

---

### Task 6: Populate `_saved_result` for outputs and minimal op mappings

**Files:**
- Modify: `tests/cpu/test_saved_tensor_hooks.py`
- Modify: `src/candle/_backends/autograd.py`

**Step 1: Write the failing test**

```python
def test_saved_result_for_exp():
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.exp(x)
    assert torch.allclose(y.grad_fn._saved_result, y)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/cpu/test_saved_tensor_hooks.py::test_saved_result_for_exp -v`
Expected: FAIL because `_saved_result` not populated.

**Step 3: Write minimal implementation**

- In autograd unary wrapper, add optional `saved_result=True` for ops like `exp`, `tanh`.
- Set `_saved_fields["result"] = output` and save raw saved tensor if needed.

**Step 4: Run test to verify it passes**

Run: `pytest tests/cpu/test_saved_tensor_hooks.py::test_saved_result_for_exp -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/cpu/test_saved_tensor_hooks.py src/candle/_backends/autograd.py
git commit -m "feat: store saved_result for select ops"
```

---

### Task 7: Drive from PyTorch test_autograd

**Files:**
- Modify: as needed per failures
- Test: `compat/pytorch/_pytorch/test/test_autograd.py` via compat runner

**Step 1: Run test to capture next failure**

Run: `python compat/pytorch/run.py --file test_autograd.py -x -vv --maxfail=1`
Expected: next failing assertion for missing saved-field mapping or hook behavior.

**Step 2: Add failing unit test**

Create a minimal reproduction in `tests/cpu/test_saved_tensor_hooks.py`.

**Step 3: Implement minimal fix**

Update autograd wrappers, `SavedTensor`, or shims as needed.

**Step 4: Verify**

Run the targeted unit test then re-run the compat test with `-x`.

**Step 5: Commit**

```bash
git add tests/cpu/test_saved_tensor_hooks.py src/candle/... \
git commit -m "fix: align saved tensor behavior for <case>"
```
