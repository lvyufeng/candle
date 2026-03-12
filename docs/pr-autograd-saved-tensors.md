## Summary
- Add PyTorch-compatible saved-tensor hook registration baseline and saved field accessors
- Document autograd saved-tensor design and implementation plan

## Testing
- pytest tests/cpu/test_saved_tensor_hooks.py::test_saved_tensor_register_hooks_requires_callables -v
