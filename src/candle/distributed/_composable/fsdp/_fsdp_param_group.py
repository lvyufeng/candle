"""FSDPParamGroup -- batched communication for parameter groups."""
from .... import distributed as dist


class FSDPParamGroup:
    """Manages a group of FSDPParam instances with batched lifecycle.

    When ``use_flat_buffer=True`` (the default for world_size > 1),
    all local shards are packed into a single contiguous flat buffer so
    that unshard / reduce-scatter each require only **one** collective
    instead of one per parameter.
    """

    def __init__(self, fsdp_params, module, mesh_info):
        self.fsdp_params = fsdp_params
        self.module = module
        self.mesh_info = mesh_info
        self._is_unsharded = False

        world_size = mesh_info.shard_mesh_size
        self._use_flat_buffer = world_size > 1 and len(fsdp_params) > 1

        if self._use_flat_buffer:
            self._init_flat_buffer()

    # ------------------------------------------------------------------
    # Flat-buffer initialisation
    # ------------------------------------------------------------------

    def _init_flat_buffer(self):
        """Pack all local shards into one flat buffer and record offsets."""
        from ...._creation import zeros

        self._shard_offsets = []
        self._shard_shapes = []
        total_numel = 0
        for fp in self.fsdp_params:
            local = fp._sharded_param.to_local()
            n = local.numel()
            self._shard_offsets.append((total_numel, total_numel + n))
            self._shard_shapes.append(local.shape)
            total_numel += n

        first_local = self.fsdp_params[0]._sharded_param.to_local()
        dtype = first_local.dtype
        device = first_local.device
        self._total_shard_numel = total_numel

        # Flat shard buffer (one per rank)
        self._flat_shard = zeros(total_numel, dtype=dtype, device=device)
        self._copy_shards_to_flat()

        # Flat full buffer (world_size * shard_numel)
        world_size = self.mesh_info.shard_mesh_size
        self._flat_full = zeros(
            total_numel * world_size, dtype=dtype, device=device
        )

    def _copy_shards_to_flat(self):
        """Copy each param's local shard into the flat buffer."""
        for fp, (start, end) in zip(self.fsdp_params, self._shard_offsets):
            local = fp._sharded_param.to_local()
            self._flat_shard[start:end] = local.reshape(-1)

    # ------------------------------------------------------------------
    # Batched shard lifecycle
    # ------------------------------------------------------------------

    def unshard(self):
        """Unshard all parameters in the group."""
        if self._is_unsharded:
            return
        if self._use_flat_buffer:
            self._unshard_flat()
        else:
            for p in self.fsdp_params:
                p.unshard()
        self._is_unsharded = True

    def _unshard_flat(self):
        """Single all-gather on the flat buffer, then scatter to params."""
        self._copy_shards_to_flat()
        pg = self.mesh_info.shard_process_group
        dist.all_gather_into_tensor(self._flat_full, self._flat_shard, group=pg)

        # Scatter gathered data back to individual params
        world_size = self.mesh_info.shard_mesh_size
        for i, fp in enumerate(self.fsdp_params):
            start, end = self._shard_offsets[i]
            numel = end - start
            # Reconstruct full param: gather chunks from each rank's region
            chunks = []
            for rank in range(world_size):
                rank_offset = rank * self._total_shard_numel
                chunk = self._flat_full[rank_offset + start:rank_offset + end]
                chunks.append(chunk)
            from ...._functional import cat, narrow
            full_flat = cat(chunks, dim=0)
            # Reshape to full param shape
            full_shape = list(fp._orig_shape)
            full_param = full_flat.reshape(*full_shape)
            # Strip padding if needed
            if fp._padded_dim_size > 0:
                orig_dim = fp._orig_shape[fp._shard_dim]
                full_param = narrow(full_param, fp._shard_dim, 0, orig_dim)
            full_param.requires_grad = fp._sharded_param.requires_grad
            fp._unsharded_param = full_param
            fp._set_param_on_module(full_param)
            fp._sharded_state = fp._sharded_state.__class__.UNSHARDED

    def reshard(self):
        """Reshard all parameters in the group."""
        if not self._is_unsharded:
            return
        for p in self.fsdp_params:
            p.reshard()
        self._is_unsharded = False

    def reduce_scatter_grads(self):
        """Reduce-scatter gradients for all parameters in the group."""
        if self._use_flat_buffer:
            self._reduce_scatter_flat()
        else:
            for p in self.fsdp_params:
                p.reduce_scatter_grad()

    def _reduce_scatter_flat(self):
        """Single reduce-scatter on flat gradient buffer."""
        from ...._creation import zeros
        from ...._functional import cat

        world_size = self.mesh_info.shard_mesh_size

        # Pack all unsharded grads into flat_full layout
        self._flat_full[:] = 0
        for i, fp in enumerate(self.fsdp_params):
            if fp._unsharded_param is None:
                continue
            grad = fp._unsharded_param.grad
            if grad is None:
                continue
            start, end = self._shard_offsets[i]
            numel = end - start
            # Pad grad if needed
            if fp._padded_dim_size > 0:
                pad_shape = list(grad.shape)
                pad_shape[fp._shard_dim] = fp._padded_dim_size
                grad = cat(
                    [grad, zeros(*pad_shape, dtype=grad.dtype, device=grad.device)],
                    dim=fp._shard_dim,
                )
            grad_flat = grad.reshape(-1)
            # Distribute into rank-strided layout
            for rank in range(world_size):
                rank_offset = rank * self._total_shard_numel
                chunk = grad_flat[rank * numel:(rank + 1) * numel]
                self._flat_full[rank_offset + start:rank_offset + end] = chunk

        # Single reduce-scatter
        pg = self.mesh_info.shard_process_group
        reduced_flat = zeros(
            self._total_shard_numel,
            dtype=self._flat_shard.dtype,
            device=self._flat_shard.device,
        )
        dist.reduce_scatter_tensor(reduced_flat, self._flat_full, group=pg)

        # Scatter reduced shards back to params
        for i, fp in enumerate(self.fsdp_params):
            start, end = self._shard_offsets[i]
            shape = self._shard_shapes[i]
            fp._sharded_param.to_local().grad = reduced_flat[start:end].reshape(*shape)

    # ------------------------------------------------------------------
    # Module hook helpers
    # ------------------------------------------------------------------

    def pre_forward(self):
        """Call before the module's forward pass."""
        self.unshard()

    def post_forward(self):
        """Call after the module's forward pass."""
        self.reshard()

    def pre_backward(self):
        """Call before the module's backward pass."""
        self.unshard()

    def post_backward(self):
        """Call after the module's backward pass."""
        self.reduce_scatter_grads()
        self.reshard()
