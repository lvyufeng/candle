"""DCP save entry point.

Mirrors ``torch.distributed.checkpoint.state_dict_saver``.
"""

from .filesystem import FileSystemWriter
from .metadata import Metadata, StorageMeta
from .planner import DefaultSavePlanner
from .storage import StorageWriter


def _dist_available():
    """Return True if distributed is initialized."""
    try:
        from .. import is_initialized  # pylint: disable=import-outside-toplevel
        return is_initialized()
    except Exception:  # pylint: disable=broad-except
        return False


def _get_dist_info(process_group):
    """Return (rank, world_size) from the process group or defaults."""
    try:
        from .. import get_rank, get_world_size  # pylint: disable=import-outside-toplevel
        return get_rank(process_group), get_world_size(process_group)
    except Exception:  # pylint: disable=broad-except
        return 0, 1


def save(
    state_dict,
    *,
    checkpoint_id=None,
    storage_writer=None,
    planner=None,
    process_group=None,
    no_dist=False,
):
    """Save *state_dict* using the DCP protocol.

    Args:
        state_dict: mapping of ``{fqn: tensor}`` to save.
        checkpoint_id: path or identifier for the checkpoint.
        storage_writer: a :class:`StorageWriter` instance.
            Defaults to ``FileSystemWriter(checkpoint_id)``.
        planner: a :class:`SavePlanner` instance.
            Defaults to :class:`DefaultSavePlanner`.
        process_group: distributed process group (or ``None`` for default).
        no_dist: if True, skip all distributed coordination.

    Returns:
        :class:`Metadata` written to the checkpoint.
    """
    use_dist = (not no_dist) and _dist_available()

    if use_dist:
        rank, world_size = _get_dist_info(process_group)
    else:
        rank, world_size = 0, 1

    is_coordinator = rank == 0

    if storage_writer is None:
        if checkpoint_id is None:
            raise ValueError("checkpoint_id is required when storage_writer is not provided")
        storage_writer = FileSystemWriter(checkpoint_id)

    if planner is None:
        planner = DefaultSavePlanner()

    # 1. Reset & set up
    storage_writer.reset(checkpoint_id)
    storage_writer.set_up_storage_writer(is_coordinator)
    if hasattr(storage_writer, "set_rank"):
        storage_writer.set_rank(rank)

    # 2. Planner creates local plan
    planner.set_up_planner(state_dict, is_coordinator)
    local_plan = planner.create_local_plan()

    # 3. Storage adjusts local plan
    local_plan = storage_writer.prepare_local_plan(local_plan)

    # 4. Coordinate plans across ranks
    if use_dist:
        from .. import all_gather_object, broadcast_object_list  # pylint: disable=import-outside-toplevel
        all_plans = [None] * world_size
        all_gather_object(all_plans, local_plan, group=process_group)
    else:
        all_plans = [local_plan]

    # 5. Coordinator creates global plan + metadata
    if is_coordinator:
        global_plans, metadata = planner.create_global_plan(all_plans)
    else:
        global_plans, metadata = None, None

    # 6. Broadcast global plans to all ranks
    if use_dist:
        bcast_list = [global_plans]
        broadcast_object_list(bcast_list, src=0, group=process_group)
        global_plans = bcast_list[0]

    # 7. Each rank gets its own plan
    my_plan = global_plans[rank]
    final_plan = planner.finish_plan(my_plan)
    final_plan = storage_writer.prepare_global_plan([final_plan])[0] \
        if isinstance(storage_writer.prepare_global_plan([final_plan]), list) \
        else storage_writer.prepare_global_plan(final_plan)

    # 8. Write data
    write_results = storage_writer.write_data(final_plan, planner).wait()

    # 9. Gather results
    if use_dist:
        all_results = [None] * world_size
        all_gather_object(all_results, write_results, group=process_group)
    else:
        all_results = [write_results]

    # 10. Coordinator writes .metadata
    if is_coordinator:
        storage_meta = storage_writer.storage_meta()
        if storage_meta is not None:
            metadata.storage_meta = storage_meta
        storage_writer.finish(metadata, all_results)

    # 11. Barrier
    if use_dist:
        from .. import barrier  # pylint: disable=import-outside-toplevel
        barrier(group=process_group)

    return metadata
