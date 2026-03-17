"""DCP load entry point.

Mirrors ``torch.distributed.checkpoint.state_dict_loader``.
"""

from .filesystem import FileSystemReader
from .planner import DefaultLoadPlanner


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


def load(
    state_dict,
    *,
    checkpoint_id=None,
    storage_reader=None,
    planner=None,
    process_group=None,
    no_dist=False,
):
    """Load a DCP checkpoint into *state_dict* in-place.

    Args:
        state_dict: mutable mapping of ``{fqn: tensor}`` to populate.
        checkpoint_id: path or identifier for the checkpoint.
        storage_reader: a :class:`StorageReader` instance.
            Defaults to ``FileSystemReader(checkpoint_id)``.
        planner: a :class:`LoadPlanner` instance.
            Defaults to :class:`DefaultLoadPlanner`.
        process_group: distributed process group (or ``None`` for default).
        no_dist: if True, skip all distributed coordination.
    """
    use_dist = (not no_dist) and _dist_available()

    if use_dist:
        rank, world_size = _get_dist_info(process_group)
    else:
        rank, world_size = 0, 1

    is_coordinator = rank == 0

    if storage_reader is None:
        if checkpoint_id is None:
            raise ValueError("checkpoint_id is required when storage_reader is not provided")
        storage_reader = FileSystemReader(checkpoint_id)

    if planner is None:
        planner = DefaultLoadPlanner()

    # 1. Reset
    storage_reader.reset(checkpoint_id)

    # 2. Coordinator reads metadata, broadcast to all
    if is_coordinator:
        metadata = storage_reader.read_metadata()
    else:
        metadata = None

    if use_dist:
        from .. import broadcast_object_list  # pylint: disable=import-outside-toplevel
        bcast = [metadata]
        broadcast_object_list(bcast, src=0, group=process_group)
        metadata = bcast[0]

    # 3. Set up reader and planner
    storage_reader.set_up_storage_reader(metadata, is_coordinator)
    planner.set_up_planner(state_dict, metadata, is_coordinator)

    # 4. Create local plan
    local_plan = planner.create_local_plan()
    local_plan = storage_reader.prepare_local_plan(local_plan)

    # 5. Coordinate plans
    if use_dist:
        from .. import all_gather_object, broadcast_object_list as bcast_obj  # pylint: disable=import-outside-toplevel
        all_plans = [None] * world_size
        all_gather_object(all_plans, local_plan, group=process_group)

        if is_coordinator:
            global_plans = planner.create_global_plan(all_plans)
        else:
            global_plans = None

        bcast2 = [global_plans]
        bcast_obj(bcast2, src=0, group=process_group)
        global_plans = bcast2[0]
        my_plan = global_plans[rank]
    else:
        all_plans = [local_plan]
        global_plans = planner.create_global_plan(all_plans)
        my_plan = global_plans[0]

    # 6. Finalize plan
    final_plan = planner.finish_plan(my_plan)
    final_plan = storage_reader.prepare_global_plan(final_plan)

    # 7. Read data
    storage_reader.read_data(final_plan, planner).wait()

    # 8. Barrier
    if use_dist:
        from .. import barrier  # pylint: disable=import-outside-toplevel
        barrier(group=process_group)
