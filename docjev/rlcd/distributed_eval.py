"""Equal forward counts for collective FSDP evaluation, without duplicate records."""


def evaluation_schedule(rows, rank, world):
    if not rows or world < 1 or not 0 <= rank < world:
        raise ValueError('Invalid collective evaluation inputs')
    for offset in range(0, len(rows), world):
        index = offset + rank
        yield rows[index] if index < len(rows) else rows[0], index < len(rows)


def training_position(length, step, rank, world):
    if length < 1 or step < 0 or world < 1 or not 0 <= rank < world:
        raise ValueError('Invalid training schedule')
    epoch_size = ((length + world - 1) // world) * world
    index = (step * world + rank) % epoch_size
    return (index, False) if index < length else (0, True)
