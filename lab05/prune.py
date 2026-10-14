from __future__ import annotations

from typing import Any, Sequence

from tensors import (
    INDEX_BYTES,
    Tensor,
    TensorError,
    dtype_bytes,
    total_parameters,
)

# The four meanings of "removed", Lecture 05 slide 8, in staircase order.
REMOVALS = ("masked", "patterned", "stored sparse", "structurally absent")

# How a tensor is written to disk. Three forms, and `classify_removal` reads
# this alongside the shapes to decide which of the four it got.
STORAGE_FORMS = ("dense", "masked", "sparse")

# What each granularity produces if you implement it the obvious way. Fine
# grained pruning in a framework leaves a mask behind; channel pruning leaves a
# smaller dense tensor. `sweep_model` uses this unless told otherwise, and
# `sweep.py --storage` is how you tell it otherwise.
GRANULARITY_STORAGE = {"fine": "masked", "channel": "dense"}

GRANULARITIES = tuple(GRANULARITY_STORAGE)

# A layer with no output channels is not a smaller layer, it is a broken one.
MIN_CHANNELS = 1

# N:M, for `classify_removal`. 2:4 is the case Ampere implements and the only
# one Lecture 05 slide 18 draws.
NM_N, NM_M = 2, 4


def _drop_count(n: int, ratio: float) -> int:
    if not 0.0 <= ratio <= 1.0:
        raise TensorError(f"ratio {ratio} is not in [0, 1]")
    return max(0, min(int(n), int(int(n) * float(ratio) + 0.5)))


def _smallest_indices(values: Sequence[float], k: int) -> tuple[int, ...]:
    if k <= 0:
        return ()
    order = sorted(range(len(values)), key=lambda i: (abs(values[i]), i))
    return tuple(sorted(order[:k]))


def _group_scores(t: Tensor, p: float = 2.0) -> tuple[float, ...]:
    if p <= 0:
        raise TensorError(f"p must be positive, got {p}")
    scores = []
    for c in range(t.channels):
        lo, hi = _channel_slice(t, c)
        acc = 0.0
        for v in t.data[lo:hi]:
            acc += abs(v) ** p
        scores.append(round(acc ** (1.0 / p), 10))
    return tuple(scores)


def _channel_slice(t: Tensor, c: int) -> tuple[int, int]:
    if not 0 <= c < t.channels:
        raise TensorError(f"{t.name}: no channel {c} in {t.channels}")
    stride = t.channel_stride
    return c * stride, (c + 1) * stride


# ---------------------------------------------------------------------------
# 1. the fine-grained criterion
# ---------------------------------------------------------------------------

def magnitude_mask(t: Tensor, ratio: float) -> tuple[int, ...]:
    """Which elements survive a magnitude prune at `ratio`, element by element.

    Returns a tuple the same length as `t.data`, `1` to keep and `0` to zero.
    Importance is `|w|` — the criterion of Lecture 05 slide 25, which the
    source itself calls a heuristic.

    The threshold is not chosen; it falls out. You are asked for a fraction,
    so exactly `_drop_count(n, ratio)` elements go, and the threshold is
    whatever value that turned out to be. Slide 30 is about the other half of
    this: whether the fraction is taken per tensor, as here, or once across
    the whole model.
    """
    n = len(t.data)
    k = _drop_count(n, ratio)
    dropset = _smallest_indices(t.data, k)

    # for mask, set everything to 1, then dropset to 0
    mask = [1] * n
    for i in dropset:
        mask[i] = 0
    
    return tuple(mask)


# ---------------------------------------------------------------------------
# 2. the structured criterion
# ---------------------------------------------------------------------------

def channel_keep(t: Tensor, ratio: float, p: float = 2.0) -> tuple[int, ...]:
    """Which output channels survive, ascending.

    Scores every channel with `_group_scores` and drops the
    `_drop_count(C_out, ratio)` weakest, except that at least `MIN_CHANNELS`
    survive — a layer pruned to zero outputs is not a smaller layer, it is a
    disconnected graph, and returning one is worse than refusing the ratio.

    The clamp is reported rather than hidden: `sparsity_row` records the
    achieved reduction, so a 90% request on a 4-channel tensor shows up in
    `sparsity.json` as an achieved 75% and the gap is visible.
    """
    removals = _drop_count(t.channels, ratio)
    drop = min(_drop_count(t.channels, ratio), t.channels - MIN_CHANNELS)
    channel_indices = _smallest_indices(_group_scores(t, p), drop)
    # extract survivors
    kept_channels = []
    for c in range(t.channels):
        if c not in channel_indices:
            kept_channels.append(c)
    kept_channels.sort()
    return tuple(kept_channels)


# ---------------------------------------------------------------------------
# 3. masking — the removal that changes no shape
# ---------------------------------------------------------------------------

def apply_mask(t: Tensor, mask: Sequence[int]) -> Tensor:
    """Return `t` with the masked elements set to zero. Same shape. Same name.

    This is the operation Lecture 05 slide 9 is about. The result has the same
    `parameters` as the input, occupies the same addresses, and will be fetched
    and multiplied by any dense kernel exactly as before. Nothing here is a
    saving; it is a set of values that happen to be zero.
    """
    # assert length match and fail otherwise
    if len(mask) != len(t.data):
        raise ValueError("Mask length does not match tensor data length")

    # apply mask
    result_data = []
    for i, value in enumerate(t.data):
        if mask[i] == 0:
            result_data.append(0.0)
        else:
            result_data.append(value)

    return Tensor(name=t.name, shape=t.shape, dtype=t.dtype, data=result_data)


# ---------------------------------------------------------------------------
# 4. channel removal — the removal that changes the shape
# ---------------------------------------------------------------------------

def drop_channels(t: Tensor, keep: Sequence[int]) -> Tensor:
    """Return a genuinely smaller tensor holding only the kept output channels.

    Axis 0 shrinks to `len(keep)` and the surviving values stay in their
    original relative order. The result is dense, has no holes, and needs
    nothing from the kernel, the format or the hardware to be faster —
    the fourth column of slide 8.

    What this function cannot do is fix up the *next* layer, whose `C_in` must
    now match. Lecture 05 slide 20 draws that propagation and it is why
    channel pruning is a graph operation in real code. This lab prunes tensor
    by tensor and accounts for it that way, which is honest as long as
    `sparsity.json` does not claim the model still runs — and it does not.
    """
    # ensure keep is non-empty, no duplicates and in range
    if not keep or len(keep) == 0 or len(set(keep)) != len(keep):
        raise ValueError("Keep list must be non-empty and contain unique indices")
    if any(c < 0 or c >= t.channels for c in keep):
        raise ValueError("Keep indices must be within the range of available channels")

    keep = sorted(keep)
    # concatenate slices
    new_data = []
    for c in keep:
        lo, hi = _channel_slice(t, c)
        lo, hi = c * t.channel_stride, (c + 1) * t.channel_stride
        new_data.extend(t.data[lo:hi])
    
    # set new shape + return
    new_shape = (len(keep),) + t.shape[1:]
    return Tensor(name=t.name, shape=new_shape, dtype=t.dtype, data=new_data)
# ---------------------------------------------------------------------------
# 5. the file
# ---------------------------------------------------------------------------

def bytes_stored(tensors: Sequence[Tensor], storage: str = "dense",
                 mask_encoding: str = "framework") -> int:
    """How many bytes these tensors occupy on disk under a storage form.

    Three forms, and the arithmetic for each is short enough to check by hand:

      `dense`   parameters x dtype_bytes. Nothing else. A channel-pruned
                tensor is stored this way and its file is smaller because its
                shape is smaller.

      `masked`  parameters x dtype_bytes, plus the mask. `mask_encoding` picks
                what the mask costs: `framework` prices it as a second dense
                tensor in the weight's own dtype, which is what
                `torch.nn.utils.prune` actually keeps until `prune.remove()`
                is called; `bitmap` prices it at the one-bit-per-weight floor,
                which is what a format designed for the job would cost.
                Under `framework` the file is exactly twice the dense file at
                every ratio, including 90%. That is the point.

      `sparse`  nonzeros x dtype_bytes, plus one INDEX_BYTES index per stored
                value. This is the only form where the file falls with the
                sparsity, and it does not fall as fast as the sparsity: at 50%
                in fp32 with 4-byte indices it does not fall at all.

    The `sparse` model ignores a real CSR layout's row-pointer array, which is
    small. Stating the omission rather than absorbing it is the rule: an
    accounting that rounds in its own favour is the thing this lab teaches you
    to distrust.
    """
    # input validation
    if mask_encoding == None:
        mask_encoding = "framework"
    if not((storage == "dense" or storage == "masked" or storage == "sparse") and 
        (mask_encoding == "framework" or mask_encoding == "bitmap")):
        print(mask_encoding)
        return -1 ##?

    # accumulate storage
    total_bytes = 0
    for tensor in tensors:
        if storage == "dense":
            n = tensor.parameters
            db = dtype_bytes(tensor.dtype)
            byts = n * db
        elif storage == "masked":
            n = tensor.parameters
            db = dtype_bytes(tensor.dtype)
            byts = n * db
            if mask_encoding == "framework":
                byts += n * db
            else:
                byts += (n + 7) // 8
        else:
            nnz = 0
            db = dtype_bytes(tensor.dtype)
            for v in tensor.data:
                if v != 0.0:
                    nnz += 1
            byts = nnz * db + nnz * INDEX_BYTES
        total_bytes += byts

    return total_bytes

# ---------------------------------------------------------------------------
# 6. one row of the accounting table
# ---------------------------------------------------------------------------

def sparsity_row(model: str, ratio: float, granularity: str,
                 before: Sequence[Tensor], after: Sequence[Tensor],
                 storage: str = "dense",
                 mask_encoding: str = "framework") -> dict[str, Any]:
    """One line of `sparsity.json`: what this prune asked for and what it got.

    Eight numbers, and the two to read side by side are `nominal_ratio` and
    `achieved_reduction`.

      `nominal_ratio`         what you asked for
      `values_zeroed`         how many numbers are now zero
      `zeroed_fraction`       that, over the parameter count
      `parameters_before`     a shape fact
      `parameters_after`      a shape fact
      `achieved_reduction`    1 - after/before. **Zero for any mask.**
      `bytes_dense`           what the unpruned tensors cost on disk
      `bytes_stored`          what these tensors cost, under `storage`

    `zeroed_fraction` and `achieved_reduction` are the same number for a
    channel prune and are 0.60 against 0.00 for a 60% mask. Every plot in
    this lab is drawn against `achieved_reduction`, and Lecture 05 slide 49
    says why: on a nominal axis the two granularities are not comparable, and
    the comparison is the lab.
    """
    if not (granularity == "fine" or granularity == "channel"):
        return {
        "source": "sparsity_row", 
        "why": "granularity verification failed"
        }

    p_before = total_parameters(before)
    p_after = total_parameters(after)
    subtracted = 0
    for t in after:
        for v in t.data:
            if v != 0.0:
                subtracted += 1
    zeroed = p_after - subtracted

    zeroed_fraction = zeroed / p_before
    achieved_reduction = 1.0 - (p_after / p_before)

    bytes_dense = bytes_stored(before, "dense", "framework")
    bytes_stored1 = bytes_stored(after, storage, mask_encoding)
    removal = classify_removal(before, after, storage)

    if storage == "dense":
        mask_encoding = None
    res = {
        "model":model,
        "granularity":granularity,
        "storage":storage,
        "mask_encoding":mask_encoding,
        "nominal_ratio": ratio,
        "values_zeroed": zeroed, 
        "zeroed_fraction": zeroed_fraction,
        "parameters_before": p_before,
        "parameters_after": p_after,
        "achieved_reduction": achieved_reduction,
        "bytes_dense": bytes_dense,
        "bytes_stored": bytes_stored1,
        "removal": removal
    }

    return res


# ---------------------------------------------------------------------------
# 7. which of the four you got
# ---------------------------------------------------------------------------

def classify_removal(before: Sequence[Tensor], after: Sequence[Tensor],
                     storage: str = "dense") -> str:
    """Name the removal, in the vocabulary of Lecture 05 slide 8.

    The order of the tests is the argument:

      1. If any shape changed, it is `structurally absent`, whatever the
         storage form says. A shape change is the strongest claim available
         and it subsumes the others — a channel-pruned tensor written to a
         sparse format is still structurally absent, and calling it
         `stored sparse` would report the weaker fact.
      2. Otherwise, if it is written sparse, it is `stored sparse`.
      3. Otherwise, if every group of `NM_M` consecutive elements holds at
         most `NM_N` nonzeros, it is `patterned`, which is the only one of
         the four the Ampere sparse tensor cores will look at.
      4. Otherwise, if anything is zero, it is `masked`.
      5. Otherwise nothing was removed, and it says `dense` rather than
         picking one of the four. "Nothing happened" is a legitimate answer
         at ratio 0 and pretending it is a removal would put a baseline row
         in the table under a removal name.

    Test 3 runs after test 1 for a reason worth an argument in Stage D: a
    tensor can satisfy 2:4 by accident at high sparsity, and this function
    will say so. It is reporting a property of the values, not a claim that
    anybody pruned with a 2:4 constraint in mind.
    """
    # returning structurally absent shape changes
    if len(before) != len(after):
        return "structurally absent"
    for t_before, t_after in zip(before, after):
        if t_before.shape != t_after.shape:
            return "structurally absent"

    # returning stored sparse - due to storage form
    if storage == "sparse":
        return "stored sparse"

    # patterned sparsity check
    zeroed_tensor_found = False
    for t_after in after:
        # return dense if no element equals 0.0
        if all(value != 0.0 for value in t_after.data):
            return "dense"
        zeroed_tensor_found = True

        # verify tensor length is divisible by M
        if len(t_after.data) % NM_M != 0:
            return "masked"

        # return masked if any block contains more than N nonzeros
        for i in range(0, len(t_after.data), NM_M):
            group = t_after.data[i:i + NM_M]
            if sum(1 for value in group if value != 0.0) > NM_N:
                return "masked"

    # every zeroed tensor satisfied the 2:4 pattern
    if zeroed_tensor_found:
        return "patterned"

    return "dense"


# ---------------------------------------------------------------------------
# 8. the whole table
# ---------------------------------------------------------------------------

def sweep_model(model: dict[str, Any], ratios: Sequence[float],
                granularities: Sequence[str] = GRANULARITIES,
                storage: str | None = None,
                mask_encoding: str = "framework",
                p: float = 2.0) -> list[dict[str, Any]]:
    """Prune one model at every ratio under every granularity. Stage A.

    Returns a flat list of `sparsity_row` dicts, ordered granularity-major and
    then by ratio ascending, so that `curves.png` can slice it without sorting
    and two students' files diff cleanly.

    `storage=None` means "whatever each granularity naturally produces", which
    is `GRANULARITY_STORAGE`: a fine-grained prune leaves a mask, a channel
    prune leaves a smaller dense tensor. Passing a form overrides both, which
    is how you get the `stored sparse` column of slide 8 out of this lab
    without implementing a sparse format.

    Ratio 0.0 is not a formality and `sweep.py` puts it in by default. It is
    the baseline every other row is a ratio against, and a sweep without it
    has four numbers and no result.
    """
    tensors = model["tensors"]
    res = []
    for g in granularities:
        form = storage if storage else GRANULARITY_STORAGE[g]
        for r in sorted(ratios):
            after = []
            for t in tensors:
                if g == "fine":
                    mask = magnitude_mask(t, r)
                    after.append(apply_mask(t, mask))
                else: 
                    keep = channel_keep(t, r, p)
                    after.append(drop_channels(t, keep))
            res.append(sparsity_row(model["name"], r, g, tensors, after, form, mask_encoding))
    return res