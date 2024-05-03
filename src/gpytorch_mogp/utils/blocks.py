"""This module implements utilities for working with block matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING

from einops import rearrange

if TYPE_CHECKING:
    import numpy as np


def interleave_blocks(
    x: np.ndarray,
    n_blocks: int | None = None,
    *,
    n_row_blocks: int | None = None,
    n_col_blocks: int | None = None,
    interleaved: bool = False,
) -> np.ndarray:
    """Convert a block matrix to an interleaved block matrix.

    Block matrix: (n_row_blocks * row_block_size, n_col_blocks * col_block_size)
    Interleaved block matrix: (row_block_size * n_row_blocks, col_block_size * n_col_blocks)

    Args:
        x: A 2D block matrix of shape (n_row_blocks * row_block_size, n_col_blocks * col_block_size).
        n_blocks: The number of blocks in the block matrix.
        n_row_blocks: The number of row blocks in the block matrix. If not provided, it is set to n_blocks.
        n_col_blocks: The number of column blocks in the block matrix. If not provided, it is set to n_blocks.
        interleaved: If True, the input block matrix is interpreted as already being interleaved. As a result, the
            output should be interpreted as a non-interleaved block matrix. This has the same effect as setting
            n_blocks/n_row_blocks/n_col_blocks to the block size(s) instead of the number of blocks.

    Returns:
        An array of shape (row_block_size * n_row_blocks, col_block_size * n_col_blocks) representing the interleaved
        block matrix. If interleaved is True, the output will be a non-interleaved block matrix.

    Examples:
        >>> import numpy as np
        >>> from einops import repeat
        >>> n_blocks = 2
        >>> block_size = 3
        >>> block_matrix = repeat(
        ...     np.arange(n_blocks * n_blocks),
        ...     "(nbr nbc) -> (nbr bsr) (nbc bsc)",
        ...     nbr=n_blocks, nbc=n_blocks, bsr=block_size, bsc=block_size
        ... )
        >>> block_matrix
        array([[0, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1],
               [2, 2, 2, 3, 3, 3],
               [2, 2, 2, 3, 3, 3],
               [2, 2, 2, 3, 3, 3]])
        >>> interleaved_block_matrix = interleave_blocks(block_matrix, n_blocks=n_blocks)
        >>> interleaved_block_matrix
        array([[0, 1, 0, 1, 0, 1],
               [2, 3, 2, 3, 2, 3],
               [0, 1, 0, 1, 0, 1],
               [2, 3, 2, 3, 2, 3],
               [0, 1, 0, 1, 0, 1],
               [2, 3, 2, 3, 2, 3]])
        >>> assert np.array_equal(interleave_blocks(interleaved_block_matrix, n_blocks=n_blocks, interleaved=True), block_matrix)

        `interleave_blocks` is its own inverse when used with `interleaved=True` or `n_blocks=block_size`:
        >>> n_blocks = 5
        >>> block_size = 7
        >>> x = np.random.rand(n_blocks * block_size, n_blocks * block_size)
        >>> assert np.array_equal(interleave_blocks(interleave_blocks(x, n_blocks=n_blocks), n_blocks=n_blocks, interleaved=True), x)
        >>> assert np.array_equal(interleave_blocks(interleave_blocks(x, n_blocks=n_blocks, interleaved=True), n_blocks=n_blocks), x)
        >>> assert np.array_equal(interleave_blocks(interleave_blocks(x, n_blocks=n_blocks), n_blocks=block_size), x)

        The number of row and column blocks don't have to be the same:
        >>> n_row_blocks = 2
        >>> n_col_blocks = 3
        >>> col_block_size = 5
        >>> row_block_size = 7
        >>> x = np.random.rand(n_row_blocks * row_block_size, n_col_blocks * col_block_size)
        >>> assert np.array_equal(interleave_blocks(interleave_blocks(x, n_row_blocks=n_row_blocks, n_col_blocks=n_col_blocks), n_row_blocks=n_row_blocks, n_col_blocks=n_col_blocks, interleaved=True), x)
    """
    if n_row_blocks is None:
        if n_blocks is None:
            msg = "Either n_blocks or n_row_blocks must be provided."
            raise ValueError(msg)
        n_row_blocks = n_blocks

    if n_col_blocks is None:
        if n_blocks is None:
            msg = "Either n_blocks or n_col_blocks must be provided."
            raise ValueError(msg)
        n_col_blocks = n_blocks

    row_block_size = x.shape[0] // n_row_blocks
    col_block_size = x.shape[1] // n_col_blocks

    if interleaved:
        pattern = (
            "(row_block_size n_row_blocks) (col_block_size n_col_blocks)"
            " -> (n_row_blocks row_block_size) (n_col_blocks col_block_size)"
        )
    else:
        pattern = (
            "(n_row_blocks row_block_size) (n_col_blocks col_block_size)"
            " -> (row_block_size n_row_blocks) (col_block_size n_col_blocks)"
        )

    return rearrange(
        x,
        pattern,
        n_row_blocks=n_row_blocks,
        row_block_size=row_block_size,
        n_col_blocks=n_col_blocks,
        col_block_size=col_block_size,
    )


def to_batched_blocks(x: np.ndarray, n_blocks: int) -> np.ndarray:
    """Converts a 2D block matrix into batches of individual blocks as a 3D array.

    Assumes that the input matrix `x` is a block matrix formed by horizontally and vertically stacking multiple square
    blocks of equal size. This function separates the input matrix into individual blocks and returns them as batches in
    a 3D array.

    Args:
        x: A 2D block matrix of shape (n_blocks * block_size, n_blocks * block_size).
        n_blocks: The number of blocks along each dimension.

    Returns:
        A 3D array of shape (n_blocks**2, block_size, block_size), where block_size = x.shape[0] // n_blocks. Each block
        in the input matrix is represented as a 2D array in the output 3D array, with all blocks batched together in the
        first dimension.

    Examples:
        >>> import numpy as np
        >>> n_blocks = 2
        >>> block_size = 3
        >>> block_matrix = np.arange(n_blocks * n_blocks).reshape(n_blocks, n_blocks)
        >>> block_matrix = np.kron(block_matrix, np.ones((block_size, block_size), dtype=int))
        >>> block_matrix
        array([[0, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1, 1],
               [2, 2, 2, 3, 3, 3],
               [2, 2, 2, 3, 3, 3],
               [2, 2, 2, 3, 3, 3]])
        >>> blocks = to_batched_blocks(block_matrix,n_blocks=n_blocks)
        >>> blocks
        array([[[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
        <BLANKLINE>
               [[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]],
        <BLANKLINE>
               [[2, 2, 2],
                [2, 2, 2],
                [2, 2, 2]],
        <BLANKLINE>
               [[3, 3, 3],
                [3, 3, 3],
                [3, 3, 3]]])
    """
    block_size = x.shape[0] // n_blocks
    return rearrange(
        x, "(nbr bsr) (nbc bsc) -> (nbr nbc) bsr bsc", nbr=n_blocks, nbc=n_blocks, bsr=block_size, bsc=block_size
    )
