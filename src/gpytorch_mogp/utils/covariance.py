"""This module implements utilities for working with covariance matrices."""

import numpy as np
from einops import rearrange

from .blocks import interleave_blocks, to_batched_blocks


def full_to_sparse_covariance(covariance: np.ndarray, n_blocks: int = 1, *, interleaved: bool = False) -> np.ndarray:
    """Convert a full covariance matrix to a sparse covariance matrix.

    By sparse we mean that just the diagonal entries of each block are retained. And the off-diagonal entries are
    removed by reshaping.

    Args:
        covariance: A 2D array of shape (n_blocks * block_size, n_blocks * block_size) representing a block matrix.
            One block corresponds to one combination of output dimensions.
        n_blocks: The number of blocks in the block matrix.
        interleaved: If True, the input block matrix is interpreted as being interleaved. Thus, it will first be converted
            to a non-interleaved block matrix before converting it to a sparse matrix.

    Returns:
        A 3D array of shape (block_size, n_blocks, n_blocks) representing the sparse covariance matrix.
    """
    block_size = covariance.shape[0] // n_blocks

    if interleaved:
        # Convert the interleaved block matrix to a non-interleaved block matrix
        covariance = interleave_blocks(covariance, n_blocks=n_blocks, interleaved=True)

    # (n_blocks * block_size, n_blocks * block_size) -> (n_blocks**2, block_size, block_size)
    blocks = to_batched_blocks(covariance, n_blocks=n_blocks)

    # Extract the diagonal of each block
    diagonals = np.diagonal(blocks, axis1=-2, axis2=-1)

    # (n_blocks**2, block_size) -> (block_size, n_blocks, n_blocks)
    return rearrange(
        diagonals,
        "(n_row_blocks n_col_blocks) block_size -> block_size n_row_blocks n_col_blocks",
        n_row_blocks=n_blocks,
        n_col_blocks=n_blocks,
        block_size=block_size,
    )
