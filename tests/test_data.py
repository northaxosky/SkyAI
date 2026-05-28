"""Tests for the data/ package"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from skyai.data.loader import DataLoader, load_tokens


class TestLoadTokens:
    def test_returns_long_tensor(self, tmp_path: Path) -> None:
        shard = tmp_path / "shard.npy"
        np.save(shard, np.arange(100, dtype=np.uint16))

        tokens = load_tokens(shard)
        assert tokens.dtype == torch.long
        assert tokens.shape == (100,)

    def test_preserves_token_values(self, tmp_path: Path) -> None:
        shard = tmp_path / "shard.npy"
        np.save(shard, np.array([5, 100, 50000], dtype=np.uint16))

        tokens = load_tokens(shard)
        assert tokens.tolist() == [5, 100, 50000]


class TestDataLoader:
    def test_batch_shape(self, synthetic_shards: Path) -> None:
        loader = DataLoader(
            synthetic_shards, split="train", batch_size=2, block_size=4
        )
        x, y = loader.next_batch()
        
        assert x.shape == (2, 4)
        assert y.shape == (2, 4)

    def test_targets_are_inputs_shifted_by_one(self, synthetic_shards: Path) -> None:
        loader = DataLoader(
            synthetic_shards, split="train", batch_size=2, block_size=4
        )
        x, y = loader.next_batch()

        flat_x = x.flatten()
        flat_y = y.flatten()

        # Each consecutive y element is the source-stream successor of x element
        assert torch.equal(flat_y[:-1], flat_x[1:])

    def test_per_rank_offset_yields_disjoint_data(self, synthetic_shards: Path) -> None:
        rank0 = DataLoader(
            synthetic_shards, split="train", batch_size=2, block_size=4,
            rank=0, world_size=2
        )
        rank1 = DataLoader(
            synthetic_shards, split="train", batch_size=2, block_size=4,
            rank=1, world_size=2
        )
        x0, _ = rank0.next_batch()
        x1, _ = rank1.next_batch()

        # Ranks must see different tokens
        assert not torch.equal(x0, x1)

    def test_shard_rotation_wraps(self, synthetic_shards: Path) -> None:
        # Each shard has 1000 tokens: B * T = 8 each step
        loader = DataLoader(
            synthetic_shards, split="train", batch_size=2, block_size=4
        )
        starts = [0]
        for _ in range(500):
            loader.next_batch()
            starts.append(loader.current_shard)

        # Should have visited both shards and wrapped at least once
        assert set(starts) == {0, 1}

    def test_state_dict_roundtrip(self, synthetic_shards: Path) -> None:
        loader = DataLoader(
            synthetic_shards, split="train", batch_size=2, block_size=4
        )

        # Advance a few batches to non-zero state
        for _ in range(3):
            loader.next_batch()
        saved_state = loader.state_dict()
        expected_batch = loader.next_batch()

        # Fresh loader, restore, and verify we get the correct next batch
        fresh = DataLoader(
            synthetic_shards, split="train", batch_size=2, block_size=4
        )
        fresh.load_state_dict(saved_state)
        actual_batch = fresh.next_batch()

        assert torch.equal(actual_batch[0], expected_batch[0])
        assert torch.equal(actual_batch[1], expected_batch[1])

    def test_reset_returns_to_start(self, synthetic_shards: Path) -> None:
        loader = DataLoader(
            synthetic_shards, split="train", batch_size=2, block_size=4
        )
        first_batch = loader.next_batch()
        for _ in range(5):
            loader.next_batch()
        loader.reset()
        same_first_batch = loader.next_batch()

        assert torch.equal(first_batch[0], same_first_batch[0])

    def test_rejects_invalid_split(self, synthetic_shards: Path) -> None:
        with pytest.raises(ValueError, match="Split must be"):
            DataLoader(
                synthetic_shards, split="test", batch_size=2, block_size=4
            )
    
    def test_rejects_invalid_rank(self, synthetic_shards: Path) -> None:
        with pytest.raises(ValueError, match="Rank must be in"):
            DataLoader(
                synthetic_shards, split="train", batch_size=2, block_size=4,
                rank=3, world_size=2
            )

    def test_raises_when_no_shards_found(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No shards matching"):
            DataLoader(empty_dir, split="train", batch_size=2, block_size=4)