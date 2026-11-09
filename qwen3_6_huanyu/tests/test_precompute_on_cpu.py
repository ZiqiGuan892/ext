"""Offline encoder/cache boundaries with small CPU tensors and fake frontends."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "huanyu_precompute_cpu_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.precompute", ROOT / "precompute.py")
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


class FakeProcessor:
    config = SimpleNamespace(text_config={"hidden_size": 6})
    spatial_merge_size = 2

    def _vision(self, images):
        return {"pixel_values": torch.arange(16, dtype=torch.float32).reshape(4, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2]])} if images else {}

    def _audio(self, audios, sampling_rate):
        if not audios:
            return [], torch.tensor([], dtype=torch.long), []
        return [torch.ones(4, 8)], torch.tensor([8]), [[1]]

    def _video(self, videos):
        return {"pixel_values_videos": torch.arange(32, dtype=torch.float32).reshape(8, 4) + 100,
                "video_grid_thw": torch.tensor([[2, 2, 2]])} if videos else {}


class FakeEncoders:
    def __init__(self):
        self.calls = []

    def audio(self, feature, length):
        self.calls.append("audio")
        return torch.ones(1, 6)

    def vision(self, pixels, grid):
        self.calls.append("vision")
        return torch.full((int(grid.prod()) // 4, 6), 2.)


class FakeBatch:
    def __init__(self, rows, decoded):
        self.rows = rows
        self.num_rows = len(rows)
        self.decoded = decoded

    def take(self, indices):
        return FakeBatch([self.rows[index] for index in indices], self.decoded)

    def to_pylist(self):
        self.decoded.extend(self.rows)
        return [{"id": value} for value in self.rows]


class FakeParquet:
    def __init__(self, rows, decoded):
        self.rows = rows
        self.decoded = decoded

    def iter_batches(self, batch_size):
        for start in range(0, len(self.rows), batch_size):
            yield FakeBatch(self.rows[start:start + batch_size], self.decoded)


class ShardingTests(unittest.TestCase):
    def test_all_workers_cover_both_files_once_before_decoding(self):
        for workers in (1, 3, 16):
            for batch_size in (1, 2, 4):
                with self.subTest(workers=workers, batch_size=batch_size):
                    all_rows = []
                    for rank in range(workers):
                        decoded = []
                        owned = []
                        offset = 0
                        for rows in ([0, 1, 2, 3, 4], [5, 6, 7, 8]):
                            for local_index, row in cli.iter_shard_rows(
                                FakeParquet(rows, decoded), batch_size, file_offset=offset,
                                num_shards=workers, shard_index=rank,
                            ):
                                self.assertEqual(row["id"], offset + local_index)
                                owned.append(row["id"])
                            offset += len(rows)
                        self.assertEqual(owned, list(range(rank, 9, workers)))
                        self.assertEqual(decoded, owned)
                        self.assertEqual(len(owned), cli.shard_row_count(9, workers, rank))
                        all_rows.extend(owned)
                    self.assertEqual(sorted(all_rows), list(range(9)))

    def test_unassigned_batches_are_not_converted_to_python(self):
        decoded = []
        rows = list(cli.iter_shard_rows(FakeParquet([0, 1, 2], decoded), 2,
                                       num_shards=8, shard_index=7))
        self.assertEqual(rows, [])
        self.assertEqual(decoded, [])
        self.assertEqual(cli.shard_row_count(3, 8, 7), 0)

    def test_empty_input_has_zero_work_for_every_rank(self):
        for rank in range(3):
            self.assertEqual(cli.shard_row_count(0, 3, rank), 0)
            self.assertEqual(list(cli.iter_shard_rows(FakeParquet([], []), 16,
                                                      num_shards=3, shard_index=rank)), [])

    def test_invalid_partitions_are_rejected(self):
        for workers, rank in ((0, 0), (-1, 0), (2, -1), (2, 2)):
            with self.subTest(workers=workers, rank=rank):
                with self.assertRaises(ValueError):
                    cli.shard_row_count(10, workers, rank)
                with self.assertRaises(ValueError):
                    list(cli.iter_shard_rows(FakeParquet([0], []), 1,
                                             num_shards=workers, shard_index=rank))


class PrecomputeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = cli.HuanyuEmbeddingCache(self.root / "cache", {"model": "huanyu"}, create=True)
        fake_dataset = ModuleType(f"{PACKAGE}.dataset")
        fake_dataset.decode_huanyu_media = lambda row, sampling_rate: (
            row.get("images"), row.get("videos"), row.get("audios"),
        )
        self.stub = patch.dict(sys.modules, {fake_dataset.__name__: fake_dataset})
        self.stub.start()
        self.addCleanup(self.stub.stop)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Optional Parquet integration dependency")
    def test_real_arrow_batches_partition_multiple_row_groups_and_files(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        inputs = []
        for name, ids in (("train", list(range(7))), ("validation", list(range(7, 11)))):
            path = self.root / f"{name}.parquet"
            pq.write_table(pa.table({"id": ids, "problem": ["sample"] * len(ids)}), path, row_group_size=3)
            inputs.append(path)
        collected = []
        for rank in range(4):
            offset = 0
            owned = []
            for path in inputs:
                parquet = pq.ParquetFile(path)
                for index, row in cli.iter_shard_rows(parquet, 2, file_offset=offset,
                                                      num_shards=4, shard_index=rank):
                    self.assertEqual(offset + index, row["id"])
                    owned.append(row["id"])
                offset += parquet.metadata.num_rows
            self.assertEqual(owned, list(range(rank, 11, 4)))
            collected.extend(owned)
        self.assertEqual(sorted(collected), list(range(11)))

    def test_populate_resume_and_verify_use_same_keys_without_reencoding(self):
        encoders = FakeEncoders()
        row = {"images": ["image"], "audios": ["audio"]}
        result = cli.process_row(row, FakeProcessor(), self.cache, encoders)
        self.assertEqual(result, {"audio": 1, "vision": 1, "created": 2, "reused": 0})
        result = cli.process_row(row, FakeProcessor(), self.cache, encoders)
        self.assertEqual(result["created"], 0)
        self.assertEqual(result["reused"], 2)
        self.assertEqual(encoders.calls, ["audio", "vision"])
        result = cli.process_row(row, FakeProcessor(), self.cache, verify_only=True)
        self.assertEqual(result["reused"], 2)

    def test_real_processor_short_tail_populates_and_verifies_identical_audio_cache(self):
        processor_module = importlib.import_module(f"{PACKAGE}.processor")
        frontend_inputs = []

        def extract_fbank(waveform):
            frontend_inputs.append(waveform.copy())
            # Keep the native framing contract while making the test feature
            # depend on every frame's waveform, including the padded tail.
            windows = torch.from_numpy(waveform.copy()).unfold(0, 400, 160)
            feature = windows.mean(dim=-1).unsqueeze(0).repeat(40, 1)
            return feature, windows.shape[0]

        processor = processor_module.HuanyuProcessor(
            SimpleNamespace(encode=lambda token, **kwargs: [1]), None,
            SimpleNamespace(extract_fbank=extract_fbank),
            SimpleNamespace(text_config={"hidden_size": 6}),
            {"feature_size": 40, "sampling_rate": 16000},
        )

        class RecordingEncoders(FakeEncoders):
            def __init__(self):
                super().__init__()
                self.audio_inputs = []

            def audio(self, feature, length):
                self.calls.append("audio")
                self.audio_inputs.append((feature.clone(), length))
                tokens = ((length + 3) // 4) // 2
                return torch.arange(1, tokens * 6 + 1, dtype=torch.float32).reshape(tokens, 6)

        for size, lengths, tokens in (
            (480749, [2998, 8], [375, 1]),
            (960749, [2998, 2998, 8], [375, 375, 1]),
            (749, [8], [1]),
            (1040, [5], [1]),
        ):
            with self.subTest(samples=size):
                frontend_inputs.clear()
                waveform = np.linspace(-0.9, 0.9, size, dtype=np.float32)
                original = waveform.copy()
                row = {"audios": [waveform]}
                cache_root = self.root / f"short-tail-{size}"
                cache = cli.HuanyuEmbeddingCache(cache_root, {"model": "huanyu"}, create=True)
                encoders = RecordingEncoders()
                result = cli.process_row(row, processor, cache, encoders)
                self.assertEqual(result, {"audio": len(lengths), "vision": 0,
                                          "created": len(lengths), "reused": 0})
                self.assertEqual([length for _, length in encoders.audio_inputs], lengths)
                self.assertEqual([len(chunk) for chunk in frontend_inputs],
                                 [480000] * (size // 480000) + [1520 if size % 480000 < 1040 else 1040])
                processed = np.concatenate(frontend_inputs)
                np.testing.assert_array_equal(processed[:size], original)
                np.testing.assert_array_equal(processed[size:], np.zeros(len(processed) - size, np.float32))
                np.testing.assert_array_equal(waveform, original)
                for (feature, length), token_count, chunk in zip(
                    encoders.audio_inputs, tokens, frontend_inputs, strict=True,
                ):
                    self.assertEqual(tuple(feature.shape), (40, length))
                    expected_frames = torch.from_numpy(chunk).unfold(0, 400, 160).mean(dim=-1)
                    torch.testing.assert_close(feature, expected_frames.unsqueeze(0).repeat(40, 1))
                    key = cli.audio_key(feature, length)
                    embedding = cache.get("audio", key, expected_tokens=token_count, hidden_size=6)
                    self.assertEqual(tuple(embedding.shape), (token_count, 6))
                    self.assertEqual(embedding.dtype, torch.bfloat16)
                before = {path.relative_to(cache_root): path.read_bytes()
                          for path in cache_root.rglob("*") if path.is_file()}
                for verify_only in (False, True):
                    result = cli.process_row(row, processor, cache, encoders, verify_only=verify_only)
                    self.assertEqual(result, {"audio": len(lengths), "vision": 0,
                                              "created": 0, "reused": len(lengths)})
                self.assertEqual(encoders.calls, ["audio"] * len(lengths))
                after = {path.relative_to(cache_root): path.read_bytes()
                         for path in cache_root.rglob("*") if path.is_file()}
                self.assertEqual(before, after)

        encoders = RecordingEncoders()
        before = {path.relative_to(self.cache.root): path.read_bytes()
                  for path in self.cache.root.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(ValueError, "empty"):
            cli.process_row({"audios": [np.zeros(0, np.float32)]}, processor, self.cache, encoders)
        self.assertEqual(encoders.calls, [])
        self.assertEqual(before, {path.relative_to(self.cache.root): path.read_bytes()
                                 for path in self.cache.root.rglob("*") if path.is_file()})

    def test_verify_missing_never_encodes(self):
        encoders = FakeEncoders()
        with self.assertRaisesRegex(FileNotFoundError, "cache miss"):
            cli.process_row({"audios": ["audio"]}, FakeProcessor(), self.cache, encoders, verify_only=True)
        self.assertEqual(encoders.calls, [])

    def test_video_is_one_temporal_cache_entry_and_resume_uses_identical_key(self):
        encoders = FakeEncoders()
        processor = FakeProcessor()
        row = {"images": ["image"], "videos": ["video"], "audios": ["audio"]}
        result = cli.process_row(row, processor, self.cache, encoders)
        self.assertEqual(result, {"audio": 1, "vision": 2, "created": 3, "reused": 0})
        self.assertEqual(encoders.calls, ["audio", "vision", "vision"])
        video = processor._video(row["videos"])
        key = cli.vision_key(video["pixel_values_videos"], video["video_grid_thw"][0])
        torch.testing.assert_close(self.cache.get("vision", key, 2, 6), torch.full((2, 6), 2., dtype=torch.bfloat16))
        result = cli.process_row(row, processor, self.cache, verify_only=True)
        self.assertEqual(result["reused"], 3)
        self.assertEqual(result["created"], 0)
        self.assertEqual(encoders.calls, ["audio", "vision", "vision"])

    def test_old_frame_entries_cannot_satisfy_a_paired_video_cache_lookup(self):
        processor = FakeProcessor()
        cli.process_row({"images": ["image"]}, processor, self.cache, FakeEncoders())
        encoders = FakeEncoders()
        with self.assertRaisesRegex(FileNotFoundError, "cache miss"):
            cli.process_row({"videos": ["video"]}, processor, self.cache, encoders, verify_only=True)
        self.assertEqual(encoders.calls, [])

    def test_plain_text_row_has_no_encoder_work(self):
        result = cli.process_row({}, FakeProcessor(), self.cache)
        self.assertEqual(sum(result.values()), 0)

    def test_vision_uses_merger_output_and_rejects_deepstack(self):
        result = torch.ones(2, 6)
        output = SimpleNamespace(pooler_output=result, last_hidden_state=torch.ones(8, 4))
        torch.testing.assert_close(cli.vision_output_tokens(output, 2, 6), result)
        torch.testing.assert_close(cli.vision_output_tokens((result, []), 2, 6), result)
        with self.assertRaisesRegex(ValueError, "post-merger"):
            cli.vision_output_tokens(torch.ones(8, 4), 2, 6)
        with self.assertRaisesRegex(ValueError, "DeepStack"):
            cli.vision_output_tokens((result, [torch.ones(2, 6)]), 2, 6)

    def test_weight_loader_selects_visual_only_and_checks_index(self):
        save_file({"model.visual.proj.weight": torch.ones(6, 4),
                   "model.language_model.weight": torch.ones(3, 2)}, str(self.root / "model.safetensors"))
        result = cli.load_vision_weights(self.root)
        self.assertEqual(set(result), {"proj.weight"})
        (self.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "model.visual.proj.weight": "model.safetensors", "model.visual.missing": "model.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "missing"):
            cli.load_vision_weights(self.root)

    def test_tiny_hf_vision_only_loads_once_and_matches_expected_tokens(self):
        class TinyVision(torch.nn.Module):
            constructions = 0

            def __init__(self, config):
                super().__init__()
                self.proj = torch.nn.Linear(4, 6, bias=False)
                TinyVision.constructions += 1

            def forward(self, pixels, grid_thw, **kwargs):
                assert kwargs == {"output_attentions": False, "output_hidden_states": False}
                values = self.proj(pixels).unsqueeze(0).unsqueeze(0)
                result, _ = fake_hf.eager_attention_forward(
                    self, torch.zeros_like(values), torch.zeros_like(values), values, None, scaling=1.,
                )
                result = result.reshape(-1, 4, 6).mean(1)
                return SimpleNamespace(pooler_output=result)

        def eager(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
            scores = (query @ key.transpose(-1, -2)) * scaling
            weights = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
            return (weights @ value).transpose(1, 2).contiguous(), weights

        fake_hf = SimpleNamespace(Qwen3_5MoeVisionModel=TinyVision, eager_attention_forward=eager)

        save_file({"model.visual.proj.weight": torch.ones(6, 4)}, str(self.root / "model.safetensors"))
        processor = FakeProcessor()
        processor.config = SimpleNamespace(text_config={"hidden_size": 6},
                                           vision_config=SimpleNamespace(deepstack_visual_indexes=[]))
        encoders = cli.FrozenEncoders(str(self.root), processor, "cpu")
        with patch.object(cli.importlib, "import_module", return_value=fake_hf):
            for _ in range(2):
                output = encoders.vision(torch.ones(4, 4), torch.tensor([1, 2, 2]))
                torch.testing.assert_close(output, torch.full((1, 6), 4., dtype=torch.bfloat16))
                self.assertIs(fake_hf.eager_attention_forward, eager)
        self.assertEqual(TinyVision.constructions, 1)
        self.assertIsNone(encoders.audio_model)

    def test_vision_rejects_an_unobserved_attention_path(self):
        class UnobservedVision(torch.nn.Module):
            def forward(self, pixels, **kwargs):
                return SimpleNamespace(pooler_output=torch.ones(1, 6))

        encoders = cli.FrozenEncoders(str(self.root), FakeProcessor(), "cpu")
        encoders.vision_model = UnobservedVision().eval()
        original = lambda *args, **kwargs: None
        encoders.vision_hf = SimpleNamespace(eager_attention_forward=original)
        with self.assertRaisesRegex(RuntimeError, "did not use the bounded eager"):
            encoders.vision(torch.ones(4, 4), torch.tensor([1, 2, 2]))
        self.assertIs(encoders.vision_hf.eager_attention_forward, original)

    def test_audio_pads_each_real_chunk_and_never_loads_text_model(self):
        seen = []
        audio_runtime = ModuleType(f"{PACKAGE}.audio_runtime")

        def attach(module, config):
            self.assertEqual(config.path, str(self.root))
            return module

        def encode(module, features, lengths):
            seen.append((features.clone(), lengths.clone()))
            return [torch.ones(1, 6)]

        audio_runtime.attach_huanyu_audio = attach
        audio_runtime.encode_huanyu_audio = encode
        encoders = cli.FrozenEncoders(str(self.root), FakeProcessor(), "cpu")
        with patch.dict(sys.modules, {audio_runtime.__name__: audio_runtime}):
            encoders.audio(torch.ones(4, 16), 10)
        self.assertEqual(tuple(seen[0][0].shape), (1, 4, 12))
        self.assertEqual(seen[0][1].tolist(), [10])
        torch.testing.assert_close(seen[0][0][:, :, 10:], torch.zeros(1, 4, 2))
        self.assertIsNone(encoders.vision_model)


if __name__ == "__main__":
    unittest.main()
