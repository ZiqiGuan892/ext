"""Generated media IDs stay trainable text across Huanyu's whole input path."""

import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_huanyu_media_boundary_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
boundary = importlib.import_module(f"{PACKAGE}.media_boundary")
processor_module = importlib.import_module(f"{PACKAGE}.processor")
forward = importlib.import_module(f"{PACKAGE}.forward")


def config():
    return SimpleNamespace(image_token_id=13, video_token_id=16, vision_start_token_id=14,
                           vision_end_token_id=15, audio_start_token_id=10, audio_end_token_id=11,
                           vision_config=SimpleNamespace(spatial_merge_size=2))


class Tokenizer:
    pad_token_id = 0
    markers = {"<|audio_start|>": 10, "<|audio_end|>": 11, "<unk>": 12,
               "<|image_pad|>": 13, "<|vision_start|>": 14, "<|vision_end|>": 15}

    def encode(self, text, **kwargs):
        return [self.markers[text]]


def make_processor(grid):
    def images(images, **kwargs):
        return {"image_grid_thw": torch.tensor([grid] * len(images)),
                "pixel_values": torch.ones(len(images) * int(torch.tensor(grid).prod()), 4)}

    proc = processor_module.HuanyuProcessor(
        Tokenizer(), images, None, config(), {"sampling_rate": 16000, "feature_size": 40})

    # A native-RoPE-shaped contract: retain nontrivial prompt spatial positions
    # and extend response text sequentially. Generated spans must not consume
    # image/video grids. Both the old and new HF signatures are tested below.
    def rope(self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        markers = input_ids.eq(13)
        assert int(markers.sum()) == (int(image_grid_thw.prod(-1).sum()) // 4
                                     if image_grid_thw is not None else 0)
        assert not input_ids.eq(16).any()
        self.rope_seen = input_ids.clone()
        positions = attention_mask.cumsum(-1) - 1
        return positions.unsqueeze(0).expand(3, -1, -1), torch.zeros(input_ids.shape[0], 1)

    with patch.object(processor_module.importlib, "import_module",
                      return_value=SimpleNamespace(Qwen3_5MoeModel=type("NativeRope", (), {"get_rope_index": rope}))):
        processor_module._bind_qwen_rope(proc)
    return proc


def make_worker(proc):
    class Worker:
        def _compute_multi_modal_inputs(self, output, input_ids):
            raise AssertionError("Huanyu must use its own processor")

        def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs, **kwargs):
            # This upstream bridge deliberately forwards grids only, as on
            # older verl. The patch must carry the boundary itself.
            positions, _ = self.processor.get_rope_index(
                input_ids=input_ids, attention_mask=attention_mask,
                image_grid_thw=multi_modal_inputs.get("image_grid_thw"))
            return positions

    modules = {"verl.experimental.agent_loop.agent_loop": SimpleNamespace(AgentLoopWorker=Worker),
               "verl.trainer.ppo.v1.agent_loop_tq": SimpleNamespace(AgentLoopWorkerTQ=Worker)}
    with patch.object(processor_module.importlib, "import_module", side_effect=modules.__getitem__):
        processor_module.install_agent_loop_bridge()
        processor_module.install_agent_loop_bridge()
    worker = Worker()
    worker.processor = proc
    return worker


class MaskedHFModel(nn.Module):
    """Exercise HF's ID-based media masks plus real embedding/autograd calls."""

    def __init__(self):
        super().__init__()
        self.config = config()
        self.embedding = nn.Embedding(32, 4)
        self.fail = False
        self.calls = 0

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, pixel_values=None, image_grid_thw=None,
                pixel_values_videos=None, video_grid_thw=None, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("native failed")
        result = self.embedding(input_ids)
        self.seen_ids = input_ids.clone()
        self.seen_kwargs = kwargs
        for token, pixels, grids, value in ((13, pixel_values, image_grid_thw, 7.),
                                            (16, pixel_values_videos, video_grid_thw, 8.)):
            if pixels is not None:
                mask = input_ids.eq(token)
                assert int(mask.sum()) == int(grids.prod(-1).sum()) // 4
                features = result.new_full((int(mask.sum()), 4), value)
                result = result.masked_scatter(mask.unsqueeze(-1).expand_as(result), features)
        return result


def make_model(*, cached=False):
    model = MaskedHFModel()
    forward.install_huanyu_forward(
        model, start_token_id=10, end_token_id=11, placeholder_token_id=12,
        image_token_id=13, video_token_id=16, use_precompute=cached,
        encode_audio=lambda *_: [torch.full((1, 4), 9.)],
    )
    return model


@pytest.mark.parametrize("grid,count,extra", [([1, 22, 30], 165, 1), ([1, 40, 88], 880, 2)])
@pytest.mark.parametrize("padding", [0, 5])
def test_reported_trajectory_runs_processor_positions_forward_and_backward(grid, count, extra, padding):
    proc = make_processor(grid)
    worker = make_worker(proc)
    prompt = [14] + [13] * count + [15, 20]
    # Include both a stray marker and a complete generated vision span.
    response = [21] + [13] * extra + [14, 16, 15, 10, 12, 11, 22]
    output = SimpleNamespace(prompt_ids=prompt[:], response_ids=response[:], multi_modal_data={"images": [object()]},
                             response_logprobs=[-0.5] * len(response), response_mask=[1] * len(response))
    ids = torch.tensor([[0] * padding + prompt + response + [0] * 3])
    mask = torch.tensor([[0] * padding + [1] * (len(prompt) + len(response)) + [0] * 3])
    original = ids.clone()
    mm = worker._compute_multi_modal_inputs(output, ids)
    assert mm["image_bound"].tolist() == [[padding + 1, padding + 1 + count]]
    assert mm["huanyu_prompt_lengths"].tolist() == [len(prompt)]
    positions = worker._compute_position_ids(ids, mask, mm)
    assert positions.shape == (3, 1, ids.shape[1])
    assert proc.rope_seen[0, padding + len(prompt):].eq(0).all()

    model = make_model()
    lookup = F.embedding(ids, model.embedding.weight)
    expected = lookup.detach().clone()
    expected[:, padding + 1:padding + 1 + count] = 7
    result = model(input_ids=ids, attention_mask=mask, position_ids=positions, **mm)
    torch.testing.assert_close(result, expected)
    assert "huanyu_prompt_lengths" not in model.seen_kwargs
    result.sum().backward()
    # The real prompt image slots receive visual features. The generated image
    # markers still receive their own token embeddings AND embedding gradients.
    torch.testing.assert_close(model.embedding.weight.grad[13], torch.full((4,), float(extra)))
    for token in (10, 11, 12, 16):
        torch.testing.assert_close(model.embedding.weight.grad[token], torch.ones(4))
    torch.testing.assert_close(ids, original)
    assert output.response_ids == response and output.prompt_ids == prompt
    assert output.response_logprobs == [-0.5] * len(response)
    assert output.response_mask == [1] * len(response)


def test_actual_bad_prompt_and_missing_boundary_are_still_rejected():
    worker = make_worker(make_processor([1, 2, 2]))
    for prompt in ([14, 13, 13, 15, 20], [14, 13, 15, 20, 13]):
        output = SimpleNamespace(prompt_ids=prompt, response_ids=[21], multi_modal_data={"images": [object()]})
        with pytest.raises(ValueError, match="image token lengths"):
            worker._compute_multi_modal_inputs(output, torch.tensor(prompt + [21]))
    with pytest.raises(ValueError, match="image token lengths"):
        worker.processor.build_multi_modal_inputs(torch.tensor([14, 13, 15, 20, 13]), {"images": [object()]})
    with pytest.raises(ValueError, match="both prompt_ids and response_ids"):
        worker.processor.build_multi_modal_inputs(torch.tensor([20]), {}, prompt_ids=[20])


@pytest.mark.parametrize("actual", [[0, 14, 13, 15, 21, 99], [0, 14, 13, 15, 99], [99, 14, 13, 15, 21]])
def test_boundary_refuses_input_mismatch_instead_of_guessing(actual):
    with pytest.raises(ValueError, match="do not match"):
        boundary.trajectory_prompt_view(torch.tensor(actual), [14, 13, 15], [21], 0, text_token_id=0)


@pytest.mark.parametrize("lengths,mask,match", [([9], None, "exceeds"), ([1, 2], None, "per sample"),
                                               ([1.5], None, "integers"), ([-1], None, "nonnegative"),
                                               ([1], torch.ones(1, 1, 4, 4), "2D attention_mask")])
def test_invalid_prompt_lengths_fail_closed(lengths, mask, match):
    with pytest.raises(ValueError, match=match):
        boundary.prompt_only_ids(torch.tensor([[20, 13, 21, 0]]), lengths, mask)


@pytest.mark.parametrize("cached", [False, True])
def test_mixed_batch_repadding_and_cached_media_keep_response_tokens(cached):
    # Sample 0: audio + image + two temporal video groups; sample 1: text only.
    prompt = [10, 12, 11, 14, 13, 15, 20, 14, 16, 15, 21, 14, 16, 15]
    response = [13, 10, 12, 11, 14, 16, 15, 22]
    row = [0, 0] + prompt + response + [0]
    ids = torch.tensor([row, [0] * 18 + [20, 21, 13, 10, 16, 15, 22]])
    mask = torch.tensor([[0, 0] + [1] * (len(prompt) + len(response)) + [0], [0] * 18 + [1] * 7])
    mm = dict(huanyu_prompt_lengths=[torch.tensor([len(prompt)]), torch.tensor([2])],
              huanyu_audio_lengths=[[8], []], huanyu_audio_chunk_counts=[1, 0],
              image_grid_thw=[torch.tensor([[1, 2, 2]]), []],
              video_grid_thw=[torch.tensor([[2, 2, 2]]), []])
    if cached:
        mm.update(huanyu_cache_enabled=[1, 1], huanyu_cached_audio_embeds=[[torch.full((1, 4), 9., dtype=torch.bfloat16)], []],
                  huanyu_cached_vision_embeds=[[torch.full((1, 4), 7., dtype=torch.bfloat16)], []],
                  huanyu_cached_image_counts=[1, 0],
                  huanyu_cached_video_embeds=[[torch.full((2, 4), 8., dtype=torch.bfloat16)], []],
                  huanyu_cached_video_counts=[1, 0])
    else:
        mm.update(huanyu_audio_features=[[torch.ones(40, 8)], []],
                  pixel_values=[torch.ones(4, 4), []], pixel_values_videos=[torch.ones(8, 4), []])
    model = make_model(cached=cached)
    expected = F.embedding(ids, model.embedding.weight).detach().clone()
    expected[0, 3], expected[0, 6], expected[0, [10, 14]] = 9., 7., 8.
    result = model(ids, attention_mask=mask, position_ids=torch.zeros(3, 2, ids.shape[1], dtype=torch.long), **mm)
    torch.testing.assert_close(result, expected)
    reordered = {key: list(reversed(value)) for key, value in mm.items()}
    again = model(ids.flip(0), attention_mask=mask.flip(0),
                  position_ids=torch.zeros(3, 2, ids.shape[1], dtype=torch.long), **reordered)
    torch.testing.assert_close(again, expected.flip(0))
    result.sum().backward()
    torch.testing.assert_close(model.embedding.weight.grad[13], torch.full((4,), 2.))


def test_failed_native_call_clears_context_and_does_not_mask_later_lookups():
    model = make_model()
    model.fail = True
    with pytest.raises(RuntimeError, match="native failed"):
        model(torch.tensor([[20, 13]]), huanyu_prompt_lengths=[1])
    assert forward._ACTIVE_MEDIA_IDS.get() is None
    ids = torch.tensor([[13, 16, 10]])
    torch.testing.assert_close(model.embedding(ids), F.embedding(ids, model.embedding.weight))
    model.fail = False
    result = model(torch.tensor([[20, 13]]), huanyu_prompt_lengths=[1])
    torch.testing.assert_close(result, F.embedding(torch.tensor([[20, 13]]), model.embedding.weight))


def test_position_bridge_leaves_other_processors_unchanged():
    proc = make_processor([1, 2, 2])
    worker = make_worker(proc)
    ids = torch.tensor([[14, 13, 15, 20]])
    seen = []
    worker.processor = SimpleNamespace(get_rope_index=lambda **kw: (seen.append(kw["input_ids"]) or ids, None))
    result = worker._compute_position_ids(ids, torch.ones_like(ids), {"huanyu_prompt_lengths": [1]})
    assert result is ids and seen[0] is ids
