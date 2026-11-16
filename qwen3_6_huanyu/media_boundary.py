"""Keep input media binding separate from generated token embeddings.

Huanyu's single-turn AR responses are text, even when sampling emits a media
marker ID. Only the prompt owns image/video grids and audio feature chunks.
Prompt lengths count valid tokens, so they survive batch padding and slicing.
"""

import torch


def media_text_token_id(config, extra_markers=()):
    """A small valid token used only in the structural (mask/RoPE) ID view."""
    names = ("image_token_id", "video_token_id", "vision_start_token_id",
             "vision_end_token_id", "audio_start_token_id", "audio_end_token_id")
    markers = {config.get(name) if isinstance(config, dict) else getattr(config, name, None)
               for name in names}
    markers.update(extra_markers)
    return next(token for token in range(len(markers) + 1) if token not in markers)


def prompt_only_ids(input_ids, prompt_lengths, attention_mask=None, *, text_token_id=0):
    """Make a structural view; never modify IDs used as targets/embeddings.

    All padding and response positions are ordinary text in this view. A generated
    complete vision span must not consume another grid, either. Without explicit
    length metadata retain the legacy strict whole-sequence contract.
    """
    if prompt_lengths is None:
        return input_ids
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("Huanyu prompt binding requires input_ids [batch, sequence]")
    lengths = torch.as_tensor(prompt_lengths, device=input_ids.device)
    if lengths.dtype == torch.bool or lengths.is_floating_point() or lengths.is_complex():
        raise ValueError("Huanyu prompt lengths must be integers")
    lengths = lengths.reshape(-1)
    if lengths.numel() != input_ids.shape[0] or bool((lengths < 0).any()):
        raise ValueError("Huanyu prompt lengths must have one nonnegative value per sample")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("Huanyu prompt binding requires a 2D attention_mask matching input_ids")
    valid = (torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None
             else attention_mask.to(device=input_ids.device, dtype=torch.bool))
    if bool((lengths > valid.sum(-1)).any()):
        raise ValueError("Huanyu prompt length exceeds the valid input sequence")
    prompt = valid & (valid.long().cumsum(-1) <= lengths[:, None])
    return input_ids.masked_fill(~prompt, text_token_id)


def trajectory_prompt_view(input_ids, prompt_ids, response_ids, pad_token_id, *, text_token_id):
    """Validate AgentLoop's known sequence and allow only outer batch padding.

    The response boundary comes from AgentLoop, never from a marker-count heuristic.
    Return padded offsets for the existing processor bounds plus a portable length.
    """
    ids = torch.as_tensor(input_ids)
    if ids.device.type != "cpu" or not (ids.ndim == 1 or (ids.ndim == 2 and ids.shape[0] == 1)):
        raise ValueError("Huanyu AgentLoop media binding requires one CPU token sequence")
    prompt, response = list(prompt_ids), list(response_ids)
    expected, actual = prompt + response, ids.reshape(-1).tolist()
    if actual == expected:
        start = 0
    else:
        if pad_token_id is None or not expected:
            raise ValueError("Huanyu input_ids do not match prompt_ids + response_ids")

        def leading_padding(tokens):
            for index, token in enumerate(tokens):
                if token != pad_token_id:
                    return index
            return len(tokens)

        start = leading_padding(actual) - leading_padding(expected)
        end = start + len(expected)
        if (start < 0 or actual[start:end] != expected
                or any(token != pad_token_id for token in actual[end:])):
            raise ValueError("Huanyu padded input_ids do not match prompt_ids + response_ids")
    view = torch.full_like(ids, text_token_id)
    view.reshape(-1)[start:start + len(prompt)] = ids.reshape(-1)[start:start + len(prompt)]
    return view, torch.tensor([len(prompt)], dtype=torch.long)
