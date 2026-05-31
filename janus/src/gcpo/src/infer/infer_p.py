import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import PIL.Image
import torch
import torch.multiprocessing as mp
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import random


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # If throughput is pursued subsequently, it can be changed to deterministic=False, benchmark=True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =================== configuration area（Only change this place） ===================
PROMPT_FILE = "Your Prompt File path"

OUTPUT_DIR = "Your Output Path"

MODEL_PATH = "Model Path"

WORLD_SIZE = 4

# Each prompt still only produces one image
PARALLEL_SIZE = 1

# How many different prompts does each GPU process in parallel at a time
PROMPT_BATCH_SIZE = 4

TEMPERATURE = 1.0
CFG_WEIGHT = 5.0
SEED = 42

IMAGE_TOKEN_NUM_PER_IMAGE = 576
IMG_SIZE = 384
PATCH_SIZE = 16
# ===========================================================


def build_prompt(vl_chat_processor, prompt_text):
    conversation = [
        {"role": "<|User|>", "content": prompt_text},
        {"role": "<|Assistant|>", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )

    return sft_format + vl_chat_processor.image_start_tag


def chunk_list(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def safe_filename(text, max_len=50):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in text)
    return safe[:min(max_len, len(safe))]

def get_prompt_token_len(vl_chat_processor, prompt_text):
    prompt = build_prompt(vl_chat_processor, prompt_text)
    return len(vl_chat_processor.tokenizer.encode(prompt))


def make_length_buckets(local_items_with_len, batch_size, max_len_diff=8):
    bucket = []
    bucket_min_len = None

    for item in local_items_with_len:
        idx, text, cur_len = item

        if not bucket:
            bucket = [item]
            bucket_min_len = cur_len
            continue

        if len(bucket) < batch_size and cur_len - bucket_min_len <= max_len_diff:
            bucket.append(item)
        else:
            yield [(i, t) for i, t, _ in bucket]
            bucket = [item]
            bucket_min_len = cur_len

    if bucket:
        yield [(i, t) for i, t, _ in bucket]

def worker(
    rank,
    world_size,
    prompt_file,
    output_dir,
    model_path,
    temperature,
    parallel_size,
    prompt_batch_size,
    cfg_weight,
    seed,
):
    torch.cuda.set_device(rank)
    seed_all(seed + rank)

    os.makedirs(output_dir, exist_ok=True)

    print(f"[rank {rank}/{world_size}] Started on GPU {rank}", flush=True)

    with open(prompt_file, "r", encoding="utf-8") as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    total = len(all_prompts)
    num_per_rank = (total + world_size - 1) // world_size
    start_idx = rank * num_per_rank
    end_idx = min(start_idx + num_per_rank, total)

    if start_idx >= total:
        print(f"[rank {rank}] No work assigned", flush=True)
        return

    print(
        f"[rank {rank}] Processing {start_idx}-{end_idx - 1} "
        f"(total: {end_idx - start_idx}), PROMPT_BATCH_SIZE={prompt_batch_size}",
        flush=True,
    )

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    local_items = [
        (
            idx,
            all_prompts[idx],
            len(
                vl_chat_processor.tokenizer.encode(
                    build_prompt(vl_chat_processor, all_prompts[idx])
                )
            ),
        )
        for idx in range(start_idx, end_idx)
    ]

    local_items.sort(key=lambda x: x[2])

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    for batch_items in make_length_buckets(
        local_items,
        prompt_batch_size,
        max_len_diff=8,
    ):
        global_indices = [x[0] for x in batch_items]
        prompt_texts = [x[1] for x in batch_items]

        generate_batch(
            mmgpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            prompt_texts=prompt_texts,
            global_indices=global_indices,
            output_dir=output_dir,
            rank=rank,
            temperature=temperature,
            parallel_size=parallel_size,
            cfg_weight=cfg_weight,
            image_token_num_per_image=IMAGE_TOKEN_NUM_PER_IMAGE,
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
        )

    print(f"[rank {rank}] Finished", flush=True)


@torch.inference_mode()
def generate_batch(
    mmgpt,
    vl_chat_processor,
    prompt_texts,
    global_indices,
    output_dir,
    rank,
    temperature=1.0,
    parallel_size=1,
    cfg_weight=5.0,
    image_token_num_per_image=576,
    img_size=384,
    patch_size=16,
):
    """
    1. The value of "parallel_size" remains 1, meaning that each prompt generates only one image. 
    2. The batch dimension comes from different prompts rather than multiple image sampling from the same prompt. 
    3. The CFG still uses the cond/uncond dual paths, so the actual language model batch size is 2 * B.
    """

    device = torch.device(f"cuda:{rank}")
    batch_size = len(prompt_texts)

    prompts = [build_prompt(vl_chat_processor, text) for text in prompt_texts]
    encoded = [vl_chat_processor.tokenizer.encode(p) for p in prompts]
    lengths = [len(x) for x in encoded]
    max_len = max(lengths)

    pad_id = vl_chat_processor.pad_id

    # shape: [2 * batch_size, max_len]
    tokens = torch.full(
        (batch_size * 2, max_len),
        fill_value=pad_id,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.zeros(
        (batch_size * 2, max_len),
        dtype=torch.long,
        device=device,
    )

    last_token_pos = []

    for b, ids in enumerate(encoded):
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)
        cur_len = len(ids)

        cond_row = 2 * b
        uncond_row = 2 * b + 1

        # conditional
        tokens[cond_row, :cur_len] = ids_tensor

        # unconditional: Retain the first and last tokens and replace the middle with pad_id
        uncond_ids = ids_tensor.clone()
        if cur_len > 2:
            uncond_ids[1:-1] = pad_id
        tokens[uncond_row, :cur_len] = uncond_ids

        # Note: Here, the pad_id inside the prompt is a valid token of CFG unconditional and requires mask=1
        attention_mask[cond_row, :cur_len] = 1
        attention_mask[uncond_row, :cur_len] = 1

        last_token_pos.extend([cur_len - 1, cur_len - 1])

    last_token_pos = torch.tensor(last_token_pos, dtype=torch.long, device=device)

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros(
        (batch_size, image_token_num_per_image),
        dtype=torch.long,
        device=device,
    )

    past_key_values = None
    cur_attention_mask = attention_mask

    for step in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=cur_attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )

        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state

        if step == 0:
            # The first step is to obtain the exact position of the last token of each prompt, which is called "image_start_tag".
            row_idx = torch.arange(batch_size * 2, device=device)
            last_hidden = hidden_states[row_idx, last_token_pos, :]
        else:
            # The length of inputs_embeds for each subsequent step is always 1.
            last_hidden = hidden_states[:, -1, :]

        logits = mmgpt.gen_head(last_hidden)

        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, step] = next_token.squeeze(dim=-1)

        # The next step is still to construct the conditional/uncertain two paths.
        next_token_2x = (
            torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1)
            .view(-1)
            .to(device)
        )

        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_2x)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

        # The cache already contains the historical tokens. The attention_mask for the new step needs to be appended with 1.
        new_mask = torch.ones(
            (batch_size * 2, 1),
            dtype=cur_attention_mask.dtype,
            device=device,
        )
        cur_attention_mask = torch.cat([cur_attention_mask, new_mask], dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[batch_size, 8, img_size // patch_size, img_size // patch_size],
    )

    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    for b in range(batch_size):
        global_idx = global_indices[b]
        prompt_text = prompt_texts[b]
        name = safe_filename(prompt_text)

        out_path = os.path.join(output_dir, f"{global_idx}_{name}.png")
        PIL.Image.fromarray(dec[b]).save(out_path)

        if global_idx % 20 == 0:
            print(f"[rank {rank}] prompt #{global_idx} → saved: {out_path}", flush=True)


if __name__ == "__main__":
    mp.spawn(
        worker,
        args=(
            WORLD_SIZE,
            PROMPT_FILE,
            OUTPUT_DIR,
            MODEL_PATH,
            TEMPERATURE,
            PARALLEL_SIZE,
            PROMPT_BATCH_SIZE,
            CFG_WEIGHT,
            SEED,
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )

    print("[MAIN] All processes completed", flush=True)