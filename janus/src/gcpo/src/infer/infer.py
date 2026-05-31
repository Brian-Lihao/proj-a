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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =================== 配置区域（只改这里） ===================
PROMPT_FILE = "Your Prompt Path"

OUTPUT_DIR = "Your Output Path"

MODEL_PATH = "Your Model Path"

WORLD_SIZE = 4       # 总卡数（想开几张卡就写几）
TEMPERATURE = 1.0
PARALLEL_SIZE = 1    # 每个prompt生成几张图
CFG_WEIGHT = 5.0
SEED = 42
# ===========================================================

def worker(rank, world_size, prompt_file, output_dir, model_path, temperature, parallel_size, cfg_weight, seed):
    """每个GPU上运行的worker进程"""

    torch.cuda.set_device(rank)
    seed_all(seed + rank)
    
    print(f"[rank {rank}/{world_size}] Started on GPU {rank}", flush=True)
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    
    total = len(all_prompts)
    num_per_rank = (total + world_size - 1) // world_size
    start_idx = rank * num_per_rank
    end_idx = min(start_idx + num_per_rank, total)
    
    if start_idx >= total:
        print(f"[rank {rank}] No work assigned")
        return
    
    print(f"[rank {rank}] Processing {start_idx}-{end_idx-1} (total: {end_idx-start_idx})", flush=True)
    
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    
    # loop
    for global_idx in range(start_idx, end_idx):
        prompt_text = all_prompts[global_idx]
        generate(
            vl_gpt, vl_chat_processor, prompt_text, global_idx, 
            output_dir, rank, temperature, parallel_size, cfg_weight
        )
    
    print(f"[rank {rank}] Finished", flush=True)

@torch.inference_mode()
def generate(mmgpt, vl_chat_processor, prompt_text, global_idx, output_dir, rank, 
             temperature=1, parallel_size=1, cfg_weight=5, 
             image_token_num_per_image=576, img_size=384, patch_size=16):
    
    conversation = [
        {"role": "<|User|>", "content": prompt_text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag

    input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(prompt))
    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    past_key_values = None

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int), 
        shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs(output_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in prompt_text)
    name_len = min(50, len(safe))
    
    for i in range(parallel_size):
        suffix = f"_{i}" if parallel_size > 1 else ""
        out_path = os.path.join(output_dir, f"{global_idx}_{safe[:name_len]}{suffix}.png")
        PIL.Image.fromarray(visual_img[i]).save(out_path)
        print(f"[rank {rank}] prompt #{global_idx} → saved: {out_path}", flush=True)

if __name__ == "__main__":
    mp.spawn(
        worker,
        args=(WORLD_SIZE, PROMPT_FILE, OUTPUT_DIR, MODEL_PATH, 
              TEMPERATURE, PARALLEL_SIZE, CFG_WEIGHT, SEED),
        nprocs=WORLD_SIZE,
        join=True
    )
    print("[MAIN] All processes completed")