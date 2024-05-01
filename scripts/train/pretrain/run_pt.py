import os
from typing import Tuple

lr: float = 2e-4
lora_rank: int = 64
lora_alpha: int = 128
lora_trainable: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save: str = "embed_tokens,lm_head"
lora_dropout: float = 0.05

pretrained_model_path: str = "/path/to/pretrained/model"
bangla_tokenizer_path: str = "/path/to/bangla-tokenizer"
dataset_dir: str = "/path/to/dataset"
data_cache: str = "/path/to/cache_dir"
output_dir: str = "/path/to/output_dir"
deepspeed_config_file: str = "ds_zero2_no_offload.json"

def main():
    torchrun_cmd = f"torchrun --nnodes 1 --nproc_per_node 1 run_clm_pt_with_peft.py "
    torchrun_cmd += f"-deepspeed {deepspeed_config_file} "
    torchrun_cmd += f"-model_name_or_path {pretrained_model_path} "
    torchrun_cmd += f"-tokenizer_name_or_path {bangla_tokenizer_path} "
    torchrun_cmd += f"-dataset_dir {dataset_dir} "
    torchrun_cmd += f"-data_cache_dir {data_cache} "
    torchrun_cmd += f"-validation_split_percentage 0.1 "
    torchrun_cmd += f"-per_device_train_batch_size 64 "
    torchrun_cmd += f"-do_train "
    torchrun_cmd += f"-seed $RANDOM "
    torchrun_cmd += f"-fp16 "
    torchrun_cmd += f"-num_train_epochs 1 "
    torchrun_cmd += f"-lr_scheduler_type cosine "
    torchrun_cmd += f"-learning_rate {lr} "
    torchrun_cmd += f"-warmup_ratio 0.05 "
    torchrun_cmd += f"-weight_decay 0.01 "
    torchrun_cmd += f"-logging_strategy steps "
    torchrun_cmd += f"-logging_steps 10 "
    torchrun_cmd += f"-save_strategy steps "
    torchrun_cmd += f"-save_total_limit 1 "
    torchrun_cmd += f"-save_steps 50 "
    torchrun_cmd += f"-gradient_accumulation_steps 2 "
    torchrun_cmd += f"-preprocessing_num_workers 8 "
    torchrun_cmd += f"-block_size 512 "
    torchrun_cmd += f"-output_dir {output_dir} "
    torchrun_cmd += f"-overwrite_output_dir "
    torchrun_cmd += f"-ddp_timeout 30000 "
    torchrun_cmd += f"-logging_first_step True "
    torchrun_cmd += f"-lora_rank {lora_rank} "
    torchrun_cmd += f"-lora_alpha {lora_alpha} "
    torchrun_cmd += f"-trainable {lora_trainable} "
    torchrun_cmd += f"-lora_dropout {lora_dropout} "
    torchrun_cmd += f"-modules_to_save {modules_to_save} "
    torchrun_cmd += f"-torch_dtype float16 "
    torchrun_cmd += f"-gradient_checkpointing "
    torchrun_cmd += f"-ddp_find_unused_parameters False "
    torchrun_cmd += f"-flash_attn True"

    os.system(torchrun_cmd)

if __name__ == "__main__":
    main()
