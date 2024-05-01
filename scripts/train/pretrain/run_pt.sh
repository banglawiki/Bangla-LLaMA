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
    torchrun --nnodes 1 --nproc_per_node 1 run_clm_pt_with_peft.py \
        -deepspeed ${deepspeed_config_file} \
        -model_name_or_path ${pretrained_model_path} \
        -tokenizer_name_or_path ${bangla_tokenizer_path} \
        -dataset_dir ${dataset_dir} \
        -data_cache_dir ${data_cache} \
        -validation_split_percentage 0.1 \
        -per_device_train_batch_size 64 \
        -do_train \
        -seed $RANDOM \
        -fp16 \
        -num_train_epochs 1 \
        -lr_scheduler_type cosine \
        -learning_rate ${lr} \
        -warmup_ratio 0.05 \
        -weight_decay 0.01 \
        -logging_strategy steps \
        -logging_steps 10 \
        -save_strategy steps \
        -save_total_limit 1 \
        -save_steps 50 \
        -gradient_accumulation_steps 2 \
        -preprocessing_num_workers 8 \
        -block_size 512 \
        -output_dir ${output_dir} \
        -overwrite_output_dir \
        -ddp_timeout 30000 \
        -logging_first_step True \
        -lora_rank ${lora_rank} \
        -lora_alpha ${lora_alpha} \
        -trainable ${lora_trainable} \
        -lora_dropout ${lora_dropout} \
        -modules_to_save ${modules_to_save} \
        -torch_dtype float16 \
        -gradient_checkpointing \
        -ddp_find_unused_parameters False \
        -flash_attn True

if __name__ == "__main__":
    main()
