import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_adapter(
    base_model_path: str,
    target_model_path: str,
    adapter_path: str
) -> None:
    """
    Merge a PEFT adapter with a base model.

    Args:
        base_model_path (str): Path to the base model.
        target_model_path (str): Path to save the target model.
        adapter_path (str): Path to the adapter model.

    Returns:
        None
    """
    logger.info("Loading adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    model = model.merge_and_unload()
    logger.info("Saving target model...")
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a PEFT adapter with a base model")
    parser.add_argument("-base_model_path", required=True, help="Path to the base model")
    parser.add_argument("-target_model_path", required=True, help="Path to save the target model")
    parser.add_argument("-adapter_path", required=True, help="Path to the adapter model")

    args = parser.parse_args()

    try:
        merge_adapter(
            base_model_path=args.base_model_path,
            target_model_path=args.target_model_path,
            adapter_path=args.adapter_path
        )
    except Exception as e:
        logger.error(f"Failed to merge adapter weights: {e}")
