import argparse
import json
import logging
from accelerate.state import PartialState
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCAL_PROCESS_INDEX = 0

def push_to_hub(
    target_model_path: str,
    repo_id: str,
    hf_token: str,
) -> None:
    """
    Push a model to the Hugging Face Hub.

    Args:
        target_model_path (str): Local path to the model directory.
        repo_id (str): Hugging Face Hub repository ID.
        hf_token (str): Hugging Face authentication token.

    Returns:
        None
    """
    if PartialState().process_index == LOCAL_PROCESS_INDEX:
        logger.info("Pushing model to hub...")
        try:
            training_params = json.load(
                open(os.path.join(target_model_path, "training_params.json"), "r")
            )
            # Optionally, remove sensitive info if needed
            # training_params.pop("token")
            json.dump(training_params, open(os.path.join(target_model_path, "training_params.json"), "w"), indent=4)
        except FileNotFoundError:
            logger.warning(f"Training parameters file not found at {target_model_path}/training_params.json")

        api = HfApi(token=hf_token)
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
            api.upload_folder(folder_path=target_model_path, repo_id=repo_id, repo_type="model")
        except Exception as e:
            logger.error(f"Error pushing model to hub: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument(
        "-target_model_path", required=True, help="Local path to the model directory"
    )
    parser.add_argument(
        "-repo_id", required=True, help="Hugging Face Hub repository ID"
    )
    parser.add_argument(
        "-hf_token", required=True, help="Hugging Face authentication token"
    )

    args = parser.parse_args()

    push_to_hub(target_model_path=args.target_model_path, repo_id=args.repo_id, hf_token=args.hf_token)
