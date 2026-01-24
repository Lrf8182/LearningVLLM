from pathlib import Path
from learning_vllm.utils.logging_config import init_loggers
import logging
from learning_vllm.utils.common import load_yaml, load_jsonl

from learning_vllm.tinyLLMUniforInterface import TinyLLM, EngineType


def test_inference_config(logger: logging.Logger):
    logger.critical("Running inference config test...")
    inference_cfg = load_yaml("./configs/inference.yml")
    logger.info(inference_cfg)
    logger.critical("Inference config test completed successfully.")


def load_inference_data(data_path: str | Path) -> list[str]:
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"The file '{data_path}' was not found.")
    data = load_jsonl(data_path)
    prompts: list[str] = [item["prompt"] for item in data if "prompt" in item]
    return prompts


def test_inference(logger: logging.Logger, inference_cfg_path: str = "./configs/inference.yml"):
    logger.critical("Running inference test...")

    inference_cfg = load_yaml(inference_cfg_path)

    llm = TinyLLM(
        engine_type=EngineType(inference_cfg.backend),
        # engine_type=inference_cfg["backend"], is also acceptable
        engine_params=inference_cfg.engine_cfg[inference_cfg.backend],
        tokenizer_params=inference_cfg.tokenizer_cfg,
        sampling_params=inference_cfg.sampling_params[inference_cfg.backend],
    )

    data: list[str] = load_inference_data(inference_cfg.input_data[0])

    logger.info(llm.generate(inputs=data))

    logger.critical("Inference test completed successfully.")


if __name__ == "__main__":
    init_loggers("configs/loggers.yml")
    test_logger = logging.getLogger("DEBUG")
    test_inference_config(test_logger)
    test_inference(test_logger)
