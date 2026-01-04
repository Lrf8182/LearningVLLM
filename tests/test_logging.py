from learning_vllm.utils.logging_config import init_loggers
import logging

if __name__ == "__main__":
    init_loggers("configs/loggers.yml")
    # Global logger
    logging.info("Hi from global logger")
    # BASIC
    basic_logger = logging.getLogger("BASIC")
    basic_logger.debug("This is a debug message from BASIC logger.")
    basic_logger.info("This is an info message from BASIC logger.")
    basic_logger.warning("This is a warning message from BASIC logger.")
    basic_logger.error("This is an error message from BASIC logger.")
    basic_logger.critical("This is a critical message from BASIC logger.")
    # DEBUG
    debug_logger = logging.getLogger("DEBUG")
    debug_logger.debug("This is a debug message from DEBUG logger.")
    debug_logger.info("This is an info message from DEBUG logger.")
    debug_logger.warning("This is a warning message from DEBUG logger.")
    debug_logger.error("This is an error message from DEBUG logger.")
    debug_logger.critical("This is a critical message from DEBUG logger.")
