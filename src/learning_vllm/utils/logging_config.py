import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import colorlog
from .common import load_yaml

LOGGER_STR_TO_LEVEL = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

LOGGER_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "purple",
}


def config_logger(
    name: str,
    level="INFO",
    to_console=True,
    save_path=None,
    mode="a",
    max_bytes=1048576,
    backup_count=3,
    log_prefix="[%(asctime)s,%(msecs)03d][pid=%(process)d][%(levelname)s][%(module)s:L%(lineno)d]",
    log_data=" > %(message)s",
    log_suffix="",
    date_fmt=r"%y-%m-%d %H:%M:%S",
) -> None:
    logger = logging.getLogger(name=name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(level=LOGGER_STR_TO_LEVEL.get(level, logging.WARNING))
    if to_console:
        color_formatter = colorlog.ColoredFormatter(
            f"%(log_color)s{log_prefix}%(reset)s{log_data}{log_suffix}",
            datefmt=date_fmt,
            log_colors=LOGGER_COLORS,
            reset=True,
            style="%",
        )
        console = logging.StreamHandler()
        console.setFormatter(color_formatter)
        console.setLevel(level=LOGGER_STR_TO_LEVEL.get(level, logging.WARNING))
        logger.addHandler(console)
    if save_path is not None:
        log_prefix = log_prefix + log_data + log_suffix
        formatter = logging.Formatter(log_prefix, datefmt=date_fmt)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        rotating_file = RotatingFileHandler(
            filename=save_path,
            mode=mode,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        rotating_file.setFormatter(formatter)
        rotating_file.setLevel(level=LOGGER_STR_TO_LEVEL.get(level, logging.WARNING))
        logger.addHandler(rotating_file)

    if name:
        logger.propagate = False


def init_loggers(logger_cfgs: dict | str, global_proc_rank: int = None) -> None:
    """
    Initialize the logger.
    """
    if not isinstance(logger_cfgs, dict):
        logger_cfgs = load_yaml(logger_cfgs)
    for k, v in logger_cfgs.items():
        if global_proc_rank is not None:
            if v.get("save_path") is not None:
                v["save_path"] += f".rk{global_proc_rank}"
            if v.get("log_prefix") is not None:
                v["log_prefix"] += f"[Rank {global_proc_rank}]"
        k = "" if k == "__default__" else k
        config_logger(name=k, **v)
