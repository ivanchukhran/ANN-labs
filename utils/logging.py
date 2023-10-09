import logging


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Function setup as many loggers as you want.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Logger object.
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
