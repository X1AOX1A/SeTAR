import re
import os
import sys
import time
import logging

__all__ = ["Logger", "setup_logger"]

class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to the log levels."""
    COLOR_ANSI = {
        "WHITE": "\033[97m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "BLUE": "\033[94m",
        "RED": "\033[91m",
        "CYAN": "\033[96m",
        "RESET": "\033[0m"
    }
    COLOR_MAPPER = {
        "WARNING": "YELLOW",
        "INFO": "WHITE",
        "DEBUG": "BLUE",
        "CRITICAL": "RED",
        "ERROR": "RED",
        "asctime": "GREEN",
        "levelname": "WHITE",
        "message": "WHITE",
        "filename": "CYAN",
        "lineno": "CYAN",
        "funcName": "CYAN",
    }

    def __init__(self, use_color=True):
        super().__init__()
        self.use_color = use_color

    def get_color(self, module_name):
        if self.use_color:
            return self.COLOR_ANSI[self.COLOR_MAPPER.get(module_name, "RESET")]
        return ''

    def format(self, record):
        record.asctime = self.formatTime(record, self.datefmt)
        formatted_log = f"{self.get_color('asctime')}{record.asctime}{self.get_color('RESET')} | " \
                        f"{self.get_color(record.levelname)}{record.levelname}{self.get_color('RESET')} | " \
                        f"{self.get_color('filename')}{record.filename}:{record.lineno}{self.get_color('RESET')} | " \
                        f"{self.get_color('funcName')}{record.funcName}{self.get_color('RESET')} | " \
                        f"{self.get_color(record.levelname)}{record.msg}{self.get_color('RESET')}"
        return formatted_log


class Logger:
    ANSI_CODE_RE = re.compile(r'\x1b\[[0-9;]*m')

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file:
            self.file.write(self.ANSI_CODE_RE.sub('', msg))

    def flush(self):
        self.console.flush()
        if self.file:
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()

    def setup_logging(self, level=logging.INFO):
        color_log_format = ColorFormatter(use_color=True)
        plain_log_format = ColorFormatter(use_color=False)

        handlers = []
        if self.file:
            file_handler = logging.StreamHandler(self.file)
            file_handler.setFormatter(plain_log_format)
            handlers.append(file_handler)

        console_handler = logging.StreamHandler(self.console)
        console_handler.setFormatter(color_log_format)
        handlers.append(console_handler)

        logging.basicConfig(level=level, handlers=handlers)


def setup_logger(log_dir="./output", log_file="logger.log", log_level=logging.INFO):
    """Setup the logger.
    Args:
        log_dir (str, optional): directory to save logging file.
        log_file (str, optional): log file name. If None, logs only to console.
        log_level (int): logging level, e.g. logging.INFO or logging.DEBUG.
    """

    log_path = None
    if log_file:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)

        if os.path.exists(log_path):
            time_string = time.strftime("-%Y-%m-%d-%H-%M-%S")
            log_path = f"{os.path.splitext(log_path)[0]}{time_string}{os.path.splitext(log_path)[1]}"

    # Set up stdout redirection
    log_instance = Logger(log_path)
    sys.stdout = log_instance

    # Remove all handlers associated with the root logger object
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set up logging redirection
    log_instance.setup_logging(level=log_level)

if __name__ == "__main__":
    setup_logger("./output", "logger.log", logging.INFO)
    print("This is a message.")
    logging.info("This is a logging.info message.")