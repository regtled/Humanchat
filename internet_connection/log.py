import logging
import os
from pathlib import Path
import warnings

logging.getLogger("autogen.oai.client").setLevel(logging.WARNING)
logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="The API key specified is not a valid OpenAI format; it won't work with the OpenAI-hosted model.")


class Log:
    """日志记录类，用于配置和管理日志记录。"""

    GREEN = "\033[92m"  # ANSI 转义序列，绿色
    RESET = "\033[0m"  # ANSI 转义序列，重置颜色

    # Create a class-level named logger
    logger = logging.getLogger('my_application')

    @classmethod
    def setup_logging(cls, level=logging.INFO):
        """设置日志记录配置。

        Args:
            level (int): 日志级别。
        """
        log_file = Path(__file__).resolve().parent / 'logs/application.log'

        # 确保目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Set the logger level
        cls.logger.setLevel(level)

        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d'))
        cls.logger.addHandler(file_handler)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            f'{cls.GREEN}%(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d{cls.RESET}'))
        cls.logger.addHandler(console_handler)

        # Optionally, prevent propagation to the root logger
        cls.logger.propagate = False

    @classmethod
    def info(cls, message):
        """记录信息级别的日志。

        Args:
            message (str): 要记录的信息。
        """
        cls.logger.info(message, stacklevel=3)

    @classmethod
    def error(cls, message):
        """记录错误级别的日志。

        Args:
            message (str): 要记录的错误信息。
        """
        cls.logger.error(message, stacklevel=3)

    @classmethod
    def warning(cls, message):
        """记录警告级别的日志。

        Args:
            message (str): 要记录的警告信息。
        """
        cls.logger.warning(message, stacklevel=3)

    @classmethod
    def debug(cls, message):
        """记录调试级别的日志。

        Args:
            message (str): 要记录的调试信息。
        """
        cls.logger.debug(message, stacklevel=3)

# 在模块导入时设置日志
Log.setup_logging()

