import sys
from loguru import logger

logger.configure(handlers=[{"sink": sys.stdout, "level": "WARNING"}])

from .dataset import Dataset
