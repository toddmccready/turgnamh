# Imports ----
import logging


# Settings ----
logger = logging.getLogger('turgnamh')

if not logger.hasHandlers():
    logging.basicConfig(
        level = logging.CRITICAL + 1,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()])
