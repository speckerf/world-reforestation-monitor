import ee
from loguru import logger


logger.warning("running __init__.py")
# initialize the Earth Engine project
ee.Initialize(project = 'ee-speckerfelix')