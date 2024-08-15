import ee
from loguru import logger

logger.warning("running __init__.py")
# initialize the Earth Engine project
# ee.Initialize(project = 'ee-speckerfelix')
service_account = "crowther-gee@gem-eth-analysis.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(
    service_account, "auth/gem-eth-analysis-24fe4261f029.json"
)
ee.Initialize(credentials)
