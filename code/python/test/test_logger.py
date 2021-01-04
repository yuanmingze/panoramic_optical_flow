
import configuration as config

from utility import logger

log = logger.Logger(__name__)
log.logger.propagate = False

def test_logger():
    log.debug("this is debug")
    log.info("this is info")
    log.warn("this is warn")
    log.error("this is error")
    log.critical("this is critical")

if __name__ == "__main__":
    test_logger()
