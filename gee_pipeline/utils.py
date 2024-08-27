import time

from loguru import logger


def wait_for_task(task):
    status = task.status()
    logger.debug(f"Waiting for task {task.id} / {status['description']}")
    while True:
        status = task.status()
        state = status["state"]

        if state in ["COMPLETED", "FAILED", "CANCELLED"]:
            logger.debug(f"Task {state.lower()}: {status['description']}")
            break
        else:
            logger.trace(f"Task running: {status['description']} (state: {state})")
            time.sleep(20)  # Wait for 10 seconds before checking again
