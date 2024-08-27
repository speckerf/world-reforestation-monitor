import time

import ee
from loguru import logger


def wait_for_task(task):
    status = task.status()
    logger.debug(f"Waiting for task {task.id} / {status['description']}")
    while True:
        status = task.status()
        state = status["state"]

        if state in ["COMPLETED"]:
            logger.debug(f"Task {state.lower()}: {status['description']}")
            break
        if state in ["FAILED", "CANCELLED"]:
            logger.error(f"Task {state.lower()}: {status['description']}")
            break
        else:
            logger.trace(f"Task running: {status['description']} (state: {state})")
            time.sleep(5)


def wait_for_task_id(task_id):
    logger.debug(f"Waiting for task {task_id}")
    while True:
        task = ee.data.getTaskStatus(task_id)[0]
        state = task["state"]

        if state in ["COMPLETED"]:
            logger.debug(f"Task {state.lower()}: {task['description']}")
            break
        if state in ["FAILED", "CANCELLED"]:
            logger.error(f"Task {state.lower()}: {task['description']}")
            break
        else:
            logger.trace(f"Task running: {task['description']} (state: {state})")
            time.sleep(5)
