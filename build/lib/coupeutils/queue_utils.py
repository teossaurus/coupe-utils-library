import json
from google.cloud import tasks_v2
from typing import Dict, Optional


class QueueUtils:
    def __init__(self, project_id: str, queue: str, location: str = "us-west1"):
        self.project_id = project_id
        self.location = location
        self.queue = queue
        self.client = tasks_v2.CloudTasksClient()

    def post(
        self, payload: Dict, endpoint: str, task_name: Optional[str] = None
    ) -> tasks_v2.Task:
        """Posts a message to a Cloud Tasks queue."""
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": endpoint,
                "headers": {"Content-type": "application/json"},
                "body": json.dumps(payload).encode(),
            }
        }
        if task_name:
            task["name"] = task_name

        parent = self.client.queue_path(self.project_id, self.location, self.queue)
        return self.client.create_task(request={"parent": parent, "task": task})
