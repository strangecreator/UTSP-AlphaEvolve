import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from django.conf import settings


@dataclass
class TaskItem:
    id: str
    path: Path
    created_at: str
    model: Optional[str]
    has_answer: bool


def queue_dir() -> Path:
    return Path(settings.MANUAL_QUEUE_DIR)


def list_tasks() -> List[TaskItem]:
    q = queue_dir()
    tasks: List[TaskItem] = []

    for p in sorted(q.glob("*.json")):
        if p.name.endswith(".answer.json"):
            continue

        task_id = p.stem  # uuid
        answer = q / f"{task_id}.answer.json"
        has_answer = answer.exists()

        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            created_at = data.get("created_at", "")
            model = data.get("model")
        except Exception:
            created_at, model = "", None

        tasks.append(TaskItem(id=task_id, path=p, created_at=created_at, model=model, has_answer=has_answer))
    
    # show only those without answers
    tasks = [t for t in tasks if not t.has_answer]
    return tasks


def read_task(task_id: str) -> Optional[dict]:
    p = queue_dir() / f"{task_id}.json"

    if not p.exists():
        return None

    return json.loads(p.read_text(encoding='utf-8'))


def write_answer(task_id: str, answer_text: str) -> None:
    q = queue_dir()
    out = q / f"{task_id}.answer.json"
    payload = {"id": task_id, "answer": answer_text}
    tmp = q / f".{task_id}.answer.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(out)