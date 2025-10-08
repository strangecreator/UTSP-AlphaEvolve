from django.http import JsonResponse, HttpResponseNotFound, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt

from .utils import list_tasks, read_task, write_answer


@require_GET
def index(request):
    return render(request, "tasks/index.html")


@require_GET
def api_tasks(request):
    items = list_tasks()
    data = [{
        "id": t.id,
        "created_at": t.created_at,
        "model": t.model,
    } for t in items]
    return JsonResponse({"tasks": data})


@require_GET
def api_task_detail(request, task_id: str):
    data = read_task(task_id)
    if data is None:
        return HttpResponseNotFound("Task not found")
    
    # Return only fields needed by UI
    return JsonResponse({
        "id": data.get("id"),
        "created_at": data.get("created_at"),
        "model": data.get("model"),
        "display_prompt": data.get("display_prompt", ""),
    })


@csrf_exempt
@require_POST
def api_task_answer(request, task_id: str):
    answer = request.POST.get("answer", "")

    if not answer.strip():
        return HttpResponseBadRequest("Answer must not be empty")
    
    data = read_task(task_id)
    if data is None:
        return HttpResponseNotFound("Task not found")
    
    write_answer(task_id, answer)
    return JsonResponse({"ok": True})