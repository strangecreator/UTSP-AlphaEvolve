from django.urls import path
from .views import index, api_tasks, api_task_detail, api_task_answer


urlpatterns = [
    path("", index, name="index"),
    path("api/tasks", api_tasks, name="api_tasks"),
    path("api/tasks/<str:task_id>", api_task_detail, name="api_task_detail"),
    path("api/tasks/<str:task_id>/answer", api_task_answer, name="api_task_answer"),
]