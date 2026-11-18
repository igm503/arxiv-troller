from django.http import JsonResponse
from django.shortcuts import render
from psycopg2.errors import QueryCanceled

TIMEOUT_MESSAGE = "That search took too long. Try a shorter date range or more specific words."


class QueryTimeoutMiddleware:
    """Answer a query stopped by the statement timeout (settings.WEB_DB_SESSION) with a 503 that says so."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_exception(self, request, exception):
        if not isinstance(exception.__cause__, QueryCanceled):
            return None
        if request.path.startswith("/api/") or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({"ok": False, "success": False, "error": TIMEOUT_MESSAGE}, status=503)
        context = {
            "query": request.GET.get("q", ""),
            "query_error": TIMEOUT_MESSAGE,
            "date_filter": request.GET.get("date_filter") or "1week",
            "category_filter": request.GET.get("category", ""),
            "results": [],
        }
        return render(request, "papers/search.html", context, status=503)
