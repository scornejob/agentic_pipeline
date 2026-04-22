"""
Datadog tools: check monitors (alerts), dashboards, and services.

Required environment variables (set in .env):
    DD_API_KEY   — Datadog API key
    DD_APP_KEY   — Datadog application key
    DD_SITE      — Datadog site (default: datadoghq.eu)
"""
from __future__ import annotations

import json
import os

from src.agent.tools import _parse_json_input, tool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dd_config():
    """Build a Datadog Configuration from environment variables."""
    from datadog_api_client import Configuration  # lazy import

    api_key = os.environ.get("DD_API_KEY", "")
    app_key = os.environ.get("DD_APP_KEY", "")
    site = os.environ.get("DD_SITE", "datadoghq.eu")

    if not api_key or not app_key:
        raise EnvironmentError(
            "DD_API_KEY and DD_APP_KEY must be set in your .env file to use Datadog tools."
        )

    cfg = Configuration()
    cfg.api_key["apiKeyAuth"] = api_key
    cfg.api_key["appKeyAuth"] = app_key
    cfg.server_variables["site"] = site
    return cfg


def _monitor_summary(m) -> dict:
    """Extract the most relevant fields from a Monitor object."""
    return {
        "id": m.id,
        "name": m.name,
        "type": str(m.type),
        "state": str(m.overall_state) if m.overall_state else "Unknown",
        "tags": list(m.tags) if m.tags else [],
        "query": m.query if hasattr(m, "query") else "",
        "message": (m.message or "")[:300],
    }


def _dashboard_summary(d) -> dict:
    """Extract the most relevant fields from a DashboardSummaryDefinition."""
    return {
        "id": d.id,
        "title": d.title,
        "url": f"https://app.datadoghq.com/dashboard/{d.id}",
        "author": d.author_name if hasattr(d, "author_name") and d.author_name else "",
        "modified": str(d.modified_at) if hasattr(d, "modified_at") and d.modified_at else "",
    }


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool(
    name="datadog_monitors",
    description=(
        "Query Datadog monitors (alerts). Filter by status, tags, or name. "
        "Returns monitor names, states, and messages."
    ),
    usage=(
        'Action Input: {"status": "Alert", "tags": "team:backend,env:prod", "name": ""}\n'
        '  status: Alert | Warn | No Data | OK | all  (default: all)\n'
        '  tags:   comma-separated tag filters         (optional)\n'
        '  name:   substring to search in monitor name (optional)\n'
        '  limit:  max results to return               (default: 50)'
    ),
)
def _datadog_monitors(input_str: str) -> str:
    try:
        from datadog_api_client import ApiClient
        from datadog_api_client.v1.api.monitors_api import MonitorsApi
    except ImportError:
        return "Error: datadog-api-client is not installed. Add it to requirements.txt and rebuild."

    data = _parse_json_input(input_str)
    status_filter = data.get("status", "all").strip()
    tag_filter = data.get("tags", "").strip()
    name_filter = data.get("name", "").strip().lower()
    limit = int(data.get("limit", 50))

    try:
        cfg = _dd_config()
    except EnvironmentError as exc:
        return f"Configuration error: {exc}"

    try:
        kwargs: dict = {}
        if tag_filter:
            kwargs["monitor_tags"] = tag_filter
        if name_filter:
            kwargs["name"] = name_filter

        with ApiClient(cfg) as client:
            api = MonitorsApi(client)
            monitors = api.list_monitors(**kwargs)

        results = []
        for m in monitors[:limit]:
            summary = _monitor_summary(m)
            # Apply status filter client-side for exact match
            if status_filter.lower() != "all":
                if summary["state"].lower() != status_filter.lower():
                    continue
            results.append(summary)

        if not results:
            return f"No monitors found matching the given filters (status={status_filter!r}, tags={tag_filter!r}, name={name_filter!r})."

        return json.dumps(results, indent=2)

    except Exception as exc:
        return f"Datadog API error: {exc}"


@tool(
    name="datadog_dashboards",
    description=(
        "List Datadog dashboards or retrieve the widget layout of a specific dashboard."
    ),
    usage=(
        'Action Input: {"action": "list", "query": "optional title search"}\n'
        '  action: "list" to see all dashboards (default)\n'
        '          "get"  to fetch a specific dashboard — also provide "id"\n'
        '  query:  filter by title substring when action=list\n'
        '  id:     dashboard id (e.g. "abc-123-xyz") when action=get\n'
        '  limit:  max results for list              (default: 50)'
    ),
)
def _datadog_dashboards(input_str: str) -> str:
    try:
        from datadog_api_client import ApiClient
        from datadog_api_client.v1.api.dashboards_api import DashboardsApi
    except ImportError:
        return "Error: datadog-api-client is not installed. Add it to requirements.txt and rebuild."

    data = _parse_json_input(input_str)
    action = data.get("action", "list").strip().lower()
    query = data.get("query", "").strip().lower()
    dashboard_id = data.get("id", "").strip()
    limit = int(data.get("limit", 50))

    try:
        cfg = _dd_config()
    except EnvironmentError as exc:
        return f"Configuration error: {exc}"

    try:
        with ApiClient(cfg) as client:
            api = DashboardsApi(client)

            if action == "get":
                if not dashboard_id:
                    return 'Error: "id" is required when action is "get".'
                dashboard = api.get_dashboard(dashboard_id)
                # Return title, description, and widget titles
                widgets = []
                for w in (dashboard.widgets or []):
                    if hasattr(w, "definition") and hasattr(w.definition, "title"):
                        widgets.append(w.definition.title)
                return json.dumps(
                    {
                        "id": dashboard.id,
                        "title": dashboard.title,
                        "description": dashboard.description or "",
                        "url": f"https://app.datadoghq.com/dashboard/{dashboard.id}",
                        "widgets": widgets,
                    },
                    indent=2,
                )

            # action == "list"
            response = api.list_dashboards()
            dashboards = response.dashboards or []

            results = []
            for d in dashboards:
                summary = _dashboard_summary(d)
                if query and query not in summary["title"].lower():
                    continue
                results.append(summary)
                if len(results) >= limit:
                    break

            if not results:
                return f"No dashboards found matching query={query!r}."

            return json.dumps(results, indent=2)

    except Exception as exc:
        return f"Datadog API error: {exc}"


@tool(
    name="datadog_services",
    description=(
        "List services tracked in Datadog Service Catalog. Filter by name, team, or language."
    ),
    usage=(
        'Action Input: {"name": "payments", "team": "backend", "language": "python"}\n'
        '  name:     substring to filter by service name     (optional)\n'
        '  team:     filter by owning team                   (optional)\n'
        '  language: filter by programming language          (optional)\n'
        '  limit:    max results to return                   (default: 50)'
    ),
)
def _datadog_services(input_str: str) -> str:
    try:
        from datadog_api_client import ApiClient
        from datadog_api_client.v2.api.software_catalog_api import SoftwareCatalogApi
    except ImportError:
        return "Error: datadog-api-client is not installed. Add it to requirements.txt and rebuild."

    data = _parse_json_input(input_str)
    name_filter = data.get("name", "").strip().lower()
    team_filter = data.get("team", "").strip().lower()
    lang_filter = data.get("language", "").strip().lower()
    limit = int(data.get("limit", 50))

    try:
        cfg = _dd_config()
    except EnvironmentError as exc:
        return f"Configuration error: {exc}"

    try:
        with ApiClient(cfg) as client:
            api = SoftwareCatalogApi(client)
            response = api.list_catalog_entity(page_limit=500)

        entities = response.data or []
        results = []

        for entity in entities:
            attrs = entity.attributes if hasattr(entity, "attributes") else None
            if attrs is None:
                continue

            # Extract common fields defensively
            kind = str(getattr(attrs, "kind", "") or "").lower()
            if kind not in ("service", ""):
                continue  # only services

            name = str(getattr(attrs, "name", "") or "")
            teams = []
            if hasattr(attrs, "teams") and attrs.teams:
                teams = [str(t) for t in attrs.teams]
            languages = []
            if hasattr(attrs, "languages") and attrs.languages:
                languages = [str(l) for l in attrs.languages]

            # Apply filters
            if name_filter and name_filter not in name.lower():
                continue
            if team_filter and not any(team_filter in t.lower() for t in teams):
                continue
            if lang_filter and not any(lang_filter in l.lower() for l in languages):
                continue

            results.append({
                "name": name,
                "kind": kind or "service",
                "teams": teams,
                "languages": languages,
                "description": str(getattr(attrs, "description", "") or "")[:200],
                "tags": list(getattr(attrs, "tags", None) or []),
            })

            if len(results) >= limit:
                break

        if not results:
            return (
                f"No services found matching filters "
                f"(name={name_filter!r}, team={team_filter!r}, language={lang_filter!r})."
            )

        return json.dumps(results, indent=2)

    except Exception as exc:
        return f"Datadog API error: {exc}"
