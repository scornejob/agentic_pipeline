"""
AWS tools: fetch CloudWatch alarms.

Credentials are resolved by boto3 in the standard order:
  1. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
  2. ~/.aws/credentials (mounted from host if desired)
  3. IAM instance/task role (when running on EC2/ECS/EKS)

Optional environment variables:
    AWS_DEFAULT_REGION  — AWS region (default: us-east-1)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from rich.console import Console

from src.agent.tools import _parse_json_input, tool

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _boto3():
    try:
        import boto3
        return boto3
    except ImportError:
        raise ImportError("boto3 is not installed. Add it to requirements.txt and rebuild the container.")


def _cw_client(region: str):
    return _boto3().client("cloudwatch", region_name=region)


def _print_aws_context(region: str) -> None:
    try:
        sts = _boto3().client("sts", region_name=region)
        account = sts.get_caller_identity()["Account"]
    except Exception:
        account = "unknown"
    try:
        iam = _boto3().client("iam", region_name=region)
        aliases = iam.list_account_aliases()["AccountAliases"]
        account_display = f"{account} ({aliases[0]})" if aliases else account
    except Exception:
        account_display = account
    console.print(f"[dim]AWS  account=[cyan]{account_display}[/cyan]  region=[cyan]{region}[/cyan][/dim]")


def _alarm_summary(alarm: dict) -> dict:
    """Extract the most relevant fields from a CloudWatch alarm dict."""
    last_updated = alarm.get("StateUpdatedTimestamp")
    return {
        "name": alarm.get("AlarmName", ""),
        "state": alarm.get("StateValue", ""),
        "reason": alarm.get("StateReason", "")[:300],
        "metric": alarm.get("MetricName", ""),
        "namespace": alarm.get("Namespace", ""),
        "dimensions": {d["Name"]: d["Value"] for d in alarm.get("Dimensions", [])},
        "actions_enabled": alarm.get("ActionsEnabled", True),
        "alarm_arn": alarm.get("AlarmArn", ""),
        "last_updated": last_updated.astimezone(timezone.utc).isoformat() if isinstance(last_updated, datetime) else str(last_updated or ""),
    }


# ── Tool ──────────────────────────────────────────────────────────────────────

@tool(
    name="cloudwatch_alarms",
    description=(
        "List AWS CloudWatch alarms. Filter by state, name prefix, or namespace. "
        "Returns alarm name, state, triggering metric, and reason."
    ),
    usage=(
        'Action Input: {"state": "ALARM", "prefix": "", "namespace": "", "region": "eu-central-1"}\n'
        '  state:     ALARM | OK | INSUFFICIENT_DATA | all  (default: all)\n'
        '  prefix:    filter alarms whose name starts with this string  (optional, leave empty for all)\n'
        '  namespace: filter by CloudWatch metric namespace, e.g. AWS/RDS, AWS/EC2, AWS/Lambda  (optional, leave empty for all)\n'
        '  region:    AWS region                                        (default: AWS_DEFAULT_REGION or us-east-1)\n'
        '  limit:     max results to return                             (default: 50)'
    ),
)
def _cloudwatch_alarms(input_str: str) -> str:
    data = _parse_json_input(input_str)
    state_filter = data.get("state", "all").strip().upper()
    prefix = data.get("prefix", "").strip()
    namespace_filter = data.get("namespace", "").strip()
    region = data.get("region", "") or os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
    limit = int(data.get("limit", 50))

    try:
        cw = _cw_client(region)
        _print_aws_context(region)
    except ImportError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error creating CloudWatch client: {exc}"

    try:
        kwargs: dict = {}

        if state_filter != "ALL":
            valid_states = {"ALARM", "OK", "INSUFFICIENT_DATA"}
            if state_filter not in valid_states:
                return f"Error: state must be one of {sorted(valid_states)} or 'all'."
            kwargs["StateValue"] = state_filter

        if prefix:
            kwargs["AlarmNamePrefix"] = prefix

        results = []
        paginator = cw.get_paginator("describe_alarms")

        for page in paginator.paginate(
            **kwargs,
            AlarmTypes=["MetricAlarm", "CompositeAlarm"],
            PaginationConfig={"PageSize": 100},
        ):
            for alarm in page.get("MetricAlarms", []) + page.get("CompositeAlarms", []):
                if namespace_filter and namespace_filter.lower() not in alarm.get("Namespace", "").lower():
                    continue
                results.append(_alarm_summary(alarm))
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break

        if not results:
            return (
                f"No alarms found matching filters "
                f"(state={state_filter!r}, prefix={prefix!r}, namespace={namespace_filter!r}, region={region!r})."
            )

        summary = {
            "total_returned": len(results),
            "truncated_at_limit": len(results) >= limit,
            "filters": {"state": state_filter, "prefix": prefix, "namespace": namespace_filter, "region": region},
            "alarms": results,
        }
        return json.dumps(summary, indent=2)

    except Exception as exc:
        return f"AWS CloudWatch error: {exc}"
