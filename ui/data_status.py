"""User-facing data-source summary for the sidebar header."""

from services.price_service import get_data_load_status


def get_data_source_summary():
    status = get_data_load_status()
    configured_source = status["configured_source"]
    resolved_source = status.get("resolved_source")
    snapshot_updated_at = status.get("snapshot_updated_at")
    fallback_reason = status.get("fallback_reason")

    if resolved_source == "live":
        return "Data: live"

    source_label = "snapshot" if resolved_source == "snapshot" else configured_source
    summary = f"Data: {source_label}"
    if fallback_reason:
        summary = f"⚠ {summary} fallback"
    if snapshot_updated_at:
        summary = f"{summary} · Updated: {snapshot_updated_at}"
    return summary
