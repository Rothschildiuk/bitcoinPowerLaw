"""Small, user-facing disclosure of the data source behind the current chart."""

import streamlit as st

from services.price_service import get_data_load_status


def render_data_source_status():
    status = get_data_load_status()
    configured_source = status["configured_source"]
    resolved_source = status.get("resolved_source")
    snapshot_updated_at = status.get("snapshot_updated_at")
    fallback_reason = status.get("fallback_reason")

    st.divider()
    if resolved_source == "live":
        st.caption(f"Data source: live (mode: {configured_source})")
        return

    snapshot_caption = f" Snapshot: {snapshot_updated_at}." if snapshot_updated_at else ""
    if resolved_source == "snapshot":
        st.caption(f"Data source: local snapshot (mode: {configured_source}).{snapshot_caption}")
        if fallback_reason:
            st.warning(
                "Live refresh was unavailable; the chart uses the last valid snapshot.",
                icon="⚠️",
            )
            with st.expander("Refresh details"):
                st.code(fallback_reason, language=None)
        return

    st.caption(f"Data mode: {configured_source}.{snapshot_caption}")
