"""Client viewport hints.

Streamlit builds the page on the server, so anything baked into the payload cannot be
fixed later by CSS: Plotly figure heights, tick density, legend sizing and the modebar
button list are all decided in Python. The request User-Agent is the only viewport
signal Streamlit exposes, so it drives those server-side choices. Everything a
stylesheet can handle on its own stays in `assets/styles.css` media queries.
"""

import re

import streamlit as st

from core.constants import KEY_CLIENT_IS_MOBILE

# Android tablets omit "Mobile", so requiring it keeps them on the desktop layout.
_MOBILE_USER_AGENT_PATTERN = re.compile(
    r"Android.+Mobile|iPhone|iPod|IEMobile|Windows Phone|BlackBerry|Opera Mini",
    re.IGNORECASE,
)


def _read_user_agent():
    try:
        headers = st.context.headers
    except Exception:
        return ""
    if not headers:
        return ""
    return str(headers.get("User-Agent") or headers.get("user-agent") or "")


def user_agent_is_mobile(user_agent):
    return bool(_MOBILE_USER_AGENT_PATTERN.search(str(user_agent or "")))


def is_mobile_client():
    """True when the session was opened from a phone-sized client.

    The User-Agent cannot change within a session, so the answer is resolved once and
    cached in session state.
    """
    cached = st.session_state.get(KEY_CLIENT_IS_MOBILE)
    if cached is not None:
        return bool(cached)

    resolved = user_agent_is_mobile(_read_user_agent())
    st.session_state[KEY_CLIENT_IS_MOBILE] = resolved
    return resolved
