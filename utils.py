import streamlit as st

def fancy_control(label, key, step, min_v, max_v, disabled=False):
    c1, c2, c3 = st.columns([1, 2.5, 1])
    st.session_state.setdefault(key, min_v)

    def on_minus():
        new_val = st.session_state[key] - step
        st.session_state[key] = round(max(min_v, new_val), 3)

    def on_plus():
        new_val = st.session_state[key] + step
        st.session_state[key] = round(min(max_v, new_val), 3)

    if c1.button("➖", key=f"{key}_m", disabled=disabled, on_click=on_minus): pass
    if c3.button("➕", key=f"{key}_p", disabled=disabled, on_click=on_plus): pass

    return c2.slider(key, min_v, max_v, key=key, step=step, label_visibility="collapsed", disabled=disabled)
