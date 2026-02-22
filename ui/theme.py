from pathlib import Path
from functools import lru_cache

import streamlit as st

THEMES = {
    "dark": {
        "c_main_bg": "#0e1117",
        "c_sidebar_bg": "#161a25",
        "c_card_bg": "#1e222d",
        "c_border": "#2d323e",
        "c_accent": "#f0b90b",
        "c_accent_text": "#111827",
        "c_control_bg": "#1b2130",
        "c_text_main": "#d1d4dc",
        "c_text_val": "#ffffff",
        "c_btn_bg": "#2d323e",
        "c_btn_hover": "#3d4251",
        "c_btn_text": "#d1d4dc",
        "pl_template": "plotly_dark",
        "pl_bg_color": "rgba(0,0,0,0)",
        "pl_grid_color": "#1e222d",
        "pl_btc_color": "#ffffff",
        "pl_legend_color": "#848e9c",
        "pl_text_color": "#848e9c",
        "c_hover_bg": "#1e222d",
        "c_hover_text": "#ffffff",
    },
}


def get_theme(_is_dark=True):
    return THEMES["dark"]


@lru_cache(maxsize=1)
def load_css_template(css_path):
    return Path(css_path).read_text(encoding="utf-8")


def apply_theme_css(theme, css_path="assets/styles.css"):
    css_template = load_css_template(css_path)
    css = css_template
    for key, value in theme.items():
        css = css.replace(f"__{key.upper()}__", str(value))
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
