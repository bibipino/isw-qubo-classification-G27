from __future__ import annotations

import streamlit as st

from ui.components.sidebar import render_sidebar
from ui.pages.router import render_page
from ui.state import initialize_state


def main() -> None:
    st.set_page_config(
        page_title="QUBO Binary Classification",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_state()

    selected_page = render_sidebar()

    render_page(selected_page)


if __name__ == "__main__":
    main()