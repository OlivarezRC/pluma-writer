import streamlit as st
import app.utils as utils
from dotenv import load_dotenv

load_dotenv()

def show_home():
    st.set_page_config(
        page_title="FSS ROE Writer",
        page_icon="✍️",
        layout="wide",
        initial_sidebar_state="expanded",  # make the left sidebar visible
    )

    # --- Sidebar "title page" ---
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/c/cb/Bangko_Sentral_ng_Pilipinas_2020_logo.png",
           width='stretch',
        )
        st.markdown("### ✍️ FSS ROE Writer")
        st.caption("Style Writer • Edit, format, and validate content against your style guide.")
        st.divider()
        # (Optional) 
        # st.session_state.current_view = st.radio(
        #     "Navigate",
        #     ["Home", "Guidelines", "Examples"],
        #     label_visibility="collapsed",
        # )

    # Keep the top app logo in the header if you still want it
    st.logo(
        "https://upload.wikimedia.org/wikipedia/commons/c/cb/Bangko_Sentral_ng_Pilipinas_2020_logo.png",
        link="https://www.bsp.gov.ph/SitePages/Default.aspx",
    )

    # --- Initial states ---
    if "content" not in st.session_state:
        st.session_state.content = ""
    if "style" not in st.session_state:
        st.session_state.style = ""
    if "styleName" not in st.session_state:
        st.session_state.styleName = ""
    if "guidelines" not in st.session_state:
        st.session_state.guidelines = ""
    if "example" not in st.session_state:
        st.session_state.example = ""
    if "exampleText" not in st.session_state:
        st.session_state.exampleText = ""
    if "locals" not in st.session_state:
        st.session_state.locals = utils.read_json("data/local_data.json")

    # --- CSS tweaks (optional) ---
    st.markdown(
        """
        <style>
        .block-container { padding-top: 3rem; }
        .stAppDeployButton, .st-emotion-cache-15ecox0,
        .viewerBadge_container__r5tak, .styles_viewerBadge__CvC9N { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# display the sidebar
def show_sidebar():
    with st.sidebar:
        with st.container(border=True):
            st.page_link("app.py", label="Style Writer", icon="📝")
            st.page_link("pages/reader.py", label="Style Reader", icon="🔍")
            st.page_link("pages/outputs.py", label="Generated Outputs", icon="📰")
            st.page_link("pages/settings.py", label="Settings", icon="⚙️")

        # st.image(
        #     "https://sa.kapamilya.com/absnews/abscbnnews/media/2020/business/11/19/20170731-bsp-md-2.jpg",
        # )
        st.write("Powered by LikhAI.")
