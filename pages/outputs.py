import streamlit as st
import app.pages as pages
import app.utils as utils

# App title
pages.show_home()
pages.show_sidebar()

st.header("📰 Generated Outputs")

_column_config = {
    "updatedAt": st.column_config.DatetimeColumn("Created At", format="YYYY-MM-DD HH:mm:ss", width="medium"),
    "user_name": st.column_config.TextColumn("User Email", width="small"),
    "styleId": st.column_config.TextColumn("Style ID", width="small"),
    "content": st.column_config.TextColumn("Input Content", width="medium", help="Original content"),
    "output": st.column_config.TextColumn("Output", width="large", help="Rewritten output"),
    "pdf": st.column_config.LinkColumn("Output PDF", display_text="Download PDF", width="small"),
    "docx": st.column_config.LinkColumn("Output DOCX", display_text="Download DOCX", width="small"),
}

# Style Writer Outputs Section
st.subheader("📝 Style Writer Outputs")
with st.spinner("Loading Style Writer outputs..."):
    writer_df = utils.get_style_writer_outputs()
    if writer_df is not None and not writer_df.empty:
        st.dataframe(
            writer_df,
            column_config=_column_config,
            height=600,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No Style Writer outputs found.")

st.divider()

# Style Refiner Outputs Section
st.subheader("🔧 Style Refiner Outputs")
with st.spinner("Loading Style Refiner outputs..."):
    refiner_df = utils.get_style_refiner_outputs()
    if refiner_df is not None and not refiner_df.empty:
        st.dataframe(
            refiner_df,
            column_config=_column_config,
            height=600,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No Style Refiner outputs found.")


