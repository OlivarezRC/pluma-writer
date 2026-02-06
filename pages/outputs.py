import streamlit as st
import app.pages as pages
import app.utils as utils

# App title
pages.show_home()
pages.show_sidebar()

st.header("📰 Generated Outputs")

# Style Writer Outputs Section
st.subheader("Style Writer Outputs")
with st.spinner("Loading Style Writer outputs..."):
    outputs_df = utils.get_outputs()
    if outputs_df is not None:
        st.dataframe(
            outputs_df,
            column_config={
                "updatedAt": st.column_config.DatetimeColumn("Created At", format="YYYY-MM-DD HH:mm:ss", width="medium"),
                "user_name": st.column_config.TextColumn("User Email", width="small"),
                "styleId": st.column_config.TextColumn("Style ID", width="small"),
                "content": st.column_config.TextColumn("Input Content", width="medium", help="Original content"),
                "output": st.column_config.TextColumn("Output", width="large", help="Rewritten output"),
                "pdf": st.column_config.LinkColumn("Output PDF", display_text="Download PDF", width="small"),
                "docx": st.column_config.LinkColumn("Output DOCX", display_text="Download DOCX", width="small")
            },
            height=600,
            use_container_width=True,
            hide_index=True
        )


