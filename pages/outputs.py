import streamlit as st
import app.pages as pages
import app.utils as utils

# App title
pages.show_home()
pages.show_sidebar()

st.header("📰 Generated Outputs")

_writer_column_config = {
    "updatedAt": st.column_config.DatetimeColumn("Created At", format="YYYY-MM-DD HH:mm:ss", width="medium"),
    "user_name": st.column_config.TextColumn("User Email", width="small"),
    "styleId": st.column_config.TextColumn("Style ID", width="small"),
    "content": st.column_config.TextColumn("Input Content", width="medium", help="Original content"),
    "keywords": st.column_config.TextColumn("Keywords", width="medium", help="Keywords for deep research"),
    "attachment_urls": st.column_config.ListColumn("Attachment Files", width="medium"),
    "source_links": st.column_config.ListColumn("Source Links", width="medium"),
    "additional_instructions": st.column_config.TextColumn("Additional Instructions", width="medium"),
    "target_word_count": st.column_config.NumberColumn("Target Words", width="small"),
    "editorial_style_guides": st.column_config.ListColumn("Editorial Style Guides", width="medium"),
    "pipeline_stages": st.column_config.ListColumn("Pipeline Stages", width="medium"),
    "output": st.column_config.TextColumn("Output", width="large", help="Rewritten output"),
    "output_word_count": st.column_config.NumberColumn("Output Words", width="small"),
    "pdf": st.column_config.LinkColumn("Output PDF", display_text="Download PDF", width="small"),
    "docx": st.column_config.LinkColumn("Output DOCX", display_text="Download DOCX", width="small"),
}

_refiner_column_config = {
    "updatedAt": st.column_config.DatetimeColumn("Created At", format="YYYY-MM-DD HH:mm:ss", width="medium"),
    "user_name": st.column_config.TextColumn("User Email", width="small"),
    "styleId": st.column_config.TextColumn("Style ID", width="small"),
    "content": st.column_config.TextColumn("Input Content", width="medium", help="Original content"),
    "additional_instructions": st.column_config.TextColumn("Additional Instructions", width="medium"),
    "editorial_style_guides": st.column_config.ListColumn("Editorial Style Guides", width="medium"),
    "target_word_count": st.column_config.NumberColumn("Target Words", width="small"),
    "output": st.column_config.TextColumn("Output", width="large", help="Rewritten output"),
    "output_word_count": st.column_config.NumberColumn("Output Words", width="small"),
    "pdf": st.column_config.LinkColumn("Output PDF", display_text="Download PDF", width="small"),
    "docx": st.column_config.LinkColumn("Output DOCX", display_text="Download DOCX", width="small"),
}

# Speech Writer Outputs Section
st.subheader("📝 Speech Writer Outputs")
with st.spinner("Loading Speech Writer outputs..."):
    writer_df = utils.get_style_writer_outputs()
    if writer_df is not None and not writer_df.empty:
        st.dataframe(
            writer_df,
            column_config=_writer_column_config,
            height=600,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No Speech Writer outputs found.")

st.divider()

# Speech Refiner Outputs Section
st.subheader("🔧 Speech Refiner Outputs")
with st.spinner("Loading Speech Refiner outputs..."):
    refiner_df = utils.get_style_refiner_outputs()
    if refiner_df is not None and not refiner_df.empty:
        st.dataframe(
            refiner_df,
            column_config=_refiner_column_config,
            height=600,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No Speech Refiner outputs found.")


