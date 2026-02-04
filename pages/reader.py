import json
import streamlit as st
import app.pages as pages
import app.utils as utils


def _split_style_name(name: str):
    if "_" in name:
        speaker, audience = name.split("_", 1)
        return speaker.strip(), audience.strip()
    return name.strip(), "General"


def _format_section_title(text: str) -> str:
    return str(text).replace("_", " ").replace("-", " ").strip()


def _value_to_editor_text(value) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    if isinstance(value, list):
        return "\n".join([str(v) for v in value])
    return "" if value is None else str(value)


with col2:
    
    uploaded_files = st.file_uploader(
        ":blue[**Upload Files:**]",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        help="Upload PDF, Word, or PowerPoint files (Max 50MB per file recommended)",
        key="content_upload",
    )
    
    # Show file size warning
    if uploaded_files:
        total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)  # Convert to MB
        if total_size > 100:
            st.warning(f"⚠️ Total file size: {total_size:.1f}MB. Large files may take longer to process.")

# Additional Prompt Instruction (Optional)
st.session_state.setdefault("additional_instruction_reader", "")
st.session_state.additional_instruction_reader = st.text_area(
    ":blue[**Additional Prompt Instruction (Optional):**]",
    st.session_state.additional_instruction_reader,
    height=100,
    key="additional_instruction_reader_input",
    help="Add any additional instructions to combine with the extracted style...",
)
st.caption("This will be appended to the extracted style when saved.")

# Extract text from uploaded files
extracted_text = ""
if uploaded_files:
    progress_bar = st.progress(0, text="Processing uploaded files...")
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            file_type = uploaded_file.name.split('.')[-1].lower()
            file_size_mb = uploaded_file.size / (1024 * 1024)
            
            # Update progress
            progress_bar.progress(
                (idx) / len(uploaded_files),
                text=f"Processing {uploaded_file.name} ({file_size_mb:.1f}MB)..."
            )
            
            # Read file content once
            file_content = uploaded_file.read()
            
            if not file_content:
                st.warning(f"⚠️ {uploaded_file.name} is empty or couldn't be read. Skipping.")
                continue
            
            if file_type == 'pdf':
                pdf_reader = PdfReader(BytesIO(file_content))
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"
                    
            elif file_type == 'docx':
                doc = Document(BytesIO(file_content))
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        extracted_text += paragraph.text + "\n"
                        
            elif file_type == 'pptx':
                prs = Presentation(BytesIO(file_content))
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text.strip():
                            extracted_text += shape.text + "\n"
            
            # Clear file content from memory
            del file_content
            
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    # Complete progress
    progress_bar.progress(1.0, text="✓ All files processed!")
    
    # Clear progress bar after a moment
    import time
    time.sleep(0.5)
    progress_bar.empty()


# ----------------------------
# PICK EXACTLY ONE CONTENT SOURCE + PREVIEW
# ----------------------------
has_input = bool(st.session_state.content.strip())
has_uploads = bool(extracted_text.strip())

# Let the user choose if both are present; otherwise auto-pick
if has_input and has_uploads:
    source = st.radio(
        ":blue[**Choose content source**]",
        ["Uploaded files", "Manual input"],
        horizontal=True,
        index=0,
        help="Use either the text you typed OR the text extracted from the uploaded files."
    )
elif has_uploads:
    source = "Uploaded files"
elif has_input:
    source = "Manual input"
else:
    source = None

# Decide combined_text and show a preview when using uploads
if source == "Uploaded files":
    combined_text = extracted_text  # keep full unicode; your PDF builder handles fonts
    with st.expander("📄 Preview: Extracted text from uploaded files", expanded=True):
        st.text_area(
            "Extracted Text",
            combined_text,
            height=240,
            key="uploaded_text_preview",
        )
elif source == "Manual input":
    combined_text = st.session_state.content
else:
    combined_text = ""
    st.markdown(
    """
    <style>
    .stButton{width:100%;}
    .stButton > button{
        width:100%;
        border-radius:12px;
        padding:0.95rem 1.1rem;
        text-align:center;
        min-height:56px;
        height:56px;
        display:flex;
        align-items:center;
        justify-content:center;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
        border:1px solid rgba(0,0,0,.12);
        background:linear-gradient(180deg,#ffffff 0%,#f7f8fb 100%);
        box-shadow:0 6px 14px rgba(15,23,42,.08), 0 1px 2px rgba(15,23,42,.06);
        font-weight:600;
        transition:transform .12s ease, box-shadow .12s ease, border-color .12s ease;
    }
    .stButton > button:hover{
        border-color:rgba(30,64,175,.35);
        box-shadow:0 10px 20px rgba(15,23,42,.12), 0 2px 6px rgba(15,23,42,.08);
        transform:translateY(-1px);
    }
    .stButton > button:active{
        transform:translateY(0);
        box-shadow:0 4px 10px rgba(15,23,42,.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

styles = utils.get_styles()
if not styles:
    st.info("No styles found.")
    st.stop()

speaker_map = {}
global_rulebooks = {}
for item in utils.get_rulebooks():
    rulebook_id = item.get("id") or item.get("container_key")
    if rulebook_id:
        global_rulebooks[rulebook_id] = item
for item in styles:
    if item.get("doc_kind") and item.get("doc_kind") != "style_fingerprint":
        continue
    name = item.get("name") or item.get("container_key") or item.get("id") or ""
    speaker = item.get("speaker")
    audience = item.get("audience_setting_classification")
    if not speaker or not audience:
        speaker, audience = _split_style_name(str(name))
    speaker_map.setdefault(speaker, {})[audience] = item

st.session_state.setdefault("editor_selected_speaker", None)
st.session_state.setdefault("editor_selected_audience", None)
st.session_state.setdefault("selected_global_rulebook_id", None)

if st.button(
    ":blue[**Extract Writing Style**]",
    key="extract",
    disabled=(
        combined_text.strip() == ""
        or st.session_state.styleName == ""
    ),
):
    with st.spinner("Processing..."):
        # Check if style name already exists
        if utils.check_style(st.session_state.styleName):
            st.session_state["extraction_error"] = f"Style name '{st.session_state.styleName}' already exists. Please choose a different name."
            st.session_state["extraction_success"] = False
        else:
            style = prompts.extract_style(combined_text, False)
            utils.save_style(style, combined_text)
            st.session_state["extracted_style"] = style
            st.session_state["extracted_style_name"] = st.session_state.styleName
            st.session_state["extraction_success"] = True
            st.session_state["extraction_error"] = None
        st.rerun()

# Display extraction results persistently (only once)
if st.session_state.get("extraction_success"):
    with st.container(border=True):
        st.success(f"✅ Style '{st.session_state.get('extracted_style_name')}' extracted and saved successfully!")
        
        with st.expander("📋 View Extracted Style", expanded=False):
            st.text_area(
                "Extracted Writing Style",
                st.session_state.get("extracted_style", ""),
                height=300,
                key="extracted_style_display",
            )

if st.session_state.get("extraction_error"):
    with st.container(border=True):
        st.error(st.session_state["extraction_error"])
