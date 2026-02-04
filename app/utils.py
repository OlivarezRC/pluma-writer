import os
import json
import time
import requests
import pandas as pd
import streamlit as st

from datetime import datetime
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, exceptions, PartitionKey
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import hashlib
from io import BytesIO
# from azure.identity import DefaultAzureCredential

load_dotenv()
# create a config dictionary
config = {
    "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_key": os.getenv("AZURE_OPENAI_KEY"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    "model": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
}

# Initialize OpenAI client
clientAOAI = AzureOpenAI(
    azure_endpoint=config["endpoint"],
    api_version=config["api_version"],
    api_key=config["api_key"],
)

# Initialize Azure AD credential
# credential = DefaultAzureCredential()


def ensure_containers_exist():
    try:
        # Define container configurations
        containers_config = [
            {
                "name": "styles_from_speeches",
                "partition_key": PartitionKey(path="/user_id"),
                "unique_keys": [{"paths": ["/name"]}]
            },
            {
                "name": "outputs",
                "partition_key": PartitionKey(path="/user_id")
            }
        ]
        
        for config in containers_config:
            try:
                # Try to read the container to check if it exists
                database.get_container_client(config["name"])
            except exceptions.CosmosResourceNotFoundError:
                # Container doesn't exist, create it
                database.create_container(
                    id=config["name"],
                    partition_key=config["partition_key"],
                    unique_keys=config.get("unique_keys", [])
                )
    except Exception as e:
        st.error(f"Error ensuring containers exist: {e}")

# Initialize Cosmos DB client with token credentials
cosmos_client = CosmosClient(
    url=os.getenv("AZURE_COSMOS_ENDPOINT"),
    credential=os.getenv("AZURE_COSMOS_KEY")
    # credential=credential
)

# Get database and container references
database = cosmos_client.get_database_client(os.getenv("AZURE_COSMOS_DATABASE"))

# Ensure containers exist with proper configuration
ensure_containers_exist()

# Get container references after ensuring they exist
styles_container = database.get_container_client("styles_from_speeches")
outputs_container = database.get_container_client("outputs")

# Initialize Azure Blob Storage client
try:
    storage_account = os.getenv('APP_AZURE_STORAGE_ACCOUNT')
    storage_key = os.getenv('APP_AZURE_STORAGE_ACCESS_KEY')
    bucket_name = os.getenv('BUCKET_NAME')
    
    if not storage_account or not storage_key or not bucket_name:
        print("WARNING: Azure Storage credentials not configured properly")
        blob_service_client = None
    else:
        blob_service_client = BlobServiceClient(
            account_url=f"https://{storage_account}.blob.core.windows.net",
            credential=storage_key
        )
        # Ensure the container exists
        try:
            container_client = blob_service_client.get_container_client(bucket_name)
            if not container_client.exists():
                container_client.create_container()
                print(f"Created blob container: {bucket_name}")
            print(f"Azure Blob Storage initialized: {storage_account}/{bucket_name}")
        except Exception as container_error:
            print(f"Container check/create error: {container_error}")
except Exception as e:
    print(f"Error initializing Azure Blob Storage: {e}")
    blob_service_client = None
    bucket_name = None

# log tracing
def trace(col2, label, message):
    with col2:
        with st.expander(f"{label}:"):
            st.write(message)
            # print(f"{label}: {message}")


# get request api
def get_request(url):
    response = requests.get(url)
    return response.json()


# chat completion
def chat(
    messages=[],
    streaming=True,
    format="text",
):
    try:
        # Response generation
        full_response = ""
        message_placeholder = st.empty()

        for completion in clientAOAI.chat.completions.create(
            model=config["model"],
            messages=messages,
            stream=True,
            response_format={"type": format},
        ):

            if completion.choices and completion.choices[0].delta.content is not None:
                full_response += completion.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        return full_response

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


# Function to read a JSON file
def read_json(file_path):
    try:
        with open(file_path, "r") as file:
            collection = json.load(file)
            return collection
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


# get styles from database
def get_styles():
    try:
        # Fetch all style fingerprints from the new container
        query = """
        SELECT *
        FROM c
        WHERE c.doc_kind = 'style_fingerprint'
        """
        parameters = []
        items = list(styles_container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        return items
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred while fetching styles: {e}")
        return []


def get_rulebook(rulebook_id: str):
    try:
        if not rulebook_id:
            return None
        query = "SELECT * FROM c WHERE c.id = @id OR c.container_key = @id"
        parameters = [{"name": "@id", "value": rulebook_id}]
        items = list(styles_container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        return items[0] if items else None
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred while fetching global rules: {e}")
        return None


def get_rulebooks():
    try:
        query = "SELECT * FROM c WHERE c.doc_kind = 'global_rulebook'"
        items = list(styles_container.query_items(
            query=query,
            parameters=[],
            enable_cross_partition_query=True
        ))
        return items
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred while fetching global rulebooks: {e}")
        return []


def get_style_fingerprint(speaker: str, audience: str):
    try:
        if not speaker or not audience:
            return None
        query = """
        SELECT *
        FROM c
        WHERE c.doc_kind = 'style_fingerprint'
          AND c.speaker = @speaker
          AND c.audience_setting_classification = @audience
        """
        parameters = [
            {"name": "@speaker", "value": speaker},
            {"name": "@audience", "value": audience},
        ]
        items = list(styles_container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        return items[0] if items else None
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred while fetching style rules: {e}")
        return None


def extract_rulebook_text(rulebook: dict) -> str:
    if not rulebook:
        return ""
    for key in ["global_rules", "rules", "rulebook", "rulebook_text"]:
        if key in rulebook and rulebook[key]:
            return rulebook[key]
    try:
        return json.dumps(rulebook, ensure_ascii=False, indent=2)
    except Exception:
        return str(rulebook)


# check if style name exists
def check_style(style_name):
    try:
        # Get current user info
        headers = st.context.headers
        user_id = headers.get('X-MS-CLIENT-PRINCIPAL-ID', '12345')
        
        if not user_id:
            return False
            
        query = "SELECT * FROM c WHERE c.name = @style_name AND c.user_id = @user_id"
        parameters = [
            {"name": "@style_name", "value": style_name},
            {"name": "@user_id", "value": user_id}
        ]
        items = list(styles_container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        return len(items) > 0
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred while checking style name: {e}")
        return False


# Import formatted document generation functions from app.py
def get_document_generators():
    """Lazy import to avoid circular dependency"""
    try:
        import sys
        import os
        # Add parent directory to path if needed
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Import from the app.py file specifically, not the app package
        import importlib.util
        app_py_path = os.path.join(parent_dir, 'app.py')
        spec = importlib.util.spec_from_file_location("app_module", app_py_path)
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        
        return app_module.make_pdf_bytes, app_module.make_docx_bytes
    except Exception as e:
        st.error(f"Error importing document generators: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None


# Generate formatted PDF from text using BSP template
def generate_pdf(text, filename, title=None):
    try:
        make_pdf_bytes, _ = get_document_generators()
        if make_pdf_bytes is None:
            st.error("PDF generator not available")
            return None
        
        pdf_bytes = make_pdf_bytes(text, title=title)
        buffer = BytesIO(pdf_bytes)
        buffer.seek(0)  # Reset to beginning for reading
        return buffer
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None


# Generate formatted DOCX from text using BSP template
def generate_docx(text, filename, title=None):
    try:
        _, make_docx_bytes = get_document_generators()
        if make_docx_bytes is None:
            st.error("DOCX generator not available")
            return None
        
        docx_bytes = make_docx_bytes(text, title=title)
        buffer = BytesIO(docx_bytes)
        buffer.seek(0)  # Reset to beginning for reading
        return buffer
    except Exception as e:
        st.error(f"Error generating DOCX: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None


# Upload file to Azure Blob Storage
def upload_to_blob(file_buffer, filename):
    try:
        # Check if blob storage is initialized
        if blob_service_client is None or bucket_name is None:
            st.error("Azure Blob Storage not configured. Please check environment variables.")
            return None
        
        # Ensure buffer is at the beginning
        file_buffer.seek(0)
        
        blob_client = blob_service_client.get_blob_client(
            container=bucket_name,
            blob=filename
        )
        
        # Read the data from buffer
        data = file_buffer.read()
        
        if not data:
            st.error(f"No data to upload for {filename}")
            return None
        
        st.info(f"Uploading {len(data)} bytes to {bucket_name}/{filename}")
        
        # Upload the data
        blob_client.upload_blob(data, overwrite=True)
        
        # Return the URL
        url = blob_client.url
        st.success(f"File uploaded successfully: {filename}")
        return url
    except Exception as e:
        st.error(f"Error uploading to blob storage: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None


# save style to database
def save_style(style, combined_text):
    try:
        # Get current user info for user_name only
        headers = st.context.headers
        user_name = headers.get('X-MS-CLIENT-PRINCIPAL-NAME', 'r0bai')
        
        now = datetime.now()
        st.session_state.styleId = str(int(time.time() * 1000))
        new_style = {
            "id": st.session_state.styleId,
            "updatedAt": now.isoformat(),
            "name": st.session_state.styleName,
            "style": style,
            "example": combined_text,
            "user_id": "allbsp",  # Set to 'allbsp' to make visible to all users
            "user_name": user_name
        }
        print("Style:")
        print(style)
        styles_container.create_item(body=new_style)
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred while saving style: {e}")


def save_style_fingerprint(style_doc: dict) -> bool:
    try:
        if not style_doc:
            st.warning("No style document to save.")
            return False

        headers = st.context.headers
        user_id = headers.get('X-MS-CLIENT-PRINCIPAL-ID', '12345')

        doc = dict(style_doc)
        if "user_id" not in doc:
            doc["user_id"] = user_id

        doc["updatedAt"] = datetime.now().isoformat()

        for key in [k for k in doc.keys() if k.startswith("_")]:
            doc.pop(key, None)

        styles_container.upsert_item(body=doc)
        return True
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred while saving style: {e}")
        return False


# save output to database
def save_output(output, content_all, pdf_bytes=None, docx_bytes=None, title=None):
    try:
        # Get current user info
        headers = st.context.headers
        user_id = headers.get('X-MS-CLIENT-PRINCIPAL-ID', '12345')
        user_name = headers.get('X-MS-CLIENT-PRINCIPAL-NAME', 'r0bai')
        
        now = datetime.now()
        output_id = str(int(time.time() * 1000))
        
        # Get style name for document title if not provided
        if title is None:
            title = st.session_state.get('styleName', 'Rewritten Content')
        
        # Generate PDF and DOCX files
        pdf_filename = f"outputs/{user_id}/{output_id}.pdf"
        docx_filename = f"outputs/{user_id}/{output_id}.docx"
        
        pdf_url = None
        docx_url = None
        
        # Only attempt file generation if blob storage is configured
        if blob_service_client is not None:
            # Upload PDF - use provided bytes or generate new
            try:
                if pdf_bytes:
                    st.info(f"Uploading pre-generated PDF to blob storage...")
                    pdf_buffer = BytesIO(pdf_bytes)
                    pdf_url = upload_to_blob(pdf_buffer, pdf_filename)
                else:
                    st.info(f"Generating PDF with style: {title}...")
                    pdf_buffer = generate_pdf(output, pdf_filename, title=title)
                    if pdf_buffer:
                        st.info(f"Uploading PDF to blob storage...")
                        pdf_url = upload_to_blob(pdf_buffer, pdf_filename)
                    else:
                        st.warning("PDF generation failed")
                
                if pdf_url:
                    st.success(f"PDF uploaded successfully")
            except Exception as pdf_error:
                st.error(f"PDF upload error: {pdf_error}")
            
            # Upload DOCX - use provided bytes or generate new
            try:
                if docx_bytes:
                    st.info(f"Uploading pre-generated DOCX to blob storage...")
                    docx_buffer = BytesIO(docx_bytes)
                    docx_url = upload_to_blob(docx_buffer, docx_filename)
                else:
                    st.info(f"Generating DOCX with style: {title}...")
                    docx_buffer = generate_docx(output, docx_filename, title=title)
                    if docx_buffer:
                        st.info(f"Uploading DOCX to blob storage...")
                        docx_url = upload_to_blob(docx_buffer, docx_filename)
                    else:
                        st.warning("DOCX generation failed")
                
                if docx_url:
                    st.success(f"DOCX uploaded successfully")
            except Exception as docx_error:
                st.error(f"DOCX upload error: {docx_error}")
        else:
            st.warning("Blob storage not configured. Files will not be generated.")
        
        new_output = {
            "id": output_id,
            "updatedAt": now.isoformat(),
            "content": content_all,
            "styleId": st.session_state.styleId,
            "output": output,
            "user_id": user_id,
            "user_name": user_name,
            "pdf": pdf_url,
            "docx": docx_url
        }
        outputs_container.create_item(body=new_output)
        st.success("Output saved to database")
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred while saving output: {e}")
    except Exception as e:
        st.error(f"Unexpected error in save_output: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")


# get outputs from database
def get_outputs():
    try:
        # --- Basic guards ---
        if 'outputs_container' not in globals() or outputs_container is None:
            st.info("no data found")
            return

        # Get current user info (fall back to anon-ish id if header absent)
        headers = getattr(st, "context", None).headers if hasattr(st, "context") else {}
        user_id = (headers or {}).get('X-MS-CLIENT-PRINCIPAL-ID', '12345')
        if not user_id:
            st.warning("User not authenticated")
            return

        # Query user-scoped items
        query = "SELECT * FROM c WHERE c.user_id = @user_id ORDER BY c.updatedAt DESC"
        parameters = [{"name": "@user_id", "value": user_id}]
        items = list(outputs_container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))

        # --- Empty DB / no rows case ---
        if not items:
            st.info("no data found")
            return

        # Keep only the latest 500 items; delete the rest
        latest_items = items[:500]
        items_to_delete = items[500:]
        for item in items_to_delete:
            try:
                outputs_container.delete_item(item=item['id'], partition_key=item.get('user_id'))
            except exceptions.CosmosHttpResponseError:
                # Non-fatal: continue deleting others
                pass

        # Build DF safely even if some fields are missing
        # Use .get with defaults to avoid KeyError
        safe_rows = []
        for it in latest_items:
            safe_rows.append({
                'updatedAt': it.get('updatedAt'),
                'styleId'  : it.get('styleId'),
                'content'  : it.get('content'),
                'output'   : it.get('output'),
                'user_name': it.get('user_name'),
                'pdf'      : it.get('pdf'),
                'docx'     : it.get('docx')
            })

        df = pd.DataFrame.from_records(safe_rows)

        # If the resulting DF is empty or all-NaN columns, inform the user
        if df.empty or df.dropna(how="all").empty:
            st.info("no data found")
            return

        # Reorder columns to show most relevant first
        column_order = ['updatedAt', 'user_name', 'styleId', 'content', 'output', 'pdf', 'docx']
        df = df[[col for col in column_order if col in df.columns]]
        
        return df

    except (exceptions.CosmosResourceNotFoundError, exceptions.CosmosHttpResponseError) as e:
        # Container or query error
        st.error(f"An error occurred while fetching outputs: {e}")
    except Exception as e:
        # Catch-all so the UI doesn't explode
        st.error(f"Unexpected error: {e}")


# get styles outputs from database
def get_styles_outputs():
    try:
        # --- Basic guards ---
        if 'styles_container' not in globals() or styles_container is None:
            st.info("no data found")
            return

        # Get current user info
        headers = getattr(st, "context", None).headers if hasattr(st, "context") else {}
        user_id = (headers or {}).get('X-MS-CLIENT-PRINCIPAL-ID', '12345')
        if not user_id:
            st.warning("User not authenticated")
            return

        # Query user-scoped items
        query = """
        SELECT * 
        FROM c 
        WHERE c.user_id IN (@user_id, 'allbsp')
        ORDER BY c.updatedAt DESC
        """
        parameters = [{"name": "@user_id", "value": user_id}]
        items = list(styles_container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))

        # --- Empty DB / no rows case ---
        if not items:
            st.info("no data found")
            return

        # Build DF safely even if some fields are missing
        safe_rows = []
        for it in items:
            safe_rows.append({
                'updatedAt': it.get('updatedAt'),
                'name'     : it.get('name'),
                'style'    : it.get('style'),
                'example'  : it.get('example'),
                'user_name': it.get('user_name')
            })

        df = pd.DataFrame.from_records(safe_rows)

        # If the resulting DF is empty or all-NaN columns, inform the user
        if df.empty or df.dropna(how="all").empty:
            st.info("no data found")
            return

        # Reorder columns
        column_order = ['updatedAt', 'user_name', 'name', 'style', 'example']
        df = df[[col for col in column_order if col in df.columns]]
        
        return df

    except (exceptions.CosmosResourceNotFoundError, exceptions.CosmosHttpResponseError) as e:
        # Container or query error
        st.error(f"An error occurred while fetching styles: {e}")
    except Exception as e:
        # Catch-all so the UI doesn't explode
        st.error(f"Unexpected error: {e}")
