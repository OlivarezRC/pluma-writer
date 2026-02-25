import os
import json
import time
import requests
import pandas as pd
import streamlit as st

from datetime import datetime
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, exceptions, PartitionKey
from dotenv import load_dotenv
import hashlib
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

# log tracing
def trace(col2, label, message):
    with col2:
        with st.expander(f"{label}:"):
            st.write(message)
            # print(f"{label}: {message}")


# get request api
def get_request(url):
    response = requests.get(url, timeout=10)
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


# save style to database
def save_style(style, combined_text):
    try:
        # Get current user info
        headers = st.context.headers
        user_id = headers.get('X-MS-CLIENT-PRINCIPAL-ID', '12345')
        user_name = headers.get('X-MS-CLIENT-PRINCIPAL-NAME', 'r0bai')
        
        now = datetime.now()
        st.session_state.styleId = str(int(time.time() * 1000))
        new_style = {
            "id": st.session_state.styleId,
            "updatedAt": now.isoformat(),
            "name": st.session_state.styleName,
            "style": style,
            "example": combined_text,
            "user_id": user_id,
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
def save_output(output, content_all):
    try:
        # Get current user info
        headers = st.context.headers
        user_id = headers.get('X-MS-CLIENT-PRINCIPAL-ID', '12345')
        user_name = headers.get('X-MS-CLIENT-PRINCIPAL-NAME', 'r0bai')
        
        now = datetime.now()
        new_output = {
            "id": str(int(time.time() * 1000)),
            "updatedAt": now.isoformat(),
            "content": content_all,
            "styleId": st.session_state.styleId,
            "output": output,
            "user_id": user_id,
            "user_name": user_name
        }
        outputs_container.create_item(body=new_output)
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred while saving output: {e}")


# get outputs from database
def get_outputs():
    try:
        # Get current user info
        headers = st.context.headers
        user_id = headers.get('X-MS-CLIENT-PRINCIPAL-ID', '12345')
        
        if not user_id:
            st.warning("User not authenticated")
            return
            
        # Query to get all items for current user ordered by updatedAt
        query = "SELECT * FROM c WHERE c.user_id = @user_id ORDER BY c.updatedAt DESC"
        parameters = [{"name": "@user_id", "value": user_id}]
        items = list(outputs_container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        # Keep only the latest 50 items
        items_to_delete = items[50:]
        latest_items = items[:50]
        
        # Delete older items
        for item in items_to_delete:
            outputs_container.delete_item(
                item=item['id']
            )
        
        # Convert to DataFrame and select only specific columns
        df = pd.DataFrame(latest_items)
        df = df[['updatedAt', 'styleId', 'content', 'output']]
        # Display the DataFrame in Streamlit
        st.dataframe(df)
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred while fetching outputs: {e}")
