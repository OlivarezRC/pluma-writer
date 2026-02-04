import streamlit as st
import app.pages as pages
import app.utils as utils
from azure.cosmos import exceptions


def _pk_paths(container):
    """Read PK paths from container settings, e.g. ['/tenantId'] or ['/tenantId','/styleKey']"""
    props = container.read()
    return props.get("partitionKey", {}).get("paths", [])

def _path_to_sql(expr_path: str) -> str:
    """
    Convert a PK path like '/tenantId' or '/a/b' to Cosmos SQL field expr:
    -> c["tenantId"] or c["a"]["b"]
    """
    parts = [p for p in expr_path.split("/") if p]  # drop empty segments
    sql = "c"
    for p in parts:
        sql += f'["{p}"]'
    return sql

def _fetch_pk_values_for_id(container, item_id):
    """Query the document once to get PK value(s) in the right order."""
    pk_paths = _pk_paths(container)  # e.g. ['/styleKey'] or ['/tenantId','/styleKey']
    if not pk_paths:
        raise RuntimeError("Container has no partition key paths configured.")

    # Build SELECT with aliases pk0, pk1, ...
    pk_selects = [f"{_path_to_sql(p)} AS pk{i}" for i, p in enumerate(pk_paths)]
    select_clause = ", ".join(["c.id"] + pk_selects)

    query = f"SELECT TOP 1 {select_clause} FROM c WHERE c.id = @id"
    items = list(container.query_items(
        query=query,
        parameters=[{"name": "@id", "value": item_id}],
        enable_cross_partition_query=True
    ))
    if not items:
        raise exceptions.CosmosResourceNotFoundError("Item not found.")

    row = items[0]
    pk_values = [row[f"pk{i}"] for i in range(len(pk_paths))]
    # If single-part PK, pass a scalar; if hierarchical, pass list
    return pk_values[0] if len(pk_values) == 1 else pk_values


# App title
pages.show_home()
pages.show_sidebar()

st.header("⚙️Settings")
# Display Azure authentication information
st.write(":blue[👤 **User Information**]")

with st.expander("Authentication Details"):
    # Get request headers using Streamlit's context
    headers = st.context.headers
    
    if headers:
        # Display user information
        user_name = headers.get('X-MS-CLIENT-PRINCIPAL-NAME', 'Not available')
        user_id = headers.get('X-MS-CLIENT-PRINCIPAL-ID', 'Not available')
        st.write(f"User Name: {user_name}")
        st.write(f"User ID: {user_id}")
        
        # Display all headers
        # st.write("**All Request Headers:**")
        # st.write(headers)
    else:
        st.warning("No authentication headers found. Make sure you're running this app in Azure App Service with authentication enabled.")

# Get all styles
styles = utils.get_styles()

if styles:
    style_names = [style["name"] for style in styles if style.get("name")]
    if style_names:
        selected_style = st.selectbox(":blue[**Select a style to delete:**]", style_names)
        selected_style_data = next((s for s in styles if s["name"] == selected_style), None)

        if selected_style_data:
            if st.button(f":blue[**Delete '{selected_style}'**]"):
                try:
                    item_id = selected_style_data["id"]

                    # Step 1: fetch actual PK value(s) from the stored document
                    pk_value = _fetch_pk_values_for_id(utils.styles_container, item_id)

                    # Optional sanity check: ensure ≤ 2048 bytes if scalar
                    if isinstance(pk_value, str) and len(pk_value.encode("utf-8")) > 2048:
                        raise ValueError("Stored partition key exceeds 2048 bytes.")

                    # Step 2: point delete with id + correct PK
                    utils.styles_container.delete_item(item=item_id, partition_key=pk_value)

                    st.success(f"Style '{selected_style}' has been deleted successfully!")
                except exceptions.CosmosResourceNotFoundError:
                    st.warning("Item not found or already deleted.")
                except Exception as e:
                    st.error(f"An error occurred while deleting the style: {e}")
    else:
        st.info("No named styles found in the database.")
else:
    st.info("No styles found in the database.")