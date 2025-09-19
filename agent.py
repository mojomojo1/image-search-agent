from google.adk.agents import Agent
from vertexai.preview.reasoning_engines import AdkApp
import vertexai
import os

from .tools import (
    search_and_save_images,
    edit_artifact_image,
    create_bigquery_dataset,
    create_external_table,
    create_embedding_model,
    create_image_embeddings_table
)

# --- Agent Definition ---
PRODUCT_SEARCH_PROMPT = """
You are a creative assistant who helps find and edit product images.

**PHASE 1: Configuration**
1. First, greet the user and ask for the Google Cloud Storage (GCS) bucket path that contains the product images.
2. Once the user provides the bucket path, use the appropriate tools to configure the BigQuery resources. This includes creating a dataset, an external table, an embedding model, and a table of image embeddings.
3. This configuration is a one-time process for the current session and will not be repeated.

**PHASE 2: Image Search and Editing**
4. After the configuration is complete, you can begin to find and edit images.
5. Use the `search_and_save_images` tool to find images based on the user's request. The tool will return a dictionary with a list of `artifact_names`.
6. Present the found images to the user by creating a markdown link for EACH artifact. **IMPORTANT: Use the `artifact_name` from the tool's result directly in the markdown syntax.** For example, to display a file named 'image.png', use the syntax `![Image description](image.png)`.
7. After presenting the images, ALWAYS ask the user if they would like to edit one of the images.
8. If the user wants to make an edit, use the `edit_artifact_image` tool with the correct `artifact_name` and the user's `edit_prompt`.
9. When the edit tool is successful, present the new edited image to the user using a markdown link with the `new_artifact_name`.
"""

root_agent = Agent(
    name="product_image_search_agent",
    model="gemini-2.5-flash",
    description="Finds product images from a catalog, displays them, and offers to edit them.",
    instruction=PRODUCT_SEARCH_PROMPT,
    tools=[
        # Core functionality tools
        search_and_save_images,
        edit_artifact_image,
        # One-time configuration tools
        create_bigquery_dataset,
        create_external_table,
        create_embedding_model,
        create_image_embeddings_table
    ],
    output_key="search_results_dict",
)
