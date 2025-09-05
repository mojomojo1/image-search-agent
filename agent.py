# agent.py

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.artifacts import GcsArtifactService
from google.adk.sessions import InMemorySessionService
from .tools import search_and_save_images
from .tools import edit_artifact_image

# --- Agent Definition ---
PRODUCT_SEARCH_PROMPT = """
You are a creative assistant who helps find and edit product images.

WORKFLOW:
1. First, use the `search_and_save_images` tool to find images based on the user's initial request.
2. The tool will return a dictionary with a list of `artifact_names` and `gcs_uris`.
3. Present the found images to the user by creating a markdown link for EACH artifact and listing its GCS URI.
4. After presenting the images, ALWAYS ask the user if they would like to edit one of the images.
5. If the user wants to make an edit, ask them which image they want to edit (e.g., "the first one", "the second one") and what changes they would like to make.
6. Use the `edit_artifact_image` tool, providing the correct `artifact_name` from the previous search and the user's `edit_prompt`.
7. When the edit tool is successful, present the new edited image to the user using a markdown link with the `new_artifact_name`.
"""

root_agent = Agent(
    name="product_image_search_agent",
    model="gemini-2.5-flash",
    description="Finds product images from a catalog, displays them, and offers to edit them.",
    instruction=PRODUCT_SEARCH_PROMPT,
    tools=[search_and_save_images,edit_artifact_image],
    output_key="search_results_dict",
)
