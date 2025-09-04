import io
import os
from google.cloud import bigquery, storage
from PIL import Image
from google.genai import types
import google.genai as genai
from google.genai import types
from io import BytesIO

async def search_and_save_images(tool_context, user_text_question: str) -> dict:
    """
    Performs a vector search, saves artifacts, and returns a dictionary
    containing artifact names and their original GCS URIs.
    """
    client = bigquery.Client()
    # BigQuery queries remain the same
    sql_generate_embedding = "CREATE OR REPLACE TABLE `your_project.your_dataset.search_embedding` AS SELECT * FROM ML.GENERATE_EMBEDDING(MODEL `your_project.your_dataset.multimodal_embedding_model`, (SELECT @user_question AS content));"
    job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("user_question", "STRING", user_text_question)])
    client.query_and_wait(sql_generate_embedding, job_config=job_config)

    sql_vector_search = "CREATE OR REPLACE TABLE `your_project.your_dataset.vector_search_results` AS SELECT base.uri AS gcs_uri, distance FROM VECTOR_SEARCH(TABLE `your_project.your_dataset.image_embeddings`, 'ml_generate_embedding_result', TABLE `your_project.your_dataset.search_embedding`, 'ml_generate_embedding_result', top_k => 3);"
    client.query_and_wait(sql_vector_search)

    sql_fetch_results = "SELECT gcs_uri, distance FROM `your_project.your_dataset.vector_search_results` ORDER BY distance;"
    results_iterator = client.query(sql_fetch_results).result()
    search_results = [(row.gcs_uri, row.distance) for row in results_iterator]
    
    if not search_results:
        return {"status": "no_results_found", "artifact_names": [], "gcs_uris": []}

    storage_client = storage.Client()
    saved_artifact_names = []
    gcs_uris = []
    for i, (gcs_uri, distance) in enumerate(search_results):
        gcs_uris.append(gcs_uri)
        try:
            # Image downloading and normalization logic
            blob = storage.Blob.from_string(gcs_uri, client=storage_client)
            blob.reload()
            print(f"Bucket: {blob.bucket.name}")
            print(f"Full Path (Name): {blob.name}")
            image_bytes = blob.download_as_bytes()
            
            # Create a short name
            artifact_name = f"search_result_{i+1}.png"
            
            report_artifact = types.Part.from_bytes(
                    data=image_bytes, mime_type="image/png"
            )

            try:
                version = await tool_context.save_artifact(filename=artifact_name, artifact=report_artifact)
                print(f"Successfully saved Python artifact '{artifact_name}' as version {version}.")
                # The event generated after this callback will contain:
                # event.actions.artifact_delta == {"generated_report.pdf": version}
            except ValueError as e:
                print(f"Error saving Python artifact: {e}. Is ArtifactService configured in Runner?")
            except Exception as e:
                # Handle potential storage errors (e.g., GCS permissions)
                print(f"An unexpected error occurred during Python artifact save: {e}")

            saved_artifact_names.append(artifact_name)
            
        except Exception as e:
            print(f"Warning: Failed to process artifact for {gcs_uri}. Error: {e}")
            continue
    
    # Return the complete dictionary
    return {
        "status": "success",
        "artifact_names": saved_artifact_names,
        "gcs_uris": gcs_uris
    }


async def edit_artifact_image(tool_context, artifact_name: str, edit_prompt: str) -> dict:
    """
    Loads an existing image artifact, edits it using a text prompt with Gemini,
    and saves the result as a new artifact.

    Args:
        tool_context: The context object provided by the ADK framework.
        artifact_name: The name of the image artifact to edit.
        edit_prompt: The user's instructions for how to edit the image.

    Returns:
        A dictionary with the status and the name of the new edited artifact.
    """
    try:
        # 1. Load the original image artifact from the session
        print(f"Loading artifact: {artifact_name}")
        original_artifact = await tool_context.load_artifact(artifact_name)
        
        # Open the image using Pillow from the loaded artifact data
        # This converts the binary data from the artifact into a PIL Image object
        original_pil_image = Image.open(BytesIO(original_artifact.inline_data.data))

        # 2. Call Gemini to generate the edited image
        print(f"Editing image with prompt: '{edit_prompt}'")
        
        # Initialize the client for the new google.genai SDK
        client = genai.Client(vertexai=True, project="your_project", location="global")

        # Construct the contents list for the API call
        # The new SDK takes the prompt string and the PIL Image object directly
        contents = [
            edit_prompt,
            original_pil_image
        ]

        # Define generation configuration (optional, but good for explicit control)
        # Setting safety settings to 'OFF' can be risky and may be disallowed by platform policies.

        generate_content_config = types.GenerateContentConfig(
            temperature = 1,
            top_p = 0.95,
            max_output_tokens = 32768,
            response_modalities = ["IMAGE", "TEXT"],
            safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
                ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_IMAGE_HATE",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_IMAGE_HARASSMENT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT",
            threshold="OFF"
            )],
        )
        
        # Make the asynchronous call to the Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview", # Or other suitable image model
            contents=contents,
            config=generate_content_config
        )
        
        # 3. Process the response to find and save the edited image
        edited_image_data = None
        counter = 0
        
        if response.candidates:
            # Iterate through all parts of the first candidate's content
            for candidate in response.candidates:
                if candidate is not None and candidate.content is not None:
                    for part in candidate.content.parts:
                        if part is not None and part.inline_data is not None and part.inline_data.mime_type == "image/png":
                            counter = counter + 1
                            artifact_name = f"edited_image_{counter}.png"
                            report_artifact = part
                            await tool_context.save_artifact(artifact_name, report_artifact)
                            print(f"Image also saved as ADK artifact: {artifact_name}")
                            return {
                                "status": "success",
                                "message": f"Image generated .  ADK artifact: {artifact_name}.",
                                "artifact_name": artifact_name,
                            }
        
        return {"status": "success", "new_artifact_name": artifact_name}

    except Exception as e:
        print(f"Error editing image: {e}")
        return {"status": "error", "message": str(e)}
