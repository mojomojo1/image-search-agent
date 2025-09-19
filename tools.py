import io
import os
from google.cloud import bigquery, storage
from google.cloud.exceptions import NotFound
from PIL import Image
from google.genai import types
import google.genai as genai
from io import BytesIO
from google.adk.tools import FunctionTool, ToolContext 

# Assume project_id is an environment variable or set globally
project_id = os.environ.get("PROJECT_ID", "your_project")
ae_deployed = os.environ.get("AE_DEPLOYED", "FALSE")

# --- Helper functions for artifact loading ---
async def _get_image_parts(tool_context, artifact_name: str, available_files=None) -> list:
    """Helper function to get image parts based on deployment environment."""
    new_image_parts = []
    
    if ae_deployed.upper() == "TRUE":
        original_artifact = await tool_context.load_artifact(artifact_name)
        new_image_parts.append(types.Part.from_bytes(
            data=original_artifact["inlineData"]["data"], mime_type=original_artifact["inlineData"]["mimeType"]
        ))
        
        if available_files:
            for filename in available_files:
                f = await tool_context.load_artifact(filename=filename)
                new_image_parts.append(types.Part.from_bytes(
                    data=f["inlineData"]["data"], mime_type=f["inlineData"]["mimeType"]
                ))
    else:
        original_artifact = await tool_context.load_artifact(artifact_name)
        if original_artifact.inline_data:
            if original_artifact.inline_data.data:
                new_image_parts.append(types.Part.from_bytes(
                    data=original_artifact.inline_data.data, mime_type=original_artifact.inline_data.mime_type
                ))
        
        if available_files:
            for filename in available_files:
                f = await tool_context.load_artifact(filename=filename)
                if f.inline_data and f.inline_data.data:
                    new_image_parts.append(types.Part.from_bytes(
                        data=f.inline_data.data, mime_type=f.inline_data.mime_type
                    ))
    
    return new_image_parts

# --- New Configuration Tools ---

async def create_bigquery_dataset(tool_context, bucket_uri: str) -> dict:
    """Creates a BigQuery dataset if it doesn't already exist and saves the name to the session."""
    
    # First, check if the dataset ID is already stored in the current session's state.
    # This avoids unnecessary API calls if we've already done this in the same conversation.
    if "configured_dataset_id" in tool_context.state:
        dataset_id = tool_context.state.get("configured_dataset_id")
        return {"status": "success", "message": f"Dataset {dataset_id} is already configured for this session."}

    # If not in the session state, proceed to create or verify it in BigQuery.
    try:
        client = bigquery.Client()
        bucket_name = bucket_uri.strip().replace("gs://", "").replace("/", "_")
        dataset_id = f"image_search_{bucket_name}"
        
        # Create a fully qualified dataset object.
        dataset = bigquery.Dataset(f"{client.project}.{dataset_id}")
        dataset.location = "us"
        
        # Use exists_ok=True. This is the key change.
        # It will create the dataset OR do nothing if it already exists, without causing an error.
        client.create_dataset(dataset, exists_ok=True)
        
        # Now that we've confirmed the dataset exists, save its ID to the session state for future calls.
        #await tool_context.set("configured_dataset_id", dataset_id)
        tool_context.state["configured_dataset_id"] = dataset_id
        
        return {"status": "success", "message": f"Dataset {dataset_id} is ready."}
        
    except Exception as e:
        # This will catch other problems like permission errors.
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}

async def create_external_table(tool_context, bucket_uri: str) -> dict:
    """Creates an external BigQuery table from a GCS bucket."""
    client = bigquery.Client()
    #dataset_id = tool_context.state.get("configured_dataset_id")
    dataset_id = tool_context.state.get("configured_dataset_id")
    
    if not dataset_id:
        return {"status": "error", "message": "Dataset not configured. Please run create_bigquery_dataset first."}
        
    bucket_name_for_table = bucket_uri.strip().replace("gs://", "").replace("/", "_")
    table_id = f"external_images_{bucket_name_for_table}"
    
    # --- Build the SQL Query ---
        # This replaces all the Python object configuration
    sql_query = f"""
            CREATE OR REPLACE EXTERNAL TABLE `{project_id}.{dataset_id}.{table_id}`
            WITH CONNECTION DEFAULT
            OPTIONS (
              uris = ['{bucket_uri}/*'],
              object_metadata = 'SIMPLE'
            );
        """

        # --- Execute the Query ---
    print(f"Executing SQL to create BigLake table: {table_id}")
    query_job = client.query(sql_query)
    query_job.result()  # Wait for the job to complete.

    # Save the name of the created table to the session state
    #await tool_context.set("configured_external_table", table_id)
    tool_context.state["configured_external_table"] = table_id
    
    return {"status": "success", "message": f"BigLake table {table_id} is ready."}

async def create_embedding_model(tool_context, bucket_uri: str) -> dict:
    """Creates a remote multimodal embedding model."""
    client = bigquery.Client()
    dataset_id = tool_context.state.get("configured_dataset_id")
    if not dataset_id:
        return {"status": "error", "message": "Dataset not configured. Please run create_bigquery_dataset first."}
        
    bucket_name_for_model = bucket_uri.strip().replace("gs://", "").replace("/", "_")
    model_id = f"multimodal_embedding_model_{bucket_name_for_model}"
    
    query = f"""
    CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.{model_id}`
    REMOTE WITH CONNECTION DEFAULT
    OPTIONS (ENDPOINT = 'multimodalembedding@001');
    """
    
    try:
        client.query_and_wait(query)
        #await tool_context.set("configured_model_id", model_id)
        tool_context.state["configured_model_id"] = model_id
        return {"status": "success", "message": f"Embedding model {model_id} created."}
    except Exception as e:
        return {"status": "error", "message": f"Error creating model: {e}"}
        
async def create_image_embeddings_table(tool_context) -> dict:
    """Generates the image embeddings and stores them in a new table."""
    client = bigquery.Client()
    dataset_id = tool_context.state.get("configured_dataset_id")
    external_table = tool_context.state.get("configured_external_table")
    model_id = tool_context.state.get("configured_model_id")
    
    if not all([dataset_id, external_table, model_id]):
        return {"status": "error", "message": "Required parameters for embeddings table creation are missing."}
    
    embeddings_table_id = f"met_image_embeddings" # fixed name within the dataset
    
    query = f"""
    CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{embeddings_table_id}` AS
    SELECT *
    FROM ML.GENERATE_EMBEDDING(
        MODEL `{project_id}.{dataset_id}.{model_id}`,
        (SELECT * FROM `{project_id}.{dataset_id}.{external_table}` WHERE (content_type = 'image/jpeg' OR content_type = 'image/png') LIMIT 1000)
    );
    """
    
    try:
        client.query_and_wait(query)
        #await tool_context.set("configured_embeddings_table", embeddings_table_id)
        tool_context.state["configured_embeddings_table"] = embeddings_table_id
        return {"status": "success", "message": "Image embeddings table created."}
    except Exception as e:
        return {"status": "error", "message": f"Error creating embeddings table: {e}"}

# --- Modified Search and Edit Tools ---

async def search_and_save_images(tool_context, user_text_question: str) -> dict:
    """
    Performs a vector search using the dynamically configured BigQuery tables.
    """
    client = bigquery.Client()

    # Check for configured state from the session
    dataset_id = tool_context.state.get("configured_dataset_id")
    embeddings_table = tool_context.state.get("configured_embeddings_table")
    model_id = tool_context.state.get("configured_model_id")

    if not all([dataset_id, embeddings_table, model_id]):
        return {"status": "error", "message": "Configuration not complete. Please provide a GCS bucket path to begin."}

    try:
        # Generate embedding for the search query
        sql_generate_embedding = f"CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.search_embedding` AS SELECT * FROM ML.GENERATE_EMBEDDING(MODEL `{project_id}.{dataset_id}.{model_id}`, (SELECT @user_question AS content));"
        job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("user_question", "STRING", user_text_question)])
        client.query_and_wait(sql_generate_embedding, job_config=job_config)

        # Perform the vector search
        sql_vector_search = f"CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.vector_search_results` AS SELECT base.uri AS gcs_uri, distance FROM VECTOR_SEARCH(TABLE `{project_id}.{dataset_id}.{embeddings_table}`, 'ml_generate_embedding_result', TABLE `{project_id}.{dataset_id}.search_embedding`, 'ml_generate_embedding_result', top_k => 3);"
        client.query_and_wait(sql_vector_search)

        # Fetch results and save as artifacts
        sql_fetch_results = f"SELECT gcs_uri, distance FROM `{project_id}.{dataset_id}.vector_search_results` ORDER BY distance;"
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
                blob = storage.Blob.from_string(gcs_uri, client=storage_client)
                image_bytes = blob.download_as_bytes()
                artifact_name = f"search_result_{i+1}.png"
                report_artifact = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                await tool_context.save_artifact(filename=artifact_name, artifact=report_artifact)
                saved_artifact_names.append(artifact_name)
            except Exception as e:
                print(f"Warning: Failed to process artifact for {gcs_uri}. Error: {e}")
                continue
        
        return {
            "status": "success",
            "artifact_names": saved_artifact_names,
            "gcs_uris": gcs_uris
        }

    except Exception as e:
        return {"status": "error", "message": f"An error occurred during search: {e}"}

async def edit_artifact_image(tool_context, artifact_name: str, edit_prompt: str) -> dict:
    """
    Loads an existing image artifact, edits it using a text prompt with Gemini,
    and saves the result as a new artifact.
    """
    try:
        image_parts = await _get_image_parts(tool_context, artifact_name)
        
        if not image_parts:
            return {"status": "error", "message": "Failed to load the original image artifact."}

        original_image_part = image_parts[0]
        original_pil_image = Image.open(BytesIO(original_image_part.inline_data.data))

        client = genai.Client(vertexai=True, project=project_id, location="global")
        contents = [edit_prompt, original_pil_image]
        
        generate_content_config = types.GenerateContentConfig(
            temperature = 1,
            top_p = 0.95,
            max_output_tokens = 32768,
            response_modalities = ["IMAGE", "TEXT"],
            safety_settings = [
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_IMAGE_HATE", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_IMAGE_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE")
            ],
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=contents,
            config=generate_content_config
        )
        
        edited_image_data = None
        counter = 0
        if response.candidates:
            for candidate in response.candidates:
                if candidate is not None and candidate.content is not None:
                    for part in candidate.content.parts:
                        if part is not None and part.inline_data is not None and part.inline_data.mime_type == "image/png":
                            counter = counter + 1
                            artifact_name = f"edited_image_{counter}.png"
                            report_artifact = part
                            await tool_context.save_artifact(artifact_name, report_artifact)
                            return {
                                "status": "success",
                                "message": f"Image generated. ADK artifact: {artifact_name}.",
                                "artifact_name": artifact_name,
                            }
        
        return {"status": "error", "message": "No edited image was generated."}
    except Exception as e:
        return {"status": "error", "message": f"Error editing image: {e}"}
