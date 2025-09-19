Agent assumes you uploaded the images of the products in the bucket and will ask you for the bucket name. Also, that a default connection is set for BigQuery and Vertex AI: https://cloud.google.com/bigquery/docs/create-cloud-resource-connection#create-cloud-resource-connection

After that, if not existing, an embeddings model will be created, as described here: https://cloud.google.com/bigquery/docs/generate-multimodal-embeddings

It only happens once, after that user can endlessly search and modify assets.
