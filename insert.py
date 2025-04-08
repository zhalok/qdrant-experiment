from data_loader import load_data_as_dict
from vector_store import get_client
from model import get_model
from download_dataset import download_dataset
from embeddings import get_embeddings
from vector_store import create_collection, prepare_qdrant_point_from_payload_descriptions, get_collection, store_in_vector_store
import os

if os.path.exists("normalized_text.csv") == False:
    download_dataset()


data = load_data_as_dict("normalized_text.csv")

vstore_client = get_client()

model = get_model(model_name="sentence-transformers/all-MiniLM-L6-v2")

collection_name = "test_collection-1"

collection = get_collection(collection_name=collection_name,client=vstore_client)

if collection == None:
    collection = create_collection(client=vstore_client,collection_name=collection_name)


payloads = [{"text":d["normalized_text"],"id":idx} for idx, d in enumerate(data)]

points = prepare_qdrant_point_from_payload_descriptions(model=model,get_embeddings_func=get_embeddings,payloads=payloads)

operation_response = store_in_vector_store(client=vstore_client,collection_name=collection_name,qdrant_points=points)

print("operation_response:",operation_response)

 


