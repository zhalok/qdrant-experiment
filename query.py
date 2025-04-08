from embeddings import get_embeddings
from vector_store import get_client
from model import get_model
from data_loader import load_data_as_dict

data = load_data_as_dict("normalized_text.csv")

query = input("ask your query: ")

model = get_model(model_name="sentence-transformers/all-MiniLM-L6-v2")

query_embeddings = get_embeddings(model=model,text=query)

client = get_client()
collection_name = "test_collection-1"

search_result = client.query_points(
    collection_name=collection_name,
    query=query_embeddings,
    with_payload=False,
    limit=3
).points

point = search_result[0]

point_id = point.id

data_point_text = data[point_id]["normalized_text"]

print(data_point_text)