from embeddings import get_embeddings
from vector_store import get_client
from model import get_model
from data_loader import load_data_as_dict

data = load_data_as_dict("normalized_text.csv")

query = input("ask your query: ")
print("\n")

model = get_model(model_name="sentence-transformers/all-MiniLM-L6-v2")

query_embeddings = get_embeddings(model=model,text=query)

client = get_client()
collection_name = "test_collection-1"

search_result = client.query_points(
    collection_name=collection_name,
    query=query_embeddings,
    with_payload=True,
    with_vectors=False,
    limit=3
).points

for res in search_result:
    product_description = res.payload["text"]
    product_id = res.payload["id"]
    similarity_score = res.score
    print( f"product id: {product_id}\nproduct description: {product_description}\nscore: {similarity_score}\n\n" )