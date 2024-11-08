from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import random

connections.connect(
    host="10.102.7.3", # Replace with your Milvus server IP
    port="19530"
)

fields = [
    FieldSchema(name="film_id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="filmVector", dtype=DataType.FLOAT_VECTOR, dim=5), # Vector field for film vectors
    FieldSchema(name="posterVector", dtype=DataType.FLOAT_VECTOR, dim=5)] # Vector field for poster vectors

schema = CollectionSchema(fields=fields,enable_dynamic_field=False)

collection = Collection(name="test_collection", schema=schema)

index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}

collection.create_index("filmVector", index_params)
collection.create_index("posterVector", index_params)

entities = []

for _ in range(1000):
    # generate random values for each field in the schema
    film_id = random.randint(1, 1000)
    film_vector = [ random.random() for _ in range(5) ]
    poster_vector = [ random.random() for _ in range(5) ]

    # create a dictionary for each entity
    entity = {
        "film_id": film_id,
        "filmVector": film_vector,
        "posterVector": poster_vector
    }

    # add the entity to the list
    entities.append(entity)
    
collection.insert(entities)

from pymilvus import AnnSearchRequest

query_filmVector = [[0.8896863042430693, 0.370613100114602, 0.23779315077113428, 0.38227915951132996, 0.5997064603128835]]

search_param_1 = {
    "data": query_filmVector, # Query vector
    "anns_field": "filmVector", # Vector field name
    "param": {
        "metric_type": "L2", # This parameter value must be identical to the one used in the collection schema
        "params": {"nprobe": 10}
    },
    "limit": 2 # Number of search results to return in this AnnSearchRequest
}
request_1 = AnnSearchRequest(**search_param_1)

query_posterVector = [[0.02550758562349764, 0.006085637357292062, 0.5325251250159071, 0.7676432650114147, 0.5521074424751443]]
search_param_2 = {
    "data": query_posterVector, # Query vector
    "anns_field": "posterVector", # Vector field name
    "param": {
        "metric_type": "L2", # This parameter value must be identical to the one used in the collection schema
        "params": {"nprobe": 10}
    },
    "limit": 2 # Number of search results to return in this AnnSearchRequest
}
request_2 = AnnSearchRequest(**search_param_2)

reqs = [request_1, request_2]

from pymilvus import WeightedRanker
rerank = WeightedRanker(0.8, 0.2)  

from pymilvus import RRFRanker

rerank = RRFRanker()

collection.load()

res = collection.hybrid_search(
    reqs, # List of AnnSearchRequests created in step 1
    rerank, # Reranking strategy specified in step 2
    limit=2 # Number of final search results to return
)

print(res)

