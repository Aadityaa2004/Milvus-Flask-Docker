from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection
import time

client = MilvusClient(
    uri='http://localhost:19530', # replace with your own Milvus server address
    token='root:Milvus' # replace with your own Milvus server token
)

client.create_user(
    user_name='user_1',
    password='P@ssw0rd'
)

schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, description="primary id")
age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128, description="vector")

# Enable partition key on a field if you need to implement multi-tenancy based on the partition-key field
position_field = FieldSchema(name="position", dtype=DataType.VARCHAR, max_length=256, is_partition_key=True)

# Set enable_dynamic_field to True if you need to use dynamic fields. 
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

# 2.2. Add fields to schema
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=5)

# 3. Create collection
client.create_collection(
    collection_name="customized_setup", 
    schema=schema, 
)

index_params = MilvusClient.prepare_index_params()

# 4.2. Add an index on the vector field.
index_params.add_index(
    field_name="vector",
    metric_type="COSINE",
    index_type="IVF_FLAT",
    index_name="vector_index",
    params={ "nlist": 128 }
)

# 4.3. Create an index file
client.create_index(
    collection_name="customized_setup",
    index_params=index_params,
    sync=False # Whether to wait for index creation to complete before returning. Defaults to True.
)


# index_params.add_index(
#     field_name="my_id",
#     index_type="STL_SORT"
# )

# index_params.add_index(
#     field_name="my_vector", 
#     index_type="IVF_FLAT",
#     metric_type="IP",
#     params={ "nlist": 128 }
# )

# client.create_collection(
#     collection_name="customized_setup_1",
#     schema=schema,
#     index_params=index_params
# )

# time.sleep(5)

# res = client.get_load_state(
#     collection_name="customized_setup_1"
# )

# print(res)

# client.create_collection(
#     collection_name="customized_setup_2",
#     schema=schema,
# )

# res = client.get_load_state(
#     collection_name="customized_setup_2"
# )

# print(res)

# res = client.describe_collection(
#     collection_name="customized_setup_2"
# )
# client.release_collection(
#     collection_name="customized_setup_2"
# )

# res = client.get_load_state(
#     collection_name="customized_setup_2"
# )



index_params = client.create_index_params() # Prepare an empty IndexParams object, without having to specify any index parameters

index_params.add_index(
    field_name="scalar_1", # Name of the scalar field to be indexed
    index_type="", # Type of index to be created. For auto indexing, leave it empty or omit this parameter.
    index_name="default_index" # Name of the index to be created
)

client.create_index(
  collection_name="test_scalar_index", # Specify the collection name
  index_params=index_params
)

index_params = client.create_index_params() #  Prepare an IndexParams object

index_params.add_index(
    field_name="scalar_2", # Name of the scalar field to be indexed
    index_type="INVERTED", # Type of index to be created
    index_name="inverted_index" # Name of the index to be created
)

client.create_index(
  collection_name="test_scalar_index", # Specify the collection name
  index_params=index_params
)