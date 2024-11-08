import random
from pymilvus import MilvusClient, DataType
import json


client = MilvusClient(
    uri="https://in03-2a916bef002c329.serverless.gcp-us-west1.cloud.zilliz.com",
    token="1d4d4fa85e19a5477b621dddfbb6efd9e5254df15b0bdef7cbf2ae48c96e45bd4f1a682f84eaeec7c4884dceadc3fb4177f74fb8" 
)

client.create_collection(
    collection_name="quick_setup",
    dimension=5,
    metric_type="IP"
)

colors = ["green", "blue", "yellow", "red", "black", "white", "purple", "pink", "orange", "brown", "grey"]
data = []

for i in range(1000):
    current_color = random.choice(colors)
    data.append({
        "id": i,
        "vector": [ random.uniform(-1, 1) for _ in range(5) ],
        "color": current_color,
        "color_tag": f"{current_color}_{str(random.randint(1000, 9999))}"
    })

res = client.insert(
    collection_name="quick_setup",
    data=data
)

print(res)


client.create_partition(
    collection_name="quick_setup",
    partition_name="red"
)

client.create_partition(
    collection_name="quick_setup",
    partition_name="blue"
)

red_data = [ {"id": i, "vector": [ random.uniform(-1, 1) for _ in range(5) ], "color": "red", "color_tag": f"red_{str(random.randint(1000, 9999))}" } for i in range(500) ]
blue_data = [ {"id": i, "vector": [ random.uniform(-1, 1) for _ in range(5) ], "color": "blue", "color_tag": f"blue_{str(random.randint(1000, 9999))}" } for i in range(500) ]

res = client.insert(
    collection_name="quick_setup",
    data=red_data,
    partition_name="red"
)

print(res)


res = client.insert(
    collection_name="quick_setup",
    data=blue_data,
    partition_name="blue"
)

print(res)

res = client.search(
    collection_name="test_collection", # Replace with the actual name of your collection
    # Replace with your query vector
    data=[[0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]],
    limit=5, # Max. number of search results to return
    search_params={"metric_type": "IP", "params": {}} # Search parameters
)

result = json.dumps(res, indent=4)
print(result)

# Single vector search
res = client.search(
    collection_name="test_collection", # Replace with the actual name of your collection
    # Replace with your query vector
    data=[[0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]],
    limit=5, # Max. number of search results to return
    search_params={"metric_type": "IP", "params": {}} # Search parameters
)

# Convert the output to a formatted JSON string
result = json.dumps(res, indent=4)
print(result)

# Bulk-vector search
res = client.search(
    collection_name="test_collection", # Replace with the actual name of your collection
    data=[
        [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104],
        [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345]
    ], # Replace with your query vectors
    limit=2, # Max. number of search results to return
    search_params={"metric_type": "IP", "params": {}} # Search parameters
)

result = json.dumps(res, indent=4)
print(result)

# 6.2 Search within a partition
query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]

res = client.search(
    collection_name="quick_setup",
    data=[query_vector],
    limit=5,
    search_params={"metric_type": "IP", "params": {"level": 1}},
    partition_names=["red"]
)

print(res)
