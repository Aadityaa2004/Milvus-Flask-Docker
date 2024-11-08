from pymilvus import connections, db

conn = connections.connect(host="127.0.0.1", port=19530)

database = db.create_database("my_database")

db.using_database("my_database")

conn = connections.connect(
    host="127.0.0.1",
    port="19530",
    db_name="my_database"
)

_URI = "http://localhost:19530"
_TOKEN = "root:Milvus"
_DB_NAME = "default"


def connect_to_milvus(db_name="default"):
    print(f"connect to milvus\n")
    connections.connect(
        uri=_URI,
        token=_TOKEN,
        db_name=db_name
    )