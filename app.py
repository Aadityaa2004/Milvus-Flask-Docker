from flask import Flask, request, jsonify
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
import os

app = Flask(__name__)

# Constants
COLLECTION_NAME = "user_profiles"
EMBEDDING_DIM = 384  
VECTOR_FIELD = "profile_embedding"
CSV_PATH = "/app/data/sample1.csv"  # This is the path inside the container
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Using model name instead of path

try:
    print("Loading the model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

def connect_to_milvus():
    """Connect to Milvus server"""
    connections.connect(
        alias="default",
        host='localhost',
        port='19530'
    )

def validate_csv_data(df):
    """Validate CSV data before processing"""
    required_columns = [
        'First Name', 'Middle Name', 'Last Name', '*Email', '*User ID',
        'Location', 'Phone Number', 'About', '**Current Work Experience (Company)',
        '**Current Work Experience (Years)', 'Past Work Experience (Company)',
        'Past Work Experience (Years)', 'Institution', '*Major',
        '*Graduation Date', 'Rating', 'Credits', 'Skills',
        '**Certificates', '**Tags'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def connect_to_milvus():
    """Connect to Milvus server with improved error handling"""
    try:
        # Check if connection already exists
        if utility.get_connection('default') is not None:
            print("Already connected to Milvus server")
            return

        print("Attempting to connect to Milvus server...")
        connections.connect(
            alias="default",
            host='localhost',
            port='19530',
            timeout=10  # 10 seconds timeout
        )
        print("Successfully connected to Milvus server")
        
        # Verify connection
        if utility.get_connection('default') is None:
            raise Exception("Connection verification failed")
            
    except Exception as e:
        error_message = str(e)
        if "Connection refused" in error_message:
            raise Exception(
                "Could not connect to Milvus server. Please ensure that:\n"
                "1. Docker is running\n"
                "2. Milvus containers are running (use 'docker ps' to check)\n"
                "3. Ports 19530 and 9091 are not being used by other applications\n"
                f"Original error: {error_message}"
            )
        elif "timeout" in error_message.lower():
            raise Exception(
                "Connection timed out. Please ensure that:\n"
                "1. Milvus server is fully started (can take a minute after container starts)\n"
                "2. Network is not blocking the connection\n"
                f"Original error: {error_message}"
            )
        else:
            raise Exception(f"Failed to connect to Milvus: {error_message}")

# Add these utility functions
def check_milvus_status():
    """Check Milvus server status"""
    try:
        # Check Docker containers
        import subprocess
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if result.returncode != 0:
            return {
                "status": "error",
                "message": "Docker is not running or not accessible"
            }
        
        # Check if Milvus container is running
        if 'milvus-standalone' not in result.stdout:
            return {
                "status": "error",
                "message": "Milvus container is not running"
            }
        
        # Try connecting to Milvus
        connect_to_milvus()
        return {
            "status": "success",
            "message": "Milvus server is running and accessible"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

def create_collection():
    """Create Milvus collection with the required schema"""
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="first_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="middle_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="last_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="email", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="phone", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="about", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="current_company", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="current_experience", dtype=DataType.INT32),
        FieldSchema(name="past_company", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="past_experience", dtype=DataType.INT32),
        FieldSchema(name="institution", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="major", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="graduation_date", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="rating", dtype=DataType.FLOAT),
        FieldSchema(name="credits", dtype=DataType.INT32),
        FieldSchema(name="skills", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="certificates", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name=VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    ]
    
    schema = CollectionSchema(fields)
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    
    # Create index for vector field
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(VECTOR_FIELD, index_params)
    return collection

def get_text_embedding(text: str) -> List[float]:
    """Convert text to embedding vector"""
    return model.encode(text).tolist()

@app.route('/')
def welcome():
    """Welcome endpoint"""
    return jsonify({
        "message": "Welcome to Guide-Milvus-Flask API",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "GET /": "Welcome message",
            "GET /test": "Test endpoint",
            "POST /load-data": "Load data from CSV",
            "POST /add-user": "Add new user",
            "DELETE /delete-user/<user_id>": "Delete user",
            "PUT /update-user/<user_id>": "Update user",
            "POST /search": "Vector search"
        }
    })

# Add a status endpoint
@app.route('/status')
def check_status():
    """Check system status"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "status": "error",
                "message": "Model not loaded"
            }), 500

        # Try connecting to Milvus
        try:
            connect_to_milvus()
            return jsonify({
                "status": "success",
                "message": "System is operational",
                "details": {
                    "milvus_connection": "connected",
                    "model_loaded": "yes"
                }
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Milvus connection failed: {str(e)}"
            }), 500

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/test')
def test():
    """Test endpoint"""
    return jsonify({
        "message": "Welcome to Guide-Milvus-Flask",
        "status": "success"
    })

@app.route('/load-data', methods=['POST'])
def load_data():
    """Load data from CSV into Milvus with improved error handling"""
    try:
        # Check if CSV file exists
        if not os.path.exists(CSV_PATH):
            return jsonify({
                "error": f"CSV file not found at path: {CSV_PATH}",
                "status": "error"
            }), 404

        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Sentence transformer model not initialized",
                "status": "error"
            }), 500

        # Try to connect to Milvus
        try:
            connect_to_milvus()
        except Exception as e:
            return jsonify({
                "error": f"Failed to connect to Milvus server: {str(e)}",
                "status": "error"
            }), 500

        # Read CSV with error handling
        try:
            df = pd.read_csv(CSV_PATH)
            print(f"Successfully read CSV with {len(df)} rows")
        except Exception as e:
            return jsonify({
                "error": f"Error reading CSV file: {str(e)}",
                "status": "error"
            }), 500

        # Validate CSV data
        try:
            validate_csv_data(df)
        except ValueError as e:
            return jsonify({
                "error": f"Data validation failed: {str(e)}",
                "status": "error"
            }), 400

        # Create collection
        try:
            collection = create_collection()
            print("Successfully created collection")
        except Exception as e:
            return jsonify({
                "error": f"Error creating collection: {str(e)}",
                "status": "error"
            }), 500

        # Clean and prepare data
        try:
            # Convert numeric columns
            df['**Current Work Experience (Years)'] = pd.to_numeric(df['**Current Work Experience (Years)'], errors='coerce').fillna(0)
            df['Past Work Experience (Years)'] = pd.to_numeric(df['Past Work Experience (Years)'], errors='coerce').fillna(0)
            df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(0.0)
            df['Credits'] = pd.to_numeric(df['Credits'], errors='coerce').fillna(0)

            # Prepare entities dictionary
            entities = {
                "user_id": df['*User ID'].astype(str).tolist(),
                "first_name": df['First Name'].astype(str).tolist(),
                "middle_name": df['Middle Name'].astype(str).tolist(),
                "last_name": df['Last Name'].astype(str).tolist(),
                "email": df['*Email'].astype(str).tolist(),
                "location": df['Location'].astype(str).tolist(),
                "phone": df['Phone Number'].astype(str).tolist(),
                "about": df['About'].astype(str).tolist(),
                "current_company": df['**Current Work Experience (Company)'].astype(str).tolist(),
                "current_experience": df['**Current Work Experience (Years)'].astype(int).tolist(),
                "past_company": df['Past Work Experience (Company)'].astype(str).tolist(),
                "past_experience": df['Past Work Experience (Years)'].astype(int).tolist(),
                "institution": df['Institution'].astype(str).tolist(),
                "major": df['*Major'].astype(str).tolist(),
                "graduation_date": df['*Graduation Date'].astype(str).tolist(),
                "rating": df['Rating'].astype(float).tolist(),
                "credits": df['Credits'].astype(int).tolist(),
                "skills": df['Skills'].astype(str).tolist(),
                "certificates": df['**Certificates'].astype(str).tolist(),
                "tags": df['**Tags'].astype(str).tolist(),
            }

            # Generate embeddings
            print("Generating embeddings...")
            profile_texts = df.apply(
                lambda x: f"{x['About']} {x['Skills']} {x['**Tags']}".strip(), 
                axis=1
            )
            entities[VECTOR_FIELD] = [
                get_text_embedding(text) if text.strip() else [0.0] * EMBEDDING_DIM 
                for text in profile_texts
            ]
            print("Embeddings generated successfully")

        except Exception as e:
            return jsonify({
                "error": f"Error preparing data: {str(e)}",
                "status": "error"
            }), 500

        # Insert data
        try:
            print("Inserting data into collection...")
            collection.insert(entities)
            collection.flush()
            print("Data inserted successfully")
            
            # Create index after insertion
            print("Creating index...")
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(VECTOR_FIELD, index_params)
            print("Index created successfully")

        except Exception as e:
            return jsonify({
                "error": f"Error inserting data into collection: {str(e)}",
                "status": "error"
            }), 500

        return jsonify({
            "message": "Data loaded successfully",
            "rows_processed": len(df),
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"Unexpected error: {str(e)}",
            "status": "error"
        }), 500

@app.route('/add-user', methods=['POST'])
def add_user():
    """Add a new user to the collection"""
    try:
        data = request.json
        connect_to_milvus()
        collection = Collection(COLLECTION_NAME)
        
        # Generate embedding for the new user
        profile_text = f"{data['about']} {data['skills']} {data['tags']}"
        embedding = get_text_embedding(profile_text)
        
        # Prepare entity
        entity = {
            "user_id": [data['user_id']],
            "first_name": [data['first_name']],
            "middle_name": [data['middle_name']],
            "last_name": [data['last_name']],
            "email": [data['email']],
            "location": [data['location']],
            "phone": [data['phone']],
            "about": [data['about']],
            "current_company": [data['current_company']],
            "current_experience": [data['current_experience']],
            "past_company": [data['past_company']],
            "past_experience": [data['past_experience']],
            "institution": [data['institution']],
            "major": [data['major']],
            "graduation_date": [data['graduation_date']],
            "rating": [data['rating']],
            "credits": [data['credits']],
            "skills": [data['skills']],
            "certificates": [data['certificates']],
            "tags": [data['tags']],
            VECTOR_FIELD: [embedding]
        }
        
        collection.insert(entity)
        collection.flush()
        
        return jsonify({"message": "User added successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete-user/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user from the collection"""
    try:
        connect_to_milvus()
        collection = Collection(COLLECTION_NAME)
        
        expr = f'user_id == "{user_id}"'
        collection.delete(expr)
        
        return jsonify({"message": "User deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update-user/<user_id>', methods=['PUT'])
def update_user(user_id):
    """Update user information"""
    try:
        connect_to_milvus()
        collection = Collection(COLLECTION_NAME)
        
        # Delete existing user
        expr = f'user_id == "{user_id}"'
        collection.delete(expr)
        
        # Add updated user information
        return add_user()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def vector_search():
    """Perform vector search based on query text"""
    try:
        data = request.json
        query_text = data['query']
        top_k = data.get('top_k', 5)
        
        connect_to_milvus()
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        # Generate query vector
        query_vector = get_text_embedding(query_text)
        
        # Search
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_vector],
            anns_field=VECTOR_FIELD,
            param=search_params,
            limit=top_k,
            output_fields=[
                "user_id", "first_name", "middle_name", "last_name", "email",
                "location", "phone", "about", "current_company", "current_experience",
                "past_company", "past_experience", "institution", "major",
                "graduation_date", "rating", "credits", "skills", "certificates", "tags"
            ]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = {
                    field: hit.entity.get(field) for field in hit.entity.keys()
                }
                result['distance'] = hit.distance
                formatted_results.append(result)
        
        return jsonify(formatted_results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        "error": "The requested resource was not found",
        "status": 404,
        "available_endpoints": {
            "GET /": "Welcome message",
            "GET /test": "Test endpoint",
            "POST /load-data": "Load data from CSV",
            "POST /add-user": "Add new user",
            "DELETE /delete-user/<user_id>": "Delete user",
            "PUT /update-user/<user_id>": "Update user",
            "POST /search": "Vector search"
        }
    }), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "status": 500
    }), 500

if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(debug=True, port=5000)

# Rest of your Flask app code remains the same...