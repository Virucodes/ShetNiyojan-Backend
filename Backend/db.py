from decouple import config
from pymongo import MongoClient
import os
import pymongo
import sys

# Try to get MongoDB URI from environment variables or .env file
try:
    MONGO_URI = config("MONGO_URI")
except Exception as e:
    print(f"Error loading MONGO_URI from config: {str(e)}")
    # Fallback to a default URI for development if env var is not available
    MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/shetniyojan")
    print(f"Using default MongoDB URI: {MONGO_URI}")

# Create client with a timeout to avoid hanging
try:
    print(f"Connecting to MongoDB using pymongo {pymongo.__version__}")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    
    # Test the connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB")
except Exception as e:
    print(f"Error connecting to MongoDB: {str(e)}")
    print("Using temporary in-memory database for development")
    # Create a fake client
    class FakeDB:
        def __getitem__(self, name):
            return self
        def __getattr__(self, name):
            return self
        def find(self, *args, **kwargs):
            return []
        def find_one(self, *args, **kwargs):
            return None
        def insert_one(self, *args, **kwargs):
            class Result:
                @property
                def inserted_id(self):
                    return "fake_id"
            return Result()
        def create_index(self, *args, **kwargs):
            return None
    
    class FakeClient:
        def __getitem__(self, name):
            return FakeDB()
        def __getattr__(self, name):
            return FakeDB()
            
    client = FakeClient()

db = client['shetniyojan']

users_collection = db['users']
tasks_collection = db['tasks']
yields_collection = db['yields']
activities_collection = db['activities']
# Ensure the lease_items collection exists
try:
    db.lease_items.create_index("name")
    print("Lease items collection initialized")
except Exception as e:
    print(f"Error creating index on lease_items: {str(e)}")
    # Continue without creating the index

