from pymongo import MongoClient

# MongoDB 연결
client = MongoClient("mongodb://localhost:27017/")
db = client["image_db"]
collection = db["images"]

# collection.delete_many({"_id": {"$ne": "_0"}})

print(collection.count_documents({}))