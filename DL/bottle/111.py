from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["image_dataset"]
collection = db["images"]

# # "source"가 "augmented"인 문서 삭제
# result = collection.delete_many({"source": "augmented"})
# print(f"삭제된 문서 수: {result.deleted_count}")



import gridfs

fs = gridfs.GridFS(db)

# "source"가 "augmented"인 문서들 조회
docs = collection.find({"source": "augmented"})
print(docs)

# GridFS에서 파일 삭제
for doc in docs:
    file_id = doc.get("file_id")
    if file_id:
        fs.delete(file_id)