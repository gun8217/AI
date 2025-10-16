from pymongo import MongoClient

# MongoDB 연결
client = MongoClient("mongodb://localhost:27017/")
db = client["image_db"]
collection = db["images"]

# origin 기준 문서 가져오기
origin_docs = list(collection.find({"source": "origin"}))

# 보정된 수 카운트
fixed_count = 0
wrong_predictions = []

for origin in origin_docs:
    filename = origin["filename"]
    origin_class = origin["class"]

    # YOLO 문서 찾기
    yolo = collection.find_one({
        "filename": filename,
        "source": "yolo"
    })

    if yolo and yolo["class"] != origin_class:
        # 틀린 예측 기록
        wrong_predictions.append({
            "filename": filename,
            "yolo_class": yolo["class"],
            "origin_class": origin_class,
            "confidence": yolo.get("confidence", None)
        })

        # 클래스 보정
        result = collection.update_one(
            {"_id": yolo["_id"]},
            {
                "$set": {
                    "class": origin_class,
                    "fixed": True  # 보정 표시
                }
            }
        )
        if result.modified_count > 0:
            fixed_count += 1
            print(f"보정됨: {filename} | {yolo['class']} → {origin_class}")

# 결과 요약
print(f"\n 총 {fixed_count}개 YOLO 클래스 보정 완료!\n")

if wrong_predictions:
    print("틀린 예측 목록:")
    for wp in wrong_predictions:
        print(f"- {wp['filename']} | yolo: {wp['yolo_class']} → origin: {wp['origin_class']} | conf: {wp['confidence']}")