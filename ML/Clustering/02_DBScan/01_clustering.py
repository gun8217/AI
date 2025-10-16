import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# 데이터 불러오기
df = pd.read_csv(ROOT_DIR / 'data/customer_summary.csv')

# 날짜 제거 (군집화에 불필요)
df.drop(columns=['LastPurchaseDate'], inplace=True)

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['CustomerID']))

# DBSCAN 파라미터 설정
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# 결과 컬럼 추가
df['DBSCAN_Cluster'] = clusters

# 군집별 개수 확인
print(df['DBSCAN_Cluster'].value_counts())

# 실루엣 점수 계산 (노이즈 -1 제외하고)
mask = clusters != -1
if sum(mask) > 1:  # 군집이 2개 이상일 때만 계산 가능
    score = silhouette_score(X_scaled[mask], clusters[mask])
    print(f'DBSCAN Silhouette Score (excluding noise): {score:.3f}')
else:
    print("군집 수가 적어 실루엣 점수를 계산할 수 없습니다.")

# 군집화 데이터 저장
df.to_csv(ROOT_DIR / 'data/DBSCAN_clusters.csv', index=False)