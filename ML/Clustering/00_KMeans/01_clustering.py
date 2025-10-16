import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv(ROOT_DIR / 'data/customer_summary.csv')

# 날짜 제거 (군집화에 불필요)
df.drop(columns=['LastPurchaseDate'], inplace=True)

# 스케일링
scaler = StandardScaler()
scaled = scaler.fit_transform(df.drop(columns=['CustomerID']))

# 최적 군집 수 찾기 (엘보우 방법)
inertia = []
K_range = range(2, 11)
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled)
    inertia.append(model.inertia_)

# 엘보우 시각화
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# 군집 수 선택 후 모델 학습
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled)

# 군집별 요약
print(df.groupby('Cluster').mean())

# 군집화 데이터 저장
df.to_csv(ROOT_DIR / 'data/KMeans_clusters.csv', index=False)
