from config import ROOT_DIR

import pandas as pd

# 데이터 불러오기
df = pd.read_csv(ROOT_DIR / 'data/OnlineRetail.csv', encoding='ISO-8859-1')

# 데이터 구조 확인
print(df.columns)
print(df.head())
print(df.info())

# 결측치 처리
print(df.isnull().sum())
df.dropna(subset=['Description', 'CustomerID'], inplace=True)

# 환불/무효 거래 제거
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# 총 구매 금액 계산
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# 날짜 처리
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 고객별 요약
summary = df.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique',  # 구매 횟수
    'TotalPrice': 'sum',     # 총 구매 금액
    'UnitPrice': 'mean',     # 평균 단가
    'StockCode': 'nunique',  # 구매한 제품 종류 수
    'InvoiceDate': 'max'     # 최근 구매일
}).reset_index()

summary.rename(columns={
    'InvoiceNo': 'PurchaseCount',
    'TotalPrice': 'TotalSpent',
    'UnitPrice': 'AvgUnitPrice',
    'StockCode': 'UniqueItems',
    'InvoiceDate': 'LastPurchaseDate'
}, inplace=True)

# 전처리 결과 저장
summary.to_csv(ROOT_DIR / 'data/custom_summary.csv', index=False)