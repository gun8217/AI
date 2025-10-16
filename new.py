from datetime import datetime
import pandas as pd

# 대출 조건
loan_amount = 160_000_000
annual_rate = 0.0255
monthly_rate = annual_rate / 12
loan_term_months = 30 * 12
grace_period_months = 12  # 1년 거치
repayment_start = datetime(2025, 9, 1)
extra_repayment_start = datetime(2028, 10, 1)
extra_repayment_amount = 15_000_000

# 상환 계산
balance = loan_amount
monthly_principal = loan_amount / (loan_term_months - grace_period_months)

data = []
month = 0
current_date = repayment_start

while balance > 0:
    interest = balance * monthly_rate
    principal = min(monthly_principal, balance)
    total_payment = interest + principal
    balance -= principal

    # 중도상환: 2028년 10월부터 매년 10월에 1,500만 원 추가 상환
    extra_payment = 0
    if current_date >= extra_repayment_start and current_date.month == 10:
        extra_payment = min(extra_repayment_amount, balance)
        balance -= extra_payment
        total_payment += extra_payment

    data.append({
        "Date": current_date,
        "Principal": round(principal),
        "Interest": round(interest),
        "ExtraRepayment": round(extra_payment),
        "TotalPayment": round(total_payment),
        "RemainingBalance": round(balance)
    })

    current_date = pd.to_datetime(current_date) + pd.DateOffset(months=1)
    month += 1

# DataFrame 생성
df = pd.DataFrame(data)

# 필터링: 10월과 11월만 보기
df_filtered = df[df['Date'].dt.strftime('%m').isin(['10', '11'])].reset_index(drop=True)

# 결과 확인
print(df_filtered)