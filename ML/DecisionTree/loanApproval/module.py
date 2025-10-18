
def predict_loan_approval(points_bin, credit_score, income):
    # points_bin은 'Low', 'Medium', 'High' 중 하나
    if points_bin == 'Low':
        return False                # 점수가 낮으면 무조건 거절
    elif credit_score <= 578.76:
        return False                # 신용 점수가 낮으면 거절
    elif income <= 64726:
        return False                # 소득이 매우 낮으면 거절
    else:
        return True                 # 나머지는 승인