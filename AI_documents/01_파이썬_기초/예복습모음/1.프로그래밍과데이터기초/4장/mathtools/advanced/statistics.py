import math

# 평균
def mean(data):
    # 데이터가 비어있으면 0으로 리턴
    if not data:
        return 0
    # sum(data) -> 모든 수의 합, len(data) -> 숫자의 개수
    return sum(data) / len(data)

# 중앙값
def median(data):
    if not data:
        return 0
        
    # 숫자들을 작은 수부터 큰 수 순서로 정렬
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2 # 가운데 위치 인덱스 => 전체개수 / 2의 몫
    
    # 개수가 짝수라면, 가운데 두 수의 평균
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        # 개수가 홀수라면, 정확히 가운데 있는 숫자가 중앙값
        return sorted_data[mid]

# 표준편차: 평균으로부터 얼마나 퍼져있는가
def standard_deviation(data):
    # 데이터가 2개 미만이면 퍼진 정도를 알 수 없어요.
    if len(data) < 2:
        return 0
    
    m = mean(data) # 평균
    
    # 평균과의 차이를 제곱해서 다 더하기
    sum_squared_diff = 0
    for x in data:
        diff = x - m
        
        sum_squared_diff += diff ** 2 # (x - m)의 제곱
        
    # 다 더한 값을 (개수 - 1)로 나누면 분산
    variance = sum_squared_diff / (len(data) - 1)
    
    # 분산에 루트(제곱근)를 씌우면 표준편차
    return math.sqrt(variance)

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    print(f"data = {data}")
    print(f"mean(data) = {mean(data)}")
    print(f"median(data) = {median(data)}")
    print(f"standard_deviation(data) = {standard_deviation(data)}")
