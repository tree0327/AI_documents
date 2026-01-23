# 더하기 함수
def add(a, b):
    return a + b

# 빼기 함수 
def subtract(a, b):
    return a - b

# 곱하기 함수
def multiply(a, b):
    return a * b

# 나누기 함수
def divide(a, b):
    # 만약에 나누는 수가 0이면 valueError
    if b == 0:
        raise ValueError("0으로는 못나")
    return a / b

if __name__ == "__main__":
    print(f"add(10, 5) = {add(10, 5)}")
    print(f"subtract(10, 5) = {subtract(10, 5)}")
    print(f"multiply(10, 5) = {multiply(10, 5)}")
    print(f"divide(10, 5) = {divide(10, 5)}")
