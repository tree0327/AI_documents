import math


# 원의 넓이 (반지름 * 반지름 * pi)
def circle_area(radius):
    return math.pi * radius ** 2 # => radius ** 2 -> 반지름 * 반지름

# 직사각형 넓이 (가로 * 세로)
def rectangle_area(width, height):
    return width * height

# 삼각형 넓이 (밑변 * 높이 / 2)
def triangle_area(base, height):
    return 0.5 * base * height

if __name__ == "__main__":
    print(f"circle_area(3) = {circle_area(3)}")
    print(f"rectangle_area(10, 5) = {rectangle_area(10, 5)}")
    print(f"triangle_area(10, 5) = {triangle_area(10, 5)}")
