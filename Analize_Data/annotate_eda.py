
import json

file_path = '/Users/gimdabin/Analize_Data/eda.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper to create markdown cell
def new_md_cell(source_text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_text.split('\n')]
    }

# Cells to keep (the original code cells)
# We assume the user hasn't modified existing code structure too much since reading.
# We will insert new markdown cells at specific positions.

# Constructing the new list of cells
new_cells = []

# Intro
intro_text = """# 📊 팁(Tips) 데이터 분석하기
안녕하세요! 이 노트북에서는 식당에서 사람들이 **팁(Tip)**을 얼마나 주는지 분석해볼 거예요.
**탐색적 데이터 분석(EDA)** 과정을 통해 데이터 안에 숨겨진 이야기를 찾아봅시다!
- **EDA**가 뭐냐고요? 🧐
  - 요리하기 전에 재료를 맛보고 상태를 확인하는 것처럼, 데이터를 본격적으로 다루기 전에 데이터를 꼼꼼히 살펴보는 과정이에요.
"""
new_cells.append(new_md_cell(intro_text))

# Original cells
cells = nb['cells']
# 0: import
new_cells.append(cells[0])

# 1: load dataset
load_text = """### 📂 데이터 불러오기
먼저 분석할 데이터를 가져와야겠죠?
`seaborn`이라는 시각화 도구 상자에 들어있는 `tips` 데이터를 가져올 거예요.
"""
new_cells.append(new_md_cell(load_text))
new_cells.append(cells[1])

# 2: head
head_text = """### 👀 데이터 미리보기 (`head`)
데이터가 어떻게 생겼는지 앞부분 5줄만 살짝 엿볼까요?
- `total_bill`: 식사 금액 ($)
- `tip`: 팁 금액 ($)
- `sex`: 성별 (Male: 남성, Female: 여성)
- `smoker`: 흡연 여부 (Yes: 흡연, No: 비흡연)
- `day`: 요일 (Thur, Fri, Sat, Sun)
- `time`: 시간대 (Lunch, Dinner)
- `size`: 손님 수
"""
new_cells.append(new_md_cell(head_text))
new_cells.append(cells[2])

# 3: info
info_text = """### ℹ️ 데이터 정보 확인하기 (`info`)
데이터의 전체적인 정보를 확인해요.
- 행(row)은 몇 개인지?
- 열(column)은 몇 개인지?
- 비어있는 값은 없는지?
- 숫자인지 글자인지? (Dtype)
"""
new_cells.append(new_md_cell(info_text))
new_cells.append(cells[3])

# 4: describe
desc_text = """### 🔢 기초 통계량 보기 (`describe`)
숫자로 된 데이터들의 요약된 정보를 한눈에 볼 수 있어요.
- `count`: 개수
- `mean`: 평균 (중간쯤 되는 값)
- `min`: 최솟값 (제일 작은 값)
- `max`: 최댓값 (제일 큰 값)
"""
new_cells.append(new_md_cell(desc_text))
new_cells.append(cells[4])

# 5, 6, 7, 8: value_counts (Categorical)
cat_text = """### 📊 범주형 데이터 세어보기 (`value_counts`)
글자로 된 데이터들이 각각 몇 개씩 있는지 세어볼까요?
성별, 흡연 여부, 요일, 시간대별로 손님이 얼마나 왔는지 알아봐요.
"""
new_cells.append(new_md_cell(cat_text))
new_cells.append(cells[5]) # sex?
new_cells.append(cells[6]) # smoker?
new_cells.append(cells[7]) # day?
new_cells.append(cells[8]) # time? (Expected order based on typical eda structure, adjusting if needed)

# 9: isnull sum
null_text = """### 텅 빈 값 찾기 (`isnull`)
혹시 비어있는 칸(**결측치**)이 있는지 확인해요.
(0이 나오면 빈 칸이 없다는 뜻이라 아주 좋은 거예요!)
"""
new_cells.append(new_md_cell(null_text))
new_cells.append(cells[9])

# 10: Histogram
hist_text = """### 📈 팁 금액 분포 보기 (`Histogram`)
사람들이 팁을 보통 얼마나 주는지 **히스토그램(막대 그래프)**으로 그려봐요.
- `kde=True`는 부드러운 곡선도 같이 그려달라는 뜻이에요.
"""
new_cells.append(new_md_cell(hist_text))
new_cells.append(cells[10])

# 11: Scatter Plot
scatter_text = """### 🌌 식사 금액과 팁의 관계 (`Scatter Plot`)
"밥을 많이 먹으면 팁도 많이 줄까?" 🤔
**산점도(점 그래프)**를 그려서 두 데이터가 어떤 관계인지 알아봐요.
- 점들이 오른쪽 위로 올라가는 모양이면, 식사 금액이 클수록 팁도 많다는 뜻이에요!
"""
new_cells.append(new_md_cell(scatter_text))
new_cells.append(cells[11])

# 12: Box Plot
box_text = """### 📦 요일별 식사 금액 비교 (`Box Plot`)
요일마다 사람들이 밥값을 얼마나 쓰는지 **박스 플롯(상자 그림)**으로 비교해봐요.
- 네모 상자 가운데 선이 '중간값'이에요.
- 상자가 위아래로 길면 금액 차이가 크다는 뜻이에요.
"""
new_cells.append(new_md_cell(box_text))
new_cells.append(cells[12])

# 13: Correlation Heatmap
heat_text = """### 🔥 데이터끼리 얼마나 친한가? (`Correlation Heatmap`)
숫자 데이터들끼리 서로 얼마나 관련이 있는지(**상관관계**) 색깔로 보여주는 **히트맵**이에요.
- 색이 밝을수록(또는 진할수록 설정에 따라 다름) 서로 관련이 깊다는 뜻이에요.
- 1에 가까울수록 아주 친한(정비례) 관계예요!
"""
new_cells.append(new_md_cell(heat_text))
new_cells.append(cells[13])

# 14: Pairplot
pair_text = """### 🧩 한눈에 모든 관계 보기 (`Pairplot`)
모든 숫자 데이터들끼리의 관계를 한 번에 그래프로 그려서 살펴봐요.
`hue` 옵션을 쓰면 성별이나 흡연 여부에 따라 색깔을 다르게 표시할 수 있어요.
"""
new_cells.append(new_md_cell(pair_text))
new_cells.append(cells[14])

nb['cells'] = new_cells

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Annotation for EDA complete!")
