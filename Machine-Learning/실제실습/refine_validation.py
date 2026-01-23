
import nbformat

def refine_validation_logic():
    nb_path = 'main.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Locate the cell for "2022년 월드컵 검증"
    target_header = "2022년 월드컵 검증"
    
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and target_header in cell.source:
            # The next cell should be the code cell we want to modify
            code_cell = nb.cells[i+1]
            if code_cell.cell_type == 'code':
                # New logic for the loop
                new_loop_logic = """
# ... (previous code)
    home_data = power_cache[home]
    away_data = power_cache[away]
    
    # power_cache stores dicts now: {'Power': value, 'AvgAge': value}
    home_power = home_data['Power']
    away_power = away_data['Power']
    
    row = {'Power_Diff': home_power - away_power}
    
    if USE_AGE_FEATURE:
        home_age = home_data['AvgAge']
        away_age = away_data['AvgAge']
        row['Age_Diff'] = home_age - away_age
        
    X_test_match = pd.DataFrame([row])
    
    # No changes needed for filtering features as X_test_match is created with correct keys
    
    pred = model.predict(X_test_match)[0]
# ... 
"""
                # Actually, I should replace the entire cell content to be safe and clean.
                # Let's reconstruct the cell content.
                
                new_source = """
# 2022 월드컵 16강 일부 경기 결과 검증
# 실제 결과: 
# 네덜란드 3-1 미국 (네덜란드 승)
# 아르헨티나 2-1 호주 (아르헨티나 승)
# 프랑스 3-1 폴란드 (프랑스 승)
# 잉글랜드 3-0 세네갈 (잉글랜드 승)

test_matches_2022 = [
    ('Netherlands', 'United States', 'Netherlands'),
    ('Argentina', 'Australia', 'Argentina'),
    ('France', 'Poland', 'France'),
    ('England', 'Senegal', 'England'),
    ('Japan', 'Croatia', 'Croatia'), # 승부차기 크로아티아 승
    ('Brazil', 'Korea Republic', 'Brazil'),
]

print(f"=== 2022 월드컵 검증 (사용 모델: {model.__class__.__name__}) ===")
correct = 0
for home, away, winner in test_matches_2022:
    if home not in power_cache or away not in power_cache:
        print(f"데이터 부족: {home} vs {away}")
        continue
        
    home_data = power_cache[home]
    away_data = power_cache[away]
    
    # 딕셔너리 구조에 맞춰 데이터 추출
    home_power = home_data['Power']
    away_power = away_data['Power']
    
    # 예측을 위한 입력 데이터 생성
    input_data = {'Power_Diff': [home_power - away_power]}
    
    if USE_AGE_FEATURE:
        home_age = home_data['AvgAge']
        away_age = away_data['AvgAge']
        input_data['Age_Diff'] = [home_age - away_age]
    
    X_test_match = pd.DataFrame(input_data)
    
    # 모델 예측
    pred = model.predict(X_test_match)[0]
    
    pred_winner = home if pred > 0 else away
    result = "적중" if pred_winner == winner else "실패"
    if result == "적중": correct += 1
        
    print(f"{home} vs {away} -> 예측: {pred_winner} (점수차: {pred:.2f}) [{result}]")

print(f"\\n검증 결과: {correct}/{len(test_matches_2022)} ({correct/len(test_matches_2022)*100:.1f}%)")
"""
                code_cell.source = new_source
                print("2022 World Cup validation logic updated.")
                break
    else:
        print("Target validation cell not found.")

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    refine_validation_logic()
