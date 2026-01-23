import json
import json
from bs4 import BeautifulSoup

try:
    import pymysql
except ImportError:
    pymysql = None

# 1. JSON 파일 로드
json_file_path = 'kia.json'  # 파일 경로가 맞는지 확인해주세요

try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    faq_items = data['data']['faqList']['items']
    print(f"총 {len(faq_items)}개의 FAQ 항목을 찾았습니다.")
    
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {json_file_path}")
    exit()
except Exception as e:
    print(f"JSON 로드 중 오류 발생: {e}")
    exit()

# 2. MySQL 연결 설정 (본인의 DB 정보로 수정 필요)
# pymysql이 없다면: pip install pymysql
db_config = {
    'host': 'localhost',
    'user': 'your_username',      # DB 사용자명
    'password': 'your_password',  # DB 비밀번호
    'db': 'your_database_name',   # 사용할 데이터베이스 이름
    'charset': 'utf8mb4'
}

try:
    # 3. 데이터베이스 연결 및 데이터 삽입
    # 실제 연결하려면 아래 주석을 해제하고 정보를 입력하세요.
    # conn = pymysql.connect(**db_config)
    # cur = conn.cursor()
    
    # 테이블 생성 예시 (필요한 경우)
    # cur.execute("""
    #     CREATE TABLE IF NOT EXISTS faq (
    #         id INT AUTO_INCREMENT PRIMARY KEY,
    #         question TEXT,
    #         answer TEXT
    #     )
    # """)
    
    print("\n--- 데이터 추출 및 변환 시작 ---")
    
    for item in faq_items:
        question = item.get('question', '')
        answer_html = item.get('answer', {}).get('html', '')
        
        # BeautifulSoup을 사용하여 HTML 태그 제거 및 텍스트 추출
        soup = BeautifulSoup(answer_html, 'html.parser')
        answer_text = soup.get_text(separator=' ', strip=True)
        
        # 결과 확인용 출력 (상위 3개만)
        if faq_items.index(item) < 3:
            print(f"\n[질문]: {question}")
            print(f"[답변(Clean)]: {answer_text}")
            print("-" * 50)
            
        # DB 삽입 로직 (주석 해제 후 사용)
        # sql = "INSERT INTO faq (question, answer) VALUES (%s, %s)"
        # cur.execute(sql, (question, answer_text))
    
    # DB 변경사항 저장 및 종료
    # conn.commit()
    # conn.close()
    # print("\n모든 데이터가 DB에 저장되었습니다.")
    
    print("\n(현재는 DB 연결 부분이 주석 처리되어 있어 실제 저장은 되지 않았습니다.)")
    print("스크립트 내의 db_config 정보를 수정하고 주석을 해제하여 사용하세요.")

except Exception as e:
    print(f"작업 중 오류 발생: {e}")
