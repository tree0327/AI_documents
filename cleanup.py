import shutil
import os

base_dir = "/Users/gimdabin/AI_documents/AI_documents/01_파이썬_기초"
targets = [
    "crawling",
    "데이터분석및시각화",
    "딥러닝",
    "머신러닝",
    "수업자료",
    "예복습모음",
    "__pycache__"
]

print(f"Cleaning up {base_dir}...")
for target in targets:
    path = os.path.join(base_dir, target)
    if os.path.exists(path):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"[OK] Deleted: {target}")
        except Exception as e:
            print(f"[ERR] Failed to delete {target}: {e}")
    else:
        print(f"[SKIP] Not found: {target}")
