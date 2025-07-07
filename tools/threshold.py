import os

def filter_txt_by_score(folder_path, min_score=1.0):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    for file_name in txt_files:
        full_path = os.path.join(folder_path, file_name)

        with open(full_path, 'r') as f:
            lines = f.readlines()

        filtered_lines = []
        for line in lines:
            tokens = line.strip().split()
            try:
                score = float(tokens[-1])  # 마지막 토큰이 score
                if score >= min_score:
                    filtered_lines.append(line)
            except (ValueError, IndexError):
                # 마지막 값이 숫자가 아니거나 비정상 라인인 경우
                continue

        # 파일 덮어쓰기
        with open(full_path, 'w') as f:
            f.writelines(filtered_lines)

    print(f"✅ 완료: {len(txt_files)}개 파일에서 score < {min_score} 라인을 필터링했습니다.")

filter_txt_by_score('/workspace/airkon4/MonoDETR/pitch_45_v2/monodetr/outputs/data')