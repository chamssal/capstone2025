import os
import numpy as np
import csv

# GT 중복 매칭 방지: 가장 가까운 GT 1개만 사용
def match_each_pred_to_nearest_gt(preds, gts):
    results = []
    used_gt_indices = set()

    for pred_idx, pred in enumerate(preds):
        distances = np.linalg.norm(gts - pred, axis=1)

        # 이미 매칭된 GT는 무한대로 설정하여 제외
        for i in used_gt_indices:
            distances[i] = np.inf

        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        # GT가 모두 소진된 경우
        if np.isinf(min_dist):
            continue

        used_gt_indices.add(min_idx)
        gt = gts[min_idx]

        results.append({
            "pred_idx": pred_idx,
            "gt_idx": min_idx,
            "pred_x": pred[0],
            "pred_z": pred[1],
            "gt_x": gt[0],
            "gt_z": gt[1],
            "error_m": min_dist
        })

    return results
def parse_kitti_txt(path, mode="pred"):
    """
    KITTI 형식의 txt 파일에서 [x, z] 위치만 추출

    mode:
    - "gt": GT 파일용
    - "pred": 예측 파일용
    """
    coords = []
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("Car"):
                continue
            parts = line.strip().split()

            try:
                x = float(parts[11])
                z = float(parts[13])
                coords.append([x, z])
            except (IndexError, ValueError):
                continue

    return np.array(coords)

# 경로 설정
pred_dir = "/workspace/airkon4/MonoDETR/pitch_45_v2/monodetr/outputs/data"
gt_dir = "/workspace/airkon4/Downloads/KITTI/training/label_2"

all_results = []

for filename in sorted(os.listdir(pred_dir)):
    if not filename.endswith(".txt"):
        continue

    pred_path = os.path.join(pred_dir, filename)
    gt_path = os.path.join(gt_dir, filename)

    if not os.path.exists(gt_path):
        print(f"⚠️ GT 파일 없음: {filename}")
        continue

    preds = parse_kitti_txt(pred_path, mode="pred")
    gts = parse_kitti_txt(gt_path, mode="gt")

    if len(preds) == 0 or len(gts) == 0:
        continue

    matched = match_each_pred_to_nearest_gt(preds, gts)

    for match in matched:
        match["frame"] = filename
        all_results.append(match)

# 결과 저장
output_csv = "all_pred_to_nearest_gt_errors.csv"
with open(output_csv, "w", newline='') as csvfile:
    fieldnames = ["frame", "pred_idx", "gt_idx", "pred_x", "pred_z", "gt_x", "gt_z", "error_m"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_results:
        writer.writerow(row)

# 평균 오차 계산 및 출력
if all_results:
    avg_error = np.mean([row["error_m"] for row in all_results])
    print(f"\n✅ 총 {len(all_results)} 개 예측 객체의 최근접 GT 오차 평균: {avg_error:.4f} meters ({avg_error*100:.2f} cm)")
    print(f"📄 결과 저장됨: {output_csv}")
else:
    print("\n❌ 유효한 객체가 없습니다.")
