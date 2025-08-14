import os
import random

# 이미지가 있는 디렉토리
img_dir = "/workspace/airkon4/Downloads/FHD/training/image_2"
# ImageSets 폴더 (train.txt와 val.txt가 저장될 곳)
imagesets_dir = "/workspace/airkon4/Downloads/FHD/ImageSets"
os.makedirs(imagesets_dir, exist_ok=True)

# 파일 이름 (확장자 제거)
image_ids = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".png")]

# 정렬 & 섞기
image_ids.sort()
random.seed(42)
random.shuffle(image_ids)

# 8:2 비율 분할
split_idx = int(len(image_ids) * 0.7)
val_ids = image_ids[:split_idx]
train_ids = image_ids[split_idx:]

# 파일 저장
with open(os.path.join(imagesets_dir, "train.txt"), "w") as f:
    f.write("\n".join(train_ids))

with open(os.path.join(imagesets_dir, "val.txt"), "w") as f:
    f.write("\n".join(val_ids))

print(f"✅ 저장 완료: {len(train_ids)}개 train / {len(val_ids)}개 val 이미지 ID가 ImageSets에 저장되었습니다.")
