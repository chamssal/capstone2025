import os
import shutil

# 경로 설정
val_txt_path = r"D:/KITTI3/KITTI/ImageSets/val.txt"
image_dir = r"D:/KITTI3/KITTI/training/image_2"
label_dir = r"C:/Users/user/MonoDETR/outputs/monodetr/outputs/data"
output_dir = r"D:/KITTI3/val_bundle"

os.makedirs(output_dir, exist_ok=True)

# val.txt 파일 읽기
with open(val_txt_path, 'r') as f:
    val_ids = [line.strip() for line in f if line.strip()]

for img_id in val_ids:
    img_filename = img_id + ".png"
    label_filename = img_id + ".txt"

    src_img_path = os.path.join(image_dir, img_filename)
    src_label_path = os.path.join(label_dir, label_filename)

    dst_img_path = os.path.join(output_dir, img_filename)
    dst_label_path = os.path.join(output_dir, label_filename)

    if os.path.exists(src_img_path):
        shutil.copyfile(src_img_path, dst_img_path)
    else:
        print(f"❌ 이미지 없음: {src_img_path}")

    if os.path.exists(src_label_path):
        shutil.copyfile(src_label_path, dst_label_path)
    else:
        print(f"⚠️ 라벨 없음: {src_label_path}")

print("✅ 완료: 이미지와 라벨을 하나의 폴더로 복사했습니다.")
