import os

# 수정할 세 폴더 경로
base_path = "/workspace/airkon4/Downloads/FHD/training/"
folders = ["calib", "image_2", "label_2"]

for folder in folders:
    dir_path = os.path.join(base_path, folder)
    for filename in os.listdir(dir_path):
        if filename.startswith("frame_"):
            old_path = os.path.join(dir_path, filename)
            new_filename = filename.replace("frame_", "", 1)  # 첫 번째 'frame_'만 제거
            new_path = os.path.join(dir_path, new_filename)

            os.rename(old_path, new_path)
            print(f"✅ renamed: {filename} → {new_filename}")
