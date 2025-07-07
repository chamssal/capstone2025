import os
import cv2
import numpy as np

def compute_box_3d(dim, location, ry):
    l, w, h = dim
    x, y, z = location
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    corners = np.array([x_corners, y_corners, z_corners])
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    corners_3d = R @ corners + np.array([[x], [y], [z]])
    return corners_3d.T

def project_to_image(pts_3d, P):
    pts_3d_homo = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    pts_2d = pts_3d_homo @ P.T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    return pts_2d.astype(int)

def draw_3d_box(image, pts_2d):
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in connections:
        pt1 = tuple(pts_2d[i])
        pt2 = tuple(pts_2d[j])
        cv2.line(image, pt1, pt2, (255, 0, 0), 2)

def visualize_labeled_ground_truth(image_dir, label_dir, output_dir, P):
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])[:20]  # ⬅️ 여기 수정

    for img_name in image_files:
        img_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, img_id + '.txt')
        output_path = os.path.join(output_dir, img_name)

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            h = float(parts[8])
            w = float(parts[9])
            l = float(parts[10])
            x = float(parts[11])
            y = float(parts[12])
            z = abs(float(parts[13]))
            ry = float(parts[14])

            corners_3d = compute_box_3d((l, w, h), (x, y, z), ry)
            pts_2d = project_to_image(corners_3d, P)
            draw_3d_box(image, pts_2d)

        cv2.imwrite(output_path, image)
        print(f"✅ 완료: {img_name}")

    print("🎉 20장 라벨 시각화 완료!")


# 카메라 내부 파라미터 (P2)
P2 = np.array([
    [447.655352, 0.0, 960.0, 0.0],
    [0.0, 447.655352, 540.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
])

visualize_labeled_ground_truth(
    image_dir=r"D:/KITTI3/KITTI/training/image_2",
    label_dir=r"D:/KITTI3/KITTI/training/label_2",
    output_dir=r"D:/KITTI3/KITTI/training/image_output_gt",
    P=P2
)
