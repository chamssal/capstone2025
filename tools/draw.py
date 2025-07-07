import os
import cv2
import numpy as np

def compute_box_3d(dim, location, ry):
    l, w, h = dim  # length, width, height
    x, y, z = location

    # 3D bounding box corners in object coordinate system
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    corners = np.array([x_corners, y_corners, z_corners])  # 3x8

    # Rotation matrix around Y axis
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # rotate and translate
    corners_3d = R @ corners + np.array([[x], [y], [z]])
    return corners_3d.T  # 8x3

def project_to_image(pts_3d, P):
    # pts_3d: (N, 3)
    pts_3d_homo = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))  # Nx4
    pts_2d = pts_3d_homo @ P.T  # Nx3
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    return pts_2d.astype(int)

def draw_3d_box(image, pts_2d):
    # 연결할 점의 인덱스
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 아래 면
        (4, 5), (5, 6), (6, 7), (7, 4),  # 위 면
        (0, 4), (1, 5), (2, 6), (3, 7)   # 위-아래 연결
    ]
    for i, j in connections:
        pt1 = tuple(pts_2d[i])
        pt2 = tuple(pts_2d[j])
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

def draw_3d_boxes_on_val_images(val_list_path, val_img_dir, output_txt_dir, vis_output_dir, P):
    os.makedirs(vis_output_dir, exist_ok=True)

    with open(val_list_path, 'r') as f:
        val_ids = [line.strip() for line in f if line.strip()]

    for img_id in val_ids:
        img_name = img_id + '.png'
        img_path = os.path.join(val_img_dir, img_name)
        txt_path = os.path.join(output_txt_dir, img_id + '.txt')
        output_img_path = os.path.join(vis_output_dir, img_name)

        if not os.path.exists(txt_path) or not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 15:
                continue  # 필수 항목 부족하면 스킵

            # confidence 필터링: parts[15]가 존재하면 score로 판단
            if len(parts) >= 16:
                score = float(parts[15])
                if score == 0.0:
                    continue  # confidence가 0이면 건너뜀

            # 추출
            h = float(parts[8])
            w = float(parts[9])
            l = float(parts[10])
            x = float(parts[11])
            y = float(parts[12])
            z = float(parts[13])
            ry = float(parts[14])

            corners_3d = compute_box_3d((l, w, h), (x, y, z), ry)
            pts_2d = project_to_image(corners_3d, P)
            draw_3d_box(image, pts_2d)


        cv2.imwrite(output_img_path, image)

    print("✅ 모든 val 이미지에 3D 박스를 시각화했습니다.")

P2 = np.array([
    [447.655352, 0.0, 960.0, 0.0],
    [0.0, 447.655352, 540.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
])

draw_3d_boxes_on_val_images(
    val_list_path=r"D:/KITTI3/KITTI/ImageSets/val.txt",
    val_img_dir=r"D:/KITTI3/KITTI/training/image_2",
    output_txt_dir=r"C:\Users\user\MonoDETR\outputs\monodetr\outputs\data",
    vis_output_dir=r"C:\Users\user\MonoDETR\outputs\monodetr\outputs\image_output_3d",
    P=P2
)
