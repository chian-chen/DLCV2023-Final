import cv2
import os

def extract_frames(video_path, output_folder):
    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 載入影片
    cap = cv2.VideoCapture(video_path)

    # 初始化幀計數器
    count = 0

    while True:
        # 讀取下一幀
        success, frame = cap.read()

        # 檢查是否成功讀取到幀
        if not success:
            break

        # 儲存幀
        frame_path = os.path.join(output_folder, f"frame{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1

    cap.release()

# 使用方法
video_path = '0e7fba95-22d9-4ab0-9815-4bb7880d8557.mp4'  # 設定影片路徑
output_folder = './frames'       # 設定輸出資料夾
extract_frames(video_path, output_folder)
