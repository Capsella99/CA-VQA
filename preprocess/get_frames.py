import os
import cv2

# 設定原始影片資料夾及輸出資料夾路徑
input_folder = '/home/ethan/original_videos_h264'
output_folder = '/home/ethan/original_videos_h264'

# 建立輸出資料夾（若不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 獲取所有影片檔案名稱
video_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

# 處理每一個影片檔案
for video_file in video_files:
    video_path = os.path.join(input_folder, video_file)
    cap = cv2.VideoCapture(video_path)
    
    clip_frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target = total_frames // 8
    
    # 建立以影片名稱命名的資料夾
    video_name = os.path.splitext(video_file)[0] + "_frames"
    video_output_folder = os.path.join(output_folder, video_name)
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Cannot read frame from {video_file} at frame {i}")
            break
        if i % target == 0:
            frame = cv2.resize(frame, (224, 224))  # Resize frame to [224, 224]
            frame_filename = os.path.join(video_output_folder, f"frame_{i}.jpg")
            cv2.imwrite(frame_filename, frame)
    
    cap.release()

print("Processing completed.")