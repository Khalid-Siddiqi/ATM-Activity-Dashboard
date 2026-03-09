import cv2
import numpy as np
import torch
import os
import csv
import random
from ultralytics import YOLO
from moviepy.video.io.VideoFileClip import VideoFileClip
import warnings

warnings.filterwarnings("ignore")

# --- BATCH CONFIGURATION ---
YOLO_MODEL_PATH = "atm.pt"
HAND_CFG = "cross-hands-yolov4-tiny.cfg"
HAND_WEIGHTS = "cross-hands-yolov4-tiny.weights"

# NEW: Point this to the FOLDER containing your 10 videos
VIDEO_FOLDER = r"C:\Users\gutech\Desktop\atm-activity\video"
LOG_FILE = "atm_logs.csv" 
SIDEBAR_WIDTH = 350  
YOLO_CLASS_MAP = { 0: "Card", 1: "Keypad", 2: "Money" }

# --- Helpers ---
def get_hand_box_yolov4(net, output_layers, frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    best_box = None
    max_conf = 0.0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4: 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                if confidence > max_conf:
                    max_conf = confidence
                    best_box = (x, y, x + w, y + h)
    return best_box

def is_overlapping(box1, box2, padding=30):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    b1_x1 = max(0, b1_x1 - padding); b1_y1 = max(0, b1_y1 - padding)
    b1_x2 += padding; b1_y2 += padding
    x_left = max(b1_x1, b2_x1); y_top = max(b1_y1, b2_y1)
    x_right = min(b1_x2, b2_x2); y_bottom = min(b1_y2, b2_y2)
    return not (x_right < x_left or y_bottom < y_top)

# --- MAIN BATCH SYSTEM ---
def run_batch_system():
    print("🚀 Initializing Models (Doing this once for speed)...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    if torch.cuda.is_available(): yolo_model.to('cuda')

    hand_net = cv2.dnn.readNet(HAND_WEIGHTS, HAND_CFG)
    hand_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    hand_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    ln = hand_net.getLayerNames()
    output_layers = [ln[i - 1] for i in hand_net.getUnconnectedOutLayers()]

    task_names = ["Card Insertion", "PIN Entry", "Card Retrieval", "Cash Withdrawal"]
    
    # Get all mp4 files in the folder
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi'))]
    if not video_files:
        print(f"❌ No videos found in {VIDEO_FOLDER}")
        return
    
    print(f"📁 Found {len(video_files)} videos to process.")

    global_keypad_box = None

    # --- LOOP THROUGH EACH VIDEO ---
    for video_name in video_files:
        video_path = os.path.join(VIDEO_FOLDER, video_name)
        print(f"\n🎬 Processing: {video_name}")

        cap = cv2.VideoCapture(video_path)
        ret, test_frame = cap.read()
        if not ret: 
            print(f"⚠️ Could not read {video_name}. Skipping.")
            continue
        
        # ... (previous code) ...
        h, w, _ = test_frame.shape

        # --- PER-VIDEO CALIBRATION ---
        window_name = f"CALIBRATION - {video_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print(f"\n👉 Step 1: Select Keypad for {video_name} and press ENTER.")
        r = cv2.selectROI(window_name, test_frame, False, False)
        cv2.destroyWindow(window_name)
        
        if r[2] == 0 or r[3] == 0:
            global_keypad_box = (0, 0, 0, 0)
            print("⚠️ No keypad selected. PIN tracking might fail for this video.")
        else:
            global_keypad_box = (int(r[0]), int(r[1]), int(r[0]+r[2]), int(r[1]+r[3]))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        new_width = w + SIDEBAR_WIDTH
        # ... (continue to video_writer setup) ...
        
        # Dynamic filenames so they don't overwrite
        temp_output = f"temp_{video_name}.avi"
        final_output = f"processed_whatsapp_{video_name}"
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        video_writer = cv2.VideoWriter(temp_output, fourcc, fps, (new_width, h))

        # Reset Trackers for the new video
        checklist = [False] * 4
        active_p1 = False
        pin_frames = 0 
        frame_counter = 0
        time_p1 = time_p2 = time_p3 = time_p4 = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_counter += 1
            current_time_sec = frame_counter / fps

            results = yolo_model(frame, verbose=False, conf=0.4)[0]
            detected_objects = [YOLO_CLASS_MAP.get(int(box.cls[0])) for box in results.boxes]
            hand_box = get_hand_box_yolov4(hand_net, output_layers, frame)

            # Phase 1: Card Insertion
            if "Card" in detected_objects and not checklist[0] and not active_p1:
                active_p1 = True
            if active_p1 and "Card" not in detected_objects:
                checklist[0] = True; active_p1 = False
                if time_p1 == 0: time_p1 = current_time_sec

            # Phase 2: PIN Entry
            elif checklist[0] and not checklist[1]:
                if hand_box and is_overlapping(global_keypad_box, hand_box):
                    pin_frames += 1
                    if pin_frames > 15: 
                        checklist[1] = True
                        if time_p2 == 0: time_p2 = current_time_sec

            # Phase 3: Card Retrieval
            elif checklist[1] and not checklist[2]:
                if "Card" in detected_objects: 
                    checklist[2] = True
                    if time_p3 == 0: time_p3 = current_time_sec

            # Phase 4: Cash Withdrawal
            if "Money" in detected_objects: 
                checklist[3] = True
                if time_p4 == 0: time_p4 = current_time_sec

            # RENDER DASHBOARD
            canvas = np.zeros((h, new_width, 3), dtype=np.uint8)
            canvas[0:h, 0:w] = frame 
            cv2.putText(canvas, f"LOG: {video_name}", (w + 20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
            cv2.line(canvas, (w + 20, 65), (w + SIDEBAR_WIDTH - 20, 65), (100, 100, 100), 1)

            for i, name in enumerate(task_names):
                y_offset = 110 + (i * 60)
                cv2.rectangle(canvas, (w + 20, y_offset - 25), (w + SIDEBAR_WIDTH - 20, y_offset + 25), (30, 30, 30), -1)
                cv2.putText(canvas, name, (w + 40, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                if checklist[i]:
                    cv2.circle(canvas, (w + SIDEBAR_WIDTH - 50, y_offset + 5), 8, (0, 255, 0), -1)
                else:
                    cv2.circle(canvas, (w + SIDEBAR_WIDTH - 50, y_offset + 5), 8, (80, 80, 80), 2)

            video_writer.write(canvas)
            cv2.imshow("Batch Processing...", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                print("🛑 User interrupted batch process.")
                cap.release(); video_writer.release(); cv2.destroyAllWindows(); return
        
        cap.release()
        video_writer.release()

        # --- SAVE TO CSV ---
        card_insert_duration = time_p1 if time_p1 > 0 else current_time_sec
        pin_entry_duration = (time_p2 - time_p1) if time_p2 > 0 else 0
        card_retrieve_duration = (time_p3 - time_p2) if time_p3 > 0 else 0
        cash_out_duration = (time_p4 - time_p3) if time_p4 > 0 else 0
        success = checklist[3]

        file_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Transaction_ID", "Total_Duration_Sec", "Approach_Sec", "Card_Insert_Sec", "PIN_Entry_Sec", "Card_Retrieve_Sec", "Cash_Out_Sec", "Success"])
            
            # Using the video filename as the Transaction ID
            txn_id = f"TXN-{video_name.split('.')[0]}" 
            writer.writerow([txn_id, current_time_sec, 5.0, card_insert_duration, pin_entry_duration, card_retrieve_duration, cash_out_duration, success])
        
        # --- WHATSAPP CONVERSION ---
        print(f"🔄 Converting {video_name} for WhatsApp...")
        try:
            clip = VideoFileClip(temp_output)
            clip.write_videofile(final_output, codec="libx264", audio=False, logger=None)
            clip.close()
            if os.path.exists(temp_output): os.remove(temp_output)
            print(f"✅ Finished {video_name}")
        except Exception as e:
            print(f"❌ Conversion failed for {video_name}: {e}")

    cv2.destroyAllWindows()
    print("\n🎉 BATCH PROCESSING COMPLETE! Check your atm_logs.csv and run your Streamlit dashboard.")

if __name__ == "__main__":
    run_batch_system()