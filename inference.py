import cv2
import numpy as np
import torch
import os
import csv # NEW: For logging data to dashboard
from ultralytics import YOLO
from moviepy.video.io.VideoFileClip import VideoFileClip
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
YOLO_MODEL_PATH = r"C:\Users\Yousuf Traders\Desktop\Projects\ATM Activity Dashboard\model\atm.pt"
HAND_CFG = r"C:\Users\Yousuf Traders\Desktop\Projects\ATM Activity Dashboard\model\cross-hands-yolov4-tiny.cfg"
HAND_WEIGHTS = r"C:\Users\Yousuf Traders\Desktop\Projects\ATM Activity Dashboard\model\cross-hands-yolov4-tiny.weights"
VIDEO_SOURCE = r"C:\Users\Yousuf Traders\Desktop\Projects\ATM Activity Dashboard\video\1.mp4"

TEMP_OUTPUT = "temp_raw_recording.avi"
FINAL_OUTPUT = "atm_surveillance.mp4" 
LOG_FILE = "atm_logs.csv" # NEW: The file your Streamlit dashboard will read

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

# --- MAIN SYSTEM ---
def run_system():
    print("🚀 Initializing Models...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    if torch.cuda.is_available(): yolo_model.to('cuda')

    hand_net = cv2.dnn.readNet(HAND_WEIGHTS, HAND_CFG)
    hand_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    hand_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    ln = hand_net.getLayerNames()
    output_layers = [ln[i - 1] for i in hand_net.getUnconnectedOutLayers()]

    checklist = [False] * 4
    task_names = ["Card Insertion", "PIN Entry", "Card Retrieval", "Cash Withdrawal"]
    active_p1 = False

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    ret, test_frame = cap.read()
    if not ret: return
    h, w, _ = test_frame.shape
    
    cv2.namedWindow("CALIBRATION", cv2.WINDOW_NORMAL)
    print("Step 1: Select Keypad and press ENTER")
    r = cv2.selectROI("CALIBRATION", test_frame, False, False)
    cv2.destroyWindow("CALIBRATION")
    
    if r[2] == 0 or r[3] == 0:
        global_keypad_box = (0, 0, 0, 0)
    else:
        global_keypad_box = (int(r[0]), int(r[1]), int(r[0]+r[2]), int(r[1]+r[3]))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    new_width = w + SIDEBAR_WIDTH

    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    video_writer = cv2.VideoWriter(TEMP_OUTPUT, fourcc, fps, (new_width, h))

    pin_frames = 0 
    
    # --- NEW: TIMING VARIABLES ---
    frame_counter = 0
    time_p1 = 0.0 # Time when Card Insertion finishes
    time_p2 = 0.0 # Time when PIN Entry finishes
    time_p3 = 0.0 # Time when Card Retrieval finishes
    time_p4 = 0.0 # Time when Cash Withdrawal finishes

    print("🎥 Recording and analyzing bottlenecks... Please wait.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_counter += 1
        current_time_sec = frame_counter / fps

        results = yolo_model(frame, verbose=False, conf=0.4)[0]
        detected_objects = [YOLO_CLASS_MAP.get(int(box.cls[0])) for box in results.boxes]
        hand_box = get_hand_box_yolov4(hand_net, output_layers, frame)

        # --- LOGIC GATES WITH TIMESTAMPS ---
        
        # Phase 1: Card Insertion
        if "Card" in detected_objects and not checklist[0] and not active_p1:
            active_p1 = True
        
        if active_p1 and "Card" not in detected_objects:
            checklist[0] = True
            active_p1 = False
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

        # --- RENDER DASHBOARD ---
        canvas = np.zeros((h, new_width, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = frame 

        cv2.putText(canvas, "ATM TRANSACTION LOG", (w + 20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
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
        cv2.imshow("ATM Dashboard", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # --- NEW: CALCULATE DURATIONS & SAVE TO CSV ---
    print("\n📊 Saving analytics to CSV for Dashboard...")
    
    # Calculate how long each phase took
    # Fallback to 0 if a phase wasn't completed
    card_insert_duration = time_p1 if time_p1 > 0 else current_time_sec
    pin_entry_duration = (time_p2 - time_p1) if time_p2 > 0 else 0
    card_retrieve_duration = (time_p3 - time_p2) if time_p3 > 0 else 0
    cash_out_duration = (time_p4 - time_p3) if time_p4 > 0 else 0
    total_duration = current_time_sec
    success = checklist[3] # True if all the way to cash out

    # Append to CSV
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write headers if file is new
        if not file_exists:
            writer.writerow(["Transaction_ID", "Total_Duration_Sec", "Approach_Sec", 
                             "Card_Insert_Sec", "PIN_Entry_Sec", "Card_Retrieve_Sec", 
                             "Cash_Out_Sec", "Success"])
        
        # Write this transaction's data (Mocking Approach Time as 5s for now, and TXN ID)
        import random
        txn_id = f"TXN-{random.randint(1000,9999)}"
        writer.writerow([txn_id, total_duration, 5.0, card_insert_duration, 
                         pin_entry_duration, card_retrieve_duration, cash_out_duration, success])
    
    print(f"✅ Analytics saved to: {LOG_FILE}")

    # --- WHATSAPP CONVERSION ---
    print("\n🔄 Converting to WhatsApp Format...")
    try:
        clip = VideoFileClip(TEMP_OUTPUT)
        clip.write_videofile(FINAL_OUTPUT, codec="libx264", audio=False, logger=None)
        clip.close()
        if os.path.exists(TEMP_OUTPUT): os.remove(TEMP_OUTPUT)
        print(f"✅ SUCCESS! File ready for WhatsApp: {FINAL_OUTPUT}")
    except Exception as e:
        print(f"❌ Error during conversion: {e}")

if __name__ == "__main__":
    run_system()