import cv2
import mediapipe as mp
import numpy as np
import time

while True:
    print('''Choose the project that you want to test:
1. Hand Tracking: Open or Closed Fist
2. Hand Tracking: Volume and Brightness
3. Hand Tracking: Scroll Up and Down
4. Hand Tracking: Applying Filters and Taking Photos
5. Facial Recognition: Head Counter
6. Exit''')
    
    choice_input = input("Enter your choice in integer here: ")
    if choice_input.strip() == '6' or choice_input.lower() == 'q':
        break

    try:
        choice = int(choice_input)
    except ValueError:
        continue

    if choice == 1:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        mp_draw = mp.solutions.drawing_utils
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access the webcam.")
            continue

        print("Hand Tracking Started! Press 'q' to quit.")

        def detect_gesture(hand_landmarks):
            landmarks = hand_landmarks.landmark
            tip_ids = [4, 8, 12, 16, 20]
            pip_ids = [2, 6, 10, 14, 18]
            extended = 0
            
            if abs(landmarks[tip_ids[0]].x - landmarks[pip_ids[0]].x) > 0.04:
                extended += 1

            for i in range(1, 5):
                if landmarks[tip_ids[i]].y < landmarks[pip_ids[i]].y:
                    extended += 1
            
            if extended >= 4:
                return "Open"
            elif extended <= 1:
                return "Closed Fist"
            else:
                return "Partial"

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            gesture = "No hand detected"

            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_label = results.multi_handedness[idx].classification[0].label
                    gesture = detect_gesture(hand_landmarks)
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    fingertip_ids = [4, 8, 12, 16, 20]
                    for tip_id in fingertip_ids:
                        lm = hand_landmarks.landmark[tip_id]
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x, y), 10, (255, 0, 255), cv2.FILLED)
                        cv2.putText(frame, str(tip_id), (x - 5, y - 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    wrist = hand_landmarks.landmark[0]
                    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(frame, f"{hand_label} Hand", (wrist_x - 40, wrist_y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            status_color = (0, 255, 0) if gesture in ["Open", "Closed Fist"] else (0, 165, 255)
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            cv2.imshow("Hand Gesture Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        hands.close()
    elif choice == 2:
        import cv2, mediapipe as mp, numpy as np
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        import screen_brightness_control as sbc

        Hands = mp.solutions.hands
        hands = Hands.Hands(min_detection_confidence = 0.7, min_tracking_confidence = 0.7)
        draw = mp.solutions.drawing_utils
        TH, IX = Hands.HandLandmark.THUMB_TIP, Hands.HandLandmark.INDEX_FINGER_TIP

        try:
            dev = AudioUtilities.GetDefaultOutputDevice() \
            if hasattr(AudioUtilities, "GetDefaultOutputDevice") \
            else AudioUtilities.GetSpeakers()
            volctl = dev.EndpointVolume.QueryInterface(IAudioEndpointVolume)
            minv, maxv = volctl.GetVolumeRange()[:2]
        except Exception as e:
            print(f"Pycaw error: {e}")
            continue

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error. Webcam is not accessible.")
            continue
        WIN = "Hand Gesture Control"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

        while True:
            ok, img = cap.read()
            if not ok:
                print("Frame was not captured.")
                break
            img = cv2.flip(img, 1)
            h, w = img.shape[:2]
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks and res.multi_handedness:
                for i, hand in enumerate(res.multi_hand_landmarks):
                    label = res.multi_handedness[i].classification[0].label
                    draw.draw_landmarks(img, hand, Hands.HAND_CONNECTIONS)
                    lm = hand.landmark
                    tp = (int(lm[TH].x*w), int(lm[TH].y*h))
                    ip = (int(lm[IX].x*w), int(lm[IX].y*h))
                    cv2.circle(img, tp, 10, (255,0,0), cv2.FILLED);
                    cv2.circle(img, ip, 10, (255,0,0), cv2.FILLED)
                    cv2.line(img, tp, ip, (0,255,0), 3)
                    dist = float(np.hypot(ip[0]-tp[0], ip[1]-tp[1]))

                    if label == "Left":
                        v = np.interp(dist, [30,300], [minv,maxv])
                        try:
                            volctl.SetMasterVolumeLevel(v, None)
                        except Exception as e:
                            print(f"Volume error: {e}")
                        bar = int(np.interp(dist, [30,300], [400,150]))
                        pct = int(np.interp(dist, [30,300], [0,100]))
                        cv2.rectangle(img, (50,150), (85,400), (255,0,0), 2)
                        cv2.rectangle(img, (50,bar), (85,400), (255,0,0), cv2.FILLED)
                        cv2.putText(img, f"{pct}%", (40,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

                    elif label == "Right":
                        b = int(np.interp(dist, [30,300], [0,100]))
                        try:
                            sbc.set_brightness(b)
                        except Exception as e:
                            print(f"Brightness error: {e}")
                        bar = int(np.interp(dist, [30,300], [400,150]))
                        x1, x2 = w-85, w-50
                        cv2.rectangle(img, (x1,150), (x2,400), (0,255,0), 2)
                        cv2.rectangle(img, (x1,bar), (x2,400), (0,255,0), cv2.FILLED)
                        cv2.putText(img, f"{b}%", (w-110,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            cv2.imshow(WIN, img)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
    elif choice == 3:
        import pyautogui

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands = 1, min_detection_confidence = 0.7)
        mp_drawing = mp.solutions.drawing_utils
        SCROLL_SPEED = 300
        SCROLL_DELAY = 1
        CAM_WIDTH, CAM_HEIGHT = 640, 480

        def detect_gesture(landmarks, handedness):
            fingers = []
            tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP,mp_hands.HandLandmark.MIDDLE_FINGER_TIP,mp_hands.HandLandmark.RING_FINGER_TIP,mp_hands.HandLandmark.PINKY_TIP]
            for tip in tips:
                if landmarks.landmark[tip].y < landmarks.landmark[tip - 2].y:
                    fingers.append(1)
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            if (handedness == "Right" and thumb_tip.x > thumb_ip.x) or (handedness == "Left" and thumb_tip.x < thumb_ip.x):
                fingers.append(1)
            return "scroll_up" if sum(fingers) == 5 else "scroll_down" if sum(fingers) == 0 else "none"
        cap = cv2.VideoCapture(0)
        cap.set(3, CAM_WIDTH)
        cap.set(4, CAM_HEIGHT)
        last_scroll = p_time = 0
        print("Gesture Scroll Control Active\nOpen palm: Scroll Up\nFist: Scroll Down\nPress 'q' to exit")
        while cap.isOpened():
            success, img = cap.read()
            if not success: break
            img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1)
            results = hands.process(img)
            gesture, handedness = "none", "Unknown"
            if results.multi_hand_landmarks:
                for hand, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                    handedness = handedness_info.classification[0].label
                    gesture = detect_gesture(hand, handedness)
                    mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
                    if (time.time() - last_scroll) > SCROLL_DELAY:
                        if gesture == "scroll_up":
                            pyautogui.scroll(SCROLL_SPEED)
                        elif gesture == "scroll_down":
                            pyautogui.scroll(-SCROLL_SPEED)
                        last_scroll = time.time()
            fps = 1/(time.time()-p_time) if (time.time()-p_time) > 0 else 0
            p_time = time.time()
            cv2.putText(img, f"FPS: {int(fps)} | Hand: {handedness} | Gesture: {gesture}", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.imshow("Gesture Control", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
    elif choice == 4:
        H = mp.solutions.hands
        TIP = H.HandLandmark
        ids = {
            "thumb": TIP.THUMB_TIP,
            "index": TIP.INDEX_FINGER_TIP,
            "middle": TIP.MIDDLE_FINGER_TIP,
            "ring": TIP.RING_FINGER_TIP,
            "pinky": TIP.PINKY_TIP
            }
        hands = H.Hands(min_detection_confidence=0.7,min_tracking_confidence=0.7)
        draw = mp.solutions.drawing_utils
        pairs = {"middle": ("SEPIA", "NEGATIVE"),"ring": ("BLUR", "GLITCH"),"pinky": ("EDGE", "CARTOON")}
        st = {k:0 for k in pairs}; cur = "SEPIA" 
        DEB, CAP, TT, TP = 0.6, 1.2, 30, 20 
        la = lc = 0; pinch_on = False 
        MAIN, POP = "Gesture-Controlled Photo App", "Captured (ESC / Close to resume)" 
        paused = False; freeze = None 
        SEPIA_M = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
        def apply(img, t): 
            if t == "SEPIA": return np.clip(cv2.transform(img, SEPIA_M), 0, 255).astype(np.uint8) 
            if t == "NEGATIVE":return cv2.bitwise_not(img) 
            if t == "BLUR": return cv2.GaussianBlur(img, (15, 15), 0) 
            if t == "GLITCH":
                h,w = img.shape[:2]; r,g,b = img[:,:,2], img[:,:,1], img[:,:,0] 
                return cv2.merge([np.roll(b, -int(0.02*w), 1), g, np.roll(r, int(0.04*w), 1)]) 
            if t == "EDGE": 
                return cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 160) 
            if t == "CARTOON": 
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                e = cv2.adaptiveThreshold(cv2.medianBlur(g, 7), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2) 
                c = cv2.bilateralFilter(img, 9, 75, 75) 
                return cv2.bitwise_and(c, c, mask=e) 
            return img
        cap = cv2.VideoCapture(0) 
        if not cap.isOpened(): 
            print("Error: Could not access the webcam.")
            continue 
        cv2.namedWindow(MAIN, cv2.WINDOW_NORMAL)
        while True: 
            if paused: 
                cv2.imshow(MAIN, freeze) 
                k = cv2.waitKey(50) & 0xFF 
                if k == ord("q"): 
                    break 
                if k == 27: 
                    paused = False; pinch_on = False 
                    try: cv2.destroyWindow(POP) 
                    except: pass 
                    continue 
                try: 
                    if cv2.getWindowProperty(POP, cv2.WND_PROP_VISIBLE) <= 0: paused = False; pinch_on = False 
                except cv2.error: paused = False; pinch_on = False 
                continue 
            ok, img = cap.read() 
            if not ok: break 
            img = cv2.flip(img, 1); h, w = img.shape[:2] 
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
            now = time.time(); capture = False 
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]; draw.draw_landmarks(img, hand, H.HAND_CONNECTIONS) 
                lm = hand.landmark; tips = {k:(int(lm[v].x*w), int(lm[v].y*h)) for k,v in ids.items()} 
                tx,ty = tips["thumb"]; ix,iy = tips["index"] 
                pinch = abs(tx-ix) < TP and abs(ty-iy) < TP 
                if pinch and not pinch_on and now-lc > CAP: pinch_on = True; capture = True; lc = now 
                if not pinch and pinch_on: pinch_on = False 
                if not pinch: 
                    t = next((k for k in pairs if abs(tx-tips[k][0]) < TT and abs(ty-tips[k][1]) < TT), None) 
                    if t and now-la > DEB: cur = pairs[t][st[t]]; st[t] ^= 1; la = now; print("Filter:", cur) 
            out = apply(img, cur) 
            if cur == "EDGE": out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR) 
            if capture:
                name = f"picture_{int(now)}.jpg"; cv2.imwrite(name, out); print("Saved:", name) 
                paused, freeze = True, out.copy(); cv2.imshow(POP, freeze) 
            cv2.imshow(MAIN, out) 
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break 
        cap.release(); cv2.destroyAllWindows(); hands.close()
    elif choice == 5:
        facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        if facecascade.empty():
            print("Error loading cascade classifier XML file.")
            continue
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("Camera unavailable")
            continue
        while True:
            ret, frame = capture.read()
            if not ret:
                print("Could not capture image.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in face:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"peoplecount:{len(face)}", (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("Face tracking and counting", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()