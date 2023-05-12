import cv2
import mediapipe as mp
import math
import numpy as np
import serial
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# 获取自己的音频设备及其参数
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


# 由于vol_range与0-100这个不是对应的关系，不方便设置实际的声音，故需要进行装换，但是无法得知其转换关系，只能通过字典的形式查询：
def vol_tansfer(x):
    dict = {0: -65.25, 1: -56.99, 2: -51.67, 3: -47.74, 4: -44.62, 5: -42.03, 6: -39.82, 7: -37.89, 8: -36.17,
            9: -34.63, 10: -33.24,
            11: -31.96, 12: -30.78, 13: -29.68, 14: -28.66, 15: -27.7, 16: -26.8, 17: -25.95, 18: -25.15, 19: -24.38,
            20: -23.65,
            21: -22.96, 22: -22.3, 23: -21.66, 24: -21.05, 25: -20.46, 26: -19.9, 27: -19.35, 28: -18.82, 29: -18.32,
            30: -17.82,
            31: -17.35, 32: -16.88, 33: -16.44, 34: -16.0, 35: -15.58, 36: -15.16, 37: -14.76, 38: -14.37, 39: -13.99,
            40: -13.62,
            41: -13.26, 42: -12.9, 43: -12.56, 44: -12.22, 45: -11.89, 46: -11.56, 47: -11.24, 48: -10.93, 49: -10.63,
            50: -10.33,
            51: -10.04, 52: -9.75, 53: -9.47, 54: -9.19, 55: -8.92, 56: -8.65, 57: -8.39, 58: -8.13, 59: -7.88,
            60: -7.63,
            61: -7.38, 62: -7.14, 63: -6.9, 64: -6.67, 65: -6.44, 66: -6.21, 67: -5.99, 68: -5.76, 69: -5.55, 70: -5.33,
            71: -5.12, 72: -4.91, 73: -4.71, 74: -4.5, 75: -4.3, 76: -4.11, 77: -3.91, 78: -3.72, 79: -3.53, 80: -3.34,
            81: -3.15, 82: -2.97, 83: -2.79, 84: -2.61, 85: -2.43, 86: -2.26, 87: -2.09, 88: -1.91, 89: -1.75,
            90: -1.58,
            91: -1.41, 92: -1.25, 93: -1.09, 94: -0.93, 95: -0.77, 96: -0.61, 97: -0.46, 98: -0.3, 99: -0.15, 100: 0.0}
    return dict[x]


# 设置声音大小
volume.SetMasterVolumeLevel(vol_tansfer(60), None)

# Initialize Mediapipe Hand model and OpenCV camera capture
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set up drawing tools
mp_drawing = mp.solutions.drawing_utils
draw_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize minimum and maximum degree values
min_degree = 180
max_degree = 0

activate_serial = False
ser = None

serial_port = "COM9"
serial_rate = 115200

if activate_serial:
    # Initialize serial connection
    ser = serial.Serial(serial_port, serial_rate)


def serial_write(d):
    global ser, activate_serial
    if activate_serial:
        if ser is None:
            ser = serial.Serial(serial_port, serial_rate)

        ser.write(("%d\n" % int(d)).encode('utf-8'))


def switch_callback(event, x, y, flags, param):
    global activate_serial
    if event == cv2.EVENT_LBUTTONDOWN and 10 <= x <= 110 and 100 <= y <= 150:
        activate_serial = not activate_serial


start_time = time.time()
# Start capturing video frames
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the image to RGB format
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    agl = None
    # Run hand detection on the image
    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        results = hands.process(image)

        # Draw hand landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    draw_spec,
                    draw_spec)

                # if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
                #     pass
                # else:
                #     continue
                # Calculate degree of curvature of index finger
                index_finger_landmarks = hand_landmarks.landmark[
                                         mp_hands.HandLandmark.INDEX_FINGER_MCP:mp_hands.HandLandmark.INDEX_FINGER_TIP]
                if len(index_finger_landmarks) == 3:
                    p1 = (index_finger_landmarks[0].x, index_finger_landmarks[0].y)
                    p2 = (index_finger_landmarks[1].x, index_finger_landmarks[1].y)
                    p3 = (index_finger_landmarks[2].x, index_finger_landmarks[2].y)
                    angle = math.degrees(
                        math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
                    if angle < 0:
                        angle += 360

                    # Update minimum and maximum degree values if necessary
                    if cv2.waitKey(1) & 0xFF == ord('n'):
                        min_degree = angle if angle < min_degree else min_degree
                    elif cv2.waitKey(1) & 0xFF == ord('m'):
                        max_degree = angle if angle > max_degree else max_degree

                    agl = angle

                    # Map degree to range 0-180 and write to serial port
                    mapped_degree = np.interp(angle, [min_degree, max_degree], [0, 180])
                    volume.SetMasterVolumeLevel(vol_tansfer(int(np.interp(angle, [min_degree, max_degree], [0, 100]))),
                                                None)
                    serial_write(mapped_degree)

    # Convert the image back to BGR format for display

    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    start_time = time.time()
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    switch_text = "Serial: {}".format("On" if activate_serial else "Off")
    cv2.putText(image, switch_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    switch_color = (0, 255, 0) if activate_serial else (0, 0, 255)
    switch_x = 10
    switch_y = 100
    switch_width = 100
    switch_height = 50

    # Draw switch button
    switch_text = "Serial: {}".format("On" if activate_serial else "Off")
    cv2.putText(image, switch_text, (switch_x, switch_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    switch_color = (0, 255, 0) if activate_serial else (0, 0, 255)
    switch_thickness = 2
    df = 20
    cv2.rectangle(image, (switch_x + df, switch_y + df), (switch_x + switch_width - df, switch_y + switch_height - df),
                  switch_color,
                  thickness=switch_thickness)

    # Draw switch toggle
    switch_toggle_x = switch_x + switch_thickness + df if not activate_serial else (
            switch_x + switch_width - switch_thickness - 2 * df)
    switch_toggle_y = switch_y + int(switch_height / 2)
    switch_toggle_color = (0, 255, 0) if activate_serial else (0, 0, 255)
    if activate_serial:
        cv2.circle(image, (switch_toggle_x + df, switch_toggle_y),
                   df,
                   switch_toggle_color, thickness=-1)
    else:
        cv2.circle(image, (switch_toggle_x, switch_toggle_y),
                   df,
                   switch_toggle_color, thickness=-1)

    degree_range = max_degree - min_degree
    if degree_range > 0 and agl is not None:
        current_degree_pos = int((agl - min_degree) / degree_range * 200)
        cv2.rectangle(image, (10, 180), (210, 200), (255, 255, 255), thickness=2)
        cv2.rectangle(image, (10, 180), (10 + current_degree_pos, 200), (0, 255, 0), thickness=-1)
    else:
        cv2.rectangle(image, (10, 180), (210, 200), (255, 0, 0), thickness=2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Display the resulting image
    cv2.imshow('Hand Detection', image)
    cv2.setMouseCallback('Hand Detection', switch_callback)

    if cv2.waitKey(1) == 27:
        break
    if cv2.getWindowProperty('Hand Detection', cv2.WND_PROP_VISIBLE) < 1:
        break
# Release OpenCV camera capture and close all windows
cap.release()
cv2.destroyAllWindows()
