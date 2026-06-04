import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import random

try:
    with open("highscore.txt", "r+") as f:
        highscore = int(f.read())
        oldhighscore = highscore #Creating this so we know when the current round has beaten the previous score
except:
    highscore = 0
    oldhighscore = 0

retry_hover_start = None

button_x = 450
button_y = 550

button_w = 350
button_h = 100

apple_img = cv2.imread("assets/apple.png", cv2.IMREAD_UNCHANGED)
orange_img = cv2.imread("assets/orange.png", cv2.IMREAD_UNCHANGED)
pineapple_img = cv2.imread("assets/pineapple.png", cv2.IMREAD_UNCHANGED)
watermelon_img = cv2.imread("assets/watermelon.png", cv2.IMREAD_UNCHANGED)
strawberry_img = cv2.imread("assets/strawberry.png", cv2.IMREAD_UNCHANGED)

fruit_images = [
    apple_img,
    orange_img,
    pineapple_img,
    watermelon_img,
    strawberry_img
]

cv2.namedWindow("Webcam", cv2.WND_PROP_FULLSCREEN)

cv2.setWindowProperty(
    "Webcam",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
base_options = python.BaseOptions(
    model_asset_path='hand_landmarker.task'
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.4
)

detector = vision.HandLandmarker.create_from_options(options)

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]
def draw_landmarks_on_image(frame, detection_result):
    global previous_x, previous_y

    annotated_image = np.copy(frame)

    hand_landmarks_list = detection_result.hand_landmarks

    height, width, _ = annotated_image.shape

    for idx in range(len(hand_landmarks_list)):
        
        hand_landmarks = hand_landmarks_list[idx]  
        index_tip = hand_landmarks[8]
        tip_x = int(index_tip.x * width)
        tip_y = int(index_tip.y * height)
        if previous_x is None:
            #Keeping track of the previous fingertip positions to smooth sudden jumps

            previous_x = tip_x 
            previous_y = tip_y
        smooth_x = int(previous_x * 0.85 + tip_x * 0.15)
        smooth_y = int(previous_y * 0.85 + tip_y * 0.15)
        distance = ((smooth_x - previous_x) ** 2 + (smooth_y - previous_y) ** 2) ** 0.5
        if distance > 80:
            return annotated_image
        previous_x = smooth_x
        previous_y = smooth_y
        trail_points.append((smooth_x,smooth_y))
        #Creating a short sword trail by storing recent fingertip positions

        if len(trail_points) > 10:
            trail_points.pop(0)
        cv2.circle(
            annotated_image, (smooth_x,smooth_y), 20, (0,0,255), -1
        )

        points = []

        for landmark in hand_landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

            cv2.circle(
                annotated_image,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )
        for i in range(1,len(trail_points)):
            cv2.line(
                annotated_image, 
                trail_points[i-1], 
                trail_points[i], 
                (255,0,255), 
                8
            )
        for connection in HAND_CONNECTIONS:

            start_idx = connection[0]
            end_idx = connection[1]

            cv2.line(
                annotated_image,
                points[start_idx],
                points[end_idx],
                (255, 255, 255),
                2
            )
        

    return annotated_image
def draw_fruit(frame, fruit):

    img = fruit["image"]

    img = cv2.resize(
        img,
        None,
        fx=0.625,
        fy=0.625
    )

    original_h, original_w = img.shape[:2]

    if fruit.get("side") == "left":
        center = (original_w, original_h // 2)

    elif fruit.get("side") == "right":
        center = (0, original_h // 2)

    else:
        center = (original_w // 2, original_h // 2)
    #Making the size of each fruit half bigger so the split pieces don't clip out and look weird
    diagonal = int(np.sqrt(
        original_w * original_w +
        original_h * original_h
    ))

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        fruit["angle"],
        1.0
    )

    rotation_matrix[0, 2] += (
        diagonal / 2 - center[0]
    )
    rotation_matrix[1, 2] += (
        diagonal / 2 - center[1]
    )

    img = cv2.warpAffine(
        img,
        rotation_matrix,
        (diagonal, diagonal),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    h, w = img.shape[:2]

    x = int(fruit["x"] - w // 2)
    y = int(fruit["y"] - h // 2)

    if (
        x + w < 0 or
        y + h < 0 or
        x > frame.shape[1] or
        y > frame.shape[0]
    ):
        return

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)

    img_x1 = x1 - x
    img_y1 = y1 - y

    img_x2 = img_x1 + (x2 - x1)
    img_y2 = img_y1 + (y2 - y1)

    img_crop = img[
        img_y1:img_y2,
        img_x1:img_x2
    ]

    alpha = img_crop[:, :, 3] / 255.0
    rgb = img_crop[:, :, :3]

    roi = frame[y1:y2, x1:x2]

    for c in range(3):
        roi[:, :, c] = (
            alpha * rgb[:, :, c] +
            (1 - alpha) * roi[:, :, c]
        )

def draw_restart_button(frame, trail_points):
    global retry_hover_start

    bx, by, bw, bh = 465, 580, 350, 80
    cx, cy = bx + bw // 2, by + bh // 2
    HOVER_DURATION = 3.0

    hovering = False

    if len(trail_points) >= 3 and all(
        bx <= px <= bx + bw and by <= py <= by + bh
        for px, py in trail_points
    ):
        hovering = True
        if retry_hover_start is None:
            retry_hover_start = time.time()
    else:
        retry_hover_start = None

    progress = 0.0
    if retry_hover_start is not None:
        progress = min((time.time() - retry_hover_start) / HOVER_DURATION, 1.0)
        if progress >= 1.0:
            retry_hover_start = None
            return True   

    
    if hovering:
        bg_color   = (40, 180, 40)
        text_color = (255, 255, 255)
        border_col = (100, 255, 100)
    else:
        bg_color   = (30, 30, 30)
        text_color = (200, 200, 200)
        border_col = (120, 120, 120)

    
    cv2.rectangle(frame, (bx + 4, by + 4), (bx + bw + 4, by + bh + 4), (0, 0, 0), -1)
    
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), bg_color, -1)
    
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), border_col, 2)

    if progress > 0:
        axes = (bw // 2 - 6, bh // 2 - 6)
        angle_end = int(-90 + 360 * progress)
        cv2.ellipse(frame, (cx, cy), axes, 0, -90, angle_end, (100, 255, 100), 3)

    label = "PLAY AGAIN"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, label, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    if hovering and progress > 0:
        secs_left = int(np.ceil(HOVER_DURATION - (time.time() - retry_hover_start))) if retry_hover_start else 0
        hint = f"Hold {secs_left}s..."
        (hw, hh), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(frame, hint, (cx - hw // 2, by - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)

    return False


trail_points=[]
previous_x = None
previous_y = None
'''fruit_x = 500
fruit_y = 900

fruit_vel_x = 0
fruit_vel_y = -25
fruit_radius = 40
'''
gravity = 0.4
last_seen_time = time.time()
fruits = []

for i in range(3):
    fruit = {
        "x": np.random.randint(200,1700),
        "y": 900,

        "vx": np.random.randint(-5,5),
        "vy": np.random.randint(-25,-17),

        "radius":40,

        "image": random.choice(fruit_images),

        "angle": 0,
        "rotation_speed": random.randint(-10,10)

    }
    fruits.append(fruit)
score = 0
split_fruits =[]


game_duration = 15
start_time = time.time()
game_over = False
while True:
    elapsed_time = time.time() - start_time
    time_left = max(0, int(game_duration - elapsed_time))

    if elapsed_time >= game_duration:
        game_over = True
    success, frame = cap.read()
    if not game_over:
        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(time.perf_counter() * 1000)

        detection_result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )
        annotated_frame = draw_landmarks_on_image(
            frame,
            detection_result
        )
        for fruit in fruits:
            fruit["x"] += fruit["vx"]
            fruit["y"] += fruit["vy"]
            fruit["x"] += fruit["vx"]
            fruit["y"] += fruit["vy"]

            #Checking if it touches the side walls to bounce off them
            if fruit["x"] - fruit["radius"] <= 0:
                fruit["x"] = fruit["radius"]
                fruit["vx"] *= -0.9

            if fruit["x"] + fruit["radius"] >= annotated_frame.shape[1]:
                fruit["x"] = annotated_frame.shape[1] - fruit["radius"]
                fruit["vx"] *= -0.9

            fruit["vy"] += gravity
            fruit["angle"] += fruit["rotation_speed"]
            fruit["vy"] += gravity
            fruit["angle"] += fruit["rotation_speed"]
            draw_fruit(
                annotated_frame,
                fruit
            )
            #Respawning if not sliced
            if fruit["y"] > 1200:

                fruit["x"] = np.random.randint(200, 1700)
                fruit["y"] = 900

                fruit["vx"] = np.random.randint(-5, 5)
                fruit["vy"] = np.random.randint(-25, -17)

                fruit["image"] = random.choice(fruit_images)

                fruit["angle"] = 0
                fruit["rotation_speed"] = random.randint(-10,10)
            if len(trail_points) > 0:
                last_seen_time = time.time()
                sword_x, sword_y = trail_points[-1]
                #The main logic to check if the sword intersects the fruit hitbox
                distance = (
                    (sword_x - fruit["x"]) ** 2 +
                    (sword_y - fruit["y"]) ** 2
                ) ** 0.5
            
                if distance < fruit["radius"]:

                    score += 1

                    img = fruit["image"]

                    h, w = img.shape[:2]

                    left_img = img[:, :w//2]
                    right_img = img[:, w//2:]

                    left_half = {
                        "side": "left",

                        "x": fruit["x"] - 15,
                        "y": fruit["y"],

                        "vx": -6,
                        "vy": -10,

                        "angle": fruit["angle"],
                        "rotation_speed": -20,

                        "image": left_img,

                        "life": 120
                    }

                    right_half = {
                        "side": "right",
                        "x": fruit["x"] + 15,
                        "y": fruit["y"],

                        "vx": 6,
                        "vy": -6,

                        "angle": fruit["angle"],
                        "rotation_speed": 20,

                        "image": right_img,

                        "life": 120
                    }

                    split_fruits.append(left_half)
                    split_fruits.append(right_half)

                    fruit["x"] = np.random.randint(200,1700)
                    fruit["y"] = 900

                    fruit["vx"] = np.random.randint(-5,5)
                    fruit["vy"] = np.random.randint(-25,-17)

                    fruit["image"] = random.choice(fruit_images)

                    fruit["angle"] = 0
                    fruit["rotation_speed"] = random.randint(-10,10)

            if len(trail_points) == 0:
                if time.time() - last_seen_time <0.3:
                    pass
                else:
                    trail_points.clear()
        for piece in split_fruits:

            piece["x"] += piece["vx"]
            piece["y"] += piece["vy"]
            if piece["x"] <= 0:
                piece["x"] = 0
                piece["vx"] *= -0.9

            if piece["x"] >= annotated_frame.shape[1]:
                piece["x"] = annotated_frame.shape[1]
                piece["vx"] *= -0.9
            piece["vy"] += gravity

            piece["angle"] += piece["rotation_speed"]

            piece["life"] -= 1

            draw_fruit(
                annotated_frame,
                piece
            )
        split_fruits = [
            piece
            for piece in split_fruits
            if piece["life"] > 0
        ]


        cv2.putText(
            annotated_frame, 
            f"Score {score}", 
            (50,50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0,0,0),
            12
        )

        cv2.putText(
            annotated_frame, 
            f"Score {score}", 
            (50,50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255,255,255),
            3
        )
        
        

        cv2.putText(
            annotated_frame,
            f"Time: {time_left}",
            (50, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            12
        )

        
        cv2.putText(
            annotated_frame,
            f"Time: {time_left}",
            (50, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3
        )
        

        cv2.putText(
            annotated_frame,
            f"High Score: {highscore}",
            (50, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,0),
            12
        )

        cv2.putText(
            annotated_frame,
            f"High Score: {highscore}",
            (50, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            3
        )

        cv2.putText(
            annotated_frame,
            f"Press 'q' to exit",
            (1000, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,0),
            12
        )

        cv2.putText(
            annotated_frame,
            f"Press 'q' to exit",
            (1000, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,128),
            3
        )

    if game_over:

        if success:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.perf_counter() * 1000)
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            annotated_frame = draw_landmarks_on_image(frame, detection_result)

        overlay = annotated_frame.copy()
        if score > highscore:
            highscore = score
            with open ("highscore.txt", "w") as f:
                f.write(str(highscore))
        cv2.rectangle(
            overlay,
            (0, 0),
            (annotated_frame.shape[1], annotated_frame.shape[0]),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.6,
            annotated_frame,
            0.4,
            0,
            annotated_frame
        )

        cv2.putText(
            annotated_frame,
            "GAME OVER!",
            (350, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 0),
            18
        )

        cv2.putText(
            annotated_frame,
            "GAME OVER!",
            (350, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 255, 255),
            6
        )
        

        cv2.putText(
            annotated_frame,
            f"Final Score: {score}",
            (400, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            15
        )

        cv2.putText(
            annotated_frame,
            f"Final Score: {score}",
            (400, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            4
        )
        if highscore > oldhighscore:


            cv2.putText(
                annotated_frame,
                f"New High Score! {highscore}",
                (350, 500),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0,0,0),
                15
            )
            
            cv2.putText(
                annotated_frame,
                f"New High Score! {highscore}",
                (350, 500),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0,215,255),
                4
            )
        else:

            cv2.putText(
                annotated_frame,
                f"High Score: {highscore}",
                (400, 500),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0,0,0),
                15
            )

            cv2.putText(
                annotated_frame,
                f"High Score: {highscore}",
                (400, 500),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255,255,255),
                4
            )

        cv2.putText(
            annotated_frame,
            f"Press 'q' to exit",
            (1000, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,0),
            15
        )

        cv2.putText(
            annotated_frame,
            f"Press 'q' to exit",
            (1000, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,128),
            4
        )

        if draw_restart_button(annotated_frame, trail_points):
            
            oldhighscore = highscore
            score = 0
            split_fruits.clear()
            trail_points.clear()
            previous_x = None
            previous_y = None
            for fruit in fruits:
                fruit["x"] = np.random.randint(200, 1700)
                fruit["y"] = 900
                fruit["vx"] = np.random.randint(-5, 5)
                fruit["vy"] = np.random.randint(-25, -17)
                fruit["image"] = random.choice(fruit_images)
                fruit["angle"] = 0
                fruit["rotation_speed"] = random.randint(-10, 10)
            start_time = time.time()
            game_over = False

    cv2.imshow("Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()