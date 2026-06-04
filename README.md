# Fruit Ninja CV

A gesture-controlled Fruit Ninja clone built using Python, OpenCV, and MediaPipe Hand Tracking. Slice fruits in real-time using your index finger and compete for the highest score.

## Features

* Real-time hand tracking using MediaPipe
* Finger-controlled fruit slicing
* Animated fruit splitting effects
* Physics-based fruit movement
* Persistent high score system
* Fullscreen gameplay
* Gesture-controlled restart button
* Hover-to-restart mechanic (hold over button for 3 seconds)
* Multiple fruit types with rotating animations

## Demo

Control the game using your webcam:

1. Launch the game.
2. Move your hand in front of the camera.
3. Use your index finger as a blade.
4. Slice fruits to earn points.
5. After the timer ends, hover over the restart button for 3 seconds to play again.

## Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy

## Installation

### Clone the repository

```bash
git clone https://github.com/your-username/fruit-ninja-cv.git
cd fruit-ninja-cv
```

### Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

### Download MediaPipe Hand Landmarker

Place the `hand_landmarker.task` file in the project root directory.

### Project Structure

```text
Fruit-Ninja-CV/
│
├── assets/
│   ├── apple.png
│   ├── orange.png
│   ├── pineapple.png
│   ├── strawberry.png
│   └── watermelon.png
│
├── hand_landmarker.task
├── highscore.txt
├── main.py
└── README.md
```

## Running the Game

```bash
python main.py
```

## Controls

| Action       | Control                                 |
| ------------ | --------------------------------------- |
| Slice Fruit  | Move index finger through fruit         |
| Restart Game | Hover over restart button for 3 seconds |
| Quit         | Press Q                                 |

## Gameplay

* The game lasts 15 seconds.
* Each fruit sliced increases your score.
* Fruits split into animated halves when cut.
* High scores are automatically saved between sessions.
* Beat your previous best score to unlock a "New High Score" notification.

## Future Improvements

* Bombs and penalties
* Combo system
* Sound effects and music
* Difficulty levels
* Additional fruit types
* Power-ups
* Leaderboard support

## Built using:

* OpenCV for rendering and image processing
* MediaPipe for real-time hand tracking
* NumPy for mathematical operations

