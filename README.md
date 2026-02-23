Multimodal Tracker 🖐️🙂

An advanced multimodal interaction system based on computer vision that combines Hand Tracking and Face Mesh recognition to trigger events and visual overlays in real-time.

This project utilizes MediaPipe for geometric inference and OpenCV for image processing, specifically optimized for Linux environments (Fedora/Hyprland).
🚀 Key Features

    Simultaneous Multimodal Detection: Tracks hands and face concurrently without significant performance degradation.

    "Combo" System: A logical architecture that maps pairs of (Hand Gesture, Facial Expression) to specific actions.

        Example: A "Thumbs Up" paired with a "Smile" generates a different overlay than a "Thumbs Up" with a "Neutral" face.

    Real-Time Visual Feedback: Image overlays with transparency support (Alpha Channel/BGRA).

    Custom Geometric Classification: Proprietary algorithms to determine states such as "Surprise" or "Wink" based on Euclidean distances and facial proportions.

🛠️ System Requirements

    Operating System: Linux (Tested on Fedora 42 with Hyprland).

    Python: Versions 3.8 to 3.11.

        Important Note: The project was developed and validated on Python 3.11. Higher versions (3.12+) show incompatibilities with certain dependencies (specifically mediapipe/distutils) as of November 2025.

    Hardware: Functional webcam.

📦 Installation

Follow these steps to set up the environment from scratch:
1. Clone the Repository
Bash

git clone https://github.com/your-user/hand-tracker.git
cd hand-tracker

2. Create a Virtual Environment (Recommended)

To keep dependencies isolated from your main system:
Bash

python3 -m venv venv_gestos
source venv_gestos/bin/activate

3. Install Dependencies

Install the required libraries by running:
Bash

pip install opencv-python mediapipe numpy matplotlib

4. ⚠️ Resource Configuration (CRITICAL)

The system requires a specific folder for graphic resources that is not included in the repository by default. You must create it manually and add your images.

    Create the images folder in the project root:
    Bash

    mkdir images

    Add .png files inside that folder. For the system to function, the filenames must match those defined below (or you can modify the paths in actions.py). Ensure you have the following images:

        Basics: like.png, dislike.png, rock.png, peace.png

        Emotions: shocked.png, look_there.png, party.png

        Positivity: super_like.png, hello.png, happy_vibes.png, idea.png

        Winks: secret.png, target_locked.png, bro_fist.png, high_five.png

    Note: The system will automatically normalize the images to BGRA format and resize them, but it is recommended to use PNG images with transparent backgrounds for the best visual effect.

📐 Technical Foundations (Mathematical Breakdown)

The classification core does not rely on "black box" neural networks for final classification, but rather on analytic geometry applied over the landmarks extracted by MediaPipe.
1. Hand Classification (Vector Logic)

To determine if a finger is raised, we avoid the computational cost of square roots in every frame by comparing squared Euclidean distances (d2).

Let Pwrist​ be the wrist, Ptip​ the fingertip, and Ppip​ the intermediate joint:
d2(Pwrist​,Ptip​)>d2(Pwrist​,Ppip​)⟹Finger Raised
2. Surprise Detection (MAR - Mouth Aspect Ratio)

To detect an open mouth (surprise), we calculate the mouth aspect ratio using the Euclidean distance:
MAR=∣∣Pleft​−Pright​∣∣∣∣Ptop​−Pbottom​∣∣​

Where ∣∣⋅∣∣ is the Euclidean norm. If MAR>0.45, it is classified as SURPRISED.
3. Wink Detection (EAR - Eye Aspect Ratio)

We use the standard EAR metric to determine eye openness. Six landmarks are considered per eye (p1​…p6​):
EAR=2⋅∣∣p1​−p4​∣∣∣∣p2​−p6​∣∣+∣∣p3​−p5​∣∣​

The system detects an intentional wink by comparing the EAR of both eyes:
If (EARleft​<0.2∧EARright​>0.2)⟹WINK_LEFT
🎮 Usage

To start the main tracking system:
Bash

python tracker.py

Controls

    ESC: Close the window and terminate the program.

⚙️ Advanced Configuration
Camera Selection

The tracker.py file attempts to locate a specific camera by its hardware ID (/dev/v4l/by-id/...) to prevent issues on Linux systems with multiple video devices.

If your camera is not detected, edit the line in tracker.py:
Python

# Change this to your camera index (usually 0 or 1)
stable_path = "/path/to/your/camera" 
# Or force the index directly in cv2.VideoCapture(0)

📂 Project Structure
Plaintext

hand-tracker/
├── actions.py       # Combo logic controller and image loading
├── tracker.py       # Main entry point (Vision loop)
├── images/          # [YOU MUST CREATE THIS] PNG resource folder
├── .gitignore       # Git exclusion configuration
└── README.md        # Documentation

🤝 Contribution

If you wish to add new combos, edit the self.combo_map dictionary in actions.py and add the corresponding image to the images/ folder.
Python

# Example of a new combo
("FIST", "SMILE"): "./images/power_up.png",

by Rosh.
