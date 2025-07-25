#ALOSS: Automatic Lane and Object Simulation System

**ALOSS** (Automatic Lane and Object Simulation System) is an AI-based video analysis framework that simulates key components of Advanced Driver Assistance Systems (ADAS) without relying on expensive sensors like LiDAR. It uses real dashcam datasets and deep learning models to detect lanes, vehicles, and dynamic interactions on the road.

---

DOWNLOAD DATASET FROM https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k
MAKE SURE TO TAKE IMAGES LABELS AND SEG OUTSIDE THE FOLDER AND STORE IN DOWNLOADS AS TAHTS THE PATH I USED


USE 100K FOR IMAGES

<3


## Features

-  **Lane Detection** with deep learning (LaneNet)
-  **Object Detection** using YOLOv8
-  **Object Tracking** with DeepSORT
- ⚠ **Event Simulation**: Lane departure, collision warnings, overtaking detection
-  **Simulation Visualizer**: Bounding boxes + lane overlays with event annotations
-  **BDD100K Support**: Works with open dashcam datasets for training & testing

---

##  Architecture

- **Input**: Raw dashcam video frames
- **Detection**:
  - YOLOv8 for cars, trucks, pedestrians, etc.
  - LaneNet (segmentation-based) for left/right lane markings
- **Tracking**: DeepSORT associates detections across frames
- **Simulation Engine**:
  - Computes lane alignment, proximity, velocity changes
  - Flags ADAS-like events (drift, overtake, sudden stop)



## 🚀 Getting Started

1. **Clone the repo**
  
   git clone https://github.com/AlanSureshJ/ALOSS.git
   cd ALOSS
 

2. **Install dependencies**

   pip install -r requirements.txt
 

3. **Download pretrained weights**
   - YOLOv8: [Download here](https://github.com/ultralytics/ultralytics)
   - LaneNet (TensorFlow or PyTorch): Add to `lanenet/weights/`


   python main.py --input data/sample.mp4 --output outputs/result.mp4


---

## 🧪 Dataset Support

- Tested with:
  - [BDD100K](https://bdd-data.berkeley.edu/)

---

## 📦 Dependencies

- Python 3.8+
- OpenCV, PyTorch, Ultralytics YOLO, DeepSORT, NumPy, tqdm

---

## 🎯 Applications

- ADAS research & simulation
- Low-cost driver safety evaluation
- Real-time road event monitoring
- AI-based lane behavior analytics

---



## 📬 Contact

**Alan Suresh Joseph**  
Email: alansuresh2004@gmail.com  
GitHub: [AlanSureshJ](https://github.com/AlanSureshJ)

 
 
