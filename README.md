**Occupancy Detection System**

This project implements an Occupancy Detection System using Detectron2, YOLOv5, and YOLOv8. The system processes real-time video streams or image inputs to detect and count people in a given space.

*Features*

Real-time detection using YOLOv5, YOLOv8, and Detectron2.

Streamlit-based web interface for easy usage.

Supports video and image input for analysis.

Displays detected persons with bounding boxes.

*Installation*

1️⃣ Install Dependencies

Ensure you have Python 3.8+ installed. Then, install the required dependencies:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy pyyaml tqdm tensorboard future
pip install git+https://github.com/facebookresearch/fvcore
pip install git+https://github.com/facebookresearch/iopath
pip install ultralytics streamlit

2️⃣ Install Detectron2

🔹 For GPU (CUDA 12.1)

pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.1/index.html

🔹 For CPU (If no GPU)

pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/index.html

3️⃣ Clone and Run the Project

git clone https://github.com/your-username/occupancy-detection.git
cd occupancy-detection
streamlit run count.py

*Usage*

Upload an image or provide a video stream.

Select the detection model (YOLOv5, YOLOv8, or Detectron2).

Click "Run" to start detection.

View the output with detected persons and occupancy count.

*Project Structure*

📂 occupancy-detection
 ├── 📜 count.py             # Main Streamlit application
 ├── 📜 requirements.txt      # Required dependencies
 ├── 📂 models               # Pre-trained models
 ├── 📂 data                 # Sample images/videos
 ├── 📂 outputs              # Processed results
 └── 📜 README.md            # Project documentation

*References*

Detectron2

YOLOv5

YOLOv8

*Author*

👨‍💻 Srivardhan S.📧 Contact: srivarthansugumar2005@gmail.com🔗 GitHub: github.com/srivardhan03

