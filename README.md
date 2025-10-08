# Soccer Tracker

An AI-powered soccer analysis system that uses computer vision and machine learning to track players, ball, and referees in soccer videos. The system provides real-time analysis including player tracking, team assignment, speed calculation, distance measurement, and ball possession tracking.

## Features

- **Object Detection & Tracking**: Detects and tracks players, ball, goalkeeper, and referees using YOLOv5
- **Team Assignment**: Automatically assigns players to teams based on jersey colors using K-means clustering
- **Ball Possession Tracking**: Determines which player has control of the ball
- **Speed & Distance Analysis**: Calculates player speed (km/h) and distance covered (meters)
- **Camera Movement Compensation**: Adjusts tracking for camera movement using optical flow
- **View Transformation**: Transforms player positions to a top-down field view
- **Video Output**: Generates annotated video with all tracking information

## Project Structure

```
soccer-tracker/
├── main.py                           # Main application entry point
├── yolo_inference.py                 # YOLO model inference testing
├── models/                           # Trained model files
│   └── best.pt                       # YOLOv5 trained model
├── input_videos/                     # Input video files
├── output_videos/                    # Generated output videos
├── stubs/                           # Cached tracking data
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── video_utils.py               # Video reading/writing functions
│   └── bbox_utils.py                # Bounding box utility functions
├── trackers/                        # Object tracking implementation
├── team_assigner/                   # Team assignment logic
├── player_ball_assigner/            # Ball possession assignment
├── camera_movement_estimator/       # Camera movement detection
├── view_transformer/                # Field view transformation
├── speed_and_distance_estimator/    # Speed and distance calculations
├── training/                        # Model training scripts and data
│   ├── football_training_yolo_v5.ipynb
│   └── football-players-detection-1/
└── development_and_analysis/        # Analysis notebooks
    └── color_assignment.ipynb
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)

### Dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install roboflow
```

### Clone Repository

```bash
git clone <repository-url>
cd soccer-tracker
```

### Download Trained Model

The project uses a custom-trained YOLOv5 model. Ensure you have the trained model file (`best.pt`) in the `models/` directory.

## Usage

### Basic Usage

1. Place your input video in the `input_videos/` directory
2. Run the main tracking script:

```bash
python main.py
```

3. The output video with annotations will be saved in `output_videos/`

### Testing YOLO Inference

To test the YOLO model on a single video:

```bash
python yolo_inference.py
```

### Training Custom Model

To train your own model:

1. Open the training notebook:
```bash
jupyter notebook training/football_training_yolo_v5.ipynb
```

2. Follow the notebook instructions to download the dataset and train the model

## Model Training

The project uses a custom dataset from Roboflow with 4 classes:
- Ball
- Goalkeeper  
- Player
- Referee

Training configuration:
- Model: YOLOv5x
- Epochs: 100
- Image size: 640x640
- Batch size: 16

## Key Components

### Object Detection
- Uses YOLOv5 for detecting players, ball, goalkeeper, and referees
- Custom trained model with soccer-specific dataset

### Team Assignment
- Analyzes jersey colors using K-means clustering
- Automatically assigns players to teams based on dominant colors

### Ball Possession
- Calculates distance between ball and players
- Assigns ball possession to nearest player

### Speed & Distance Tracking
- Tracks player movement across frames
- Calculates speed in km/h and total distance in meters
- Compensates for camera movement

### Camera Movement
- Uses optical flow to detect camera movement
- Adjusts player positions for accurate tracking

## Output

The system generates:
- Annotated video with bounding boxes around detected objects
- Team colors displayed for each player
- Speed and distance information overlaid on players
- Ball possession indicators
- Camera movement compensation data

## Performance Optimization

- Use `read_from_stub=True` to cache tracking results
- GPU acceleration recommended for real-time processing
- Adjust batch size based on available memory

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure `best.pt` is in the `models/` directory
2. **Video codec issues**: Install additional codecs if video saving fails
3. **Memory errors**: Reduce batch size or use smaller input videos
4. **Slow inference**: Enable GPU acceleration with CUDA

### Performance Tips

- Use shorter video clips for testing
- Cache results using stub files
- Ensure adequate GPU memory for large videos
