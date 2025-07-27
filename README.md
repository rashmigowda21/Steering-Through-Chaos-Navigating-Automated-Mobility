# Steering-Through-Chaos-Navigating-Automated-Mobility
A simulation-based software system that enables autonomous vehicles to effectively navigate through Indian roads. Key metrics: Obstacle detection accuracy, Non-standard path navigation. Perception modules based on computer vision and sensor fusion techniques.  Simulation Tool: CARLA

## CARLA Autonomous Driving Simulation

This project implements an autonomous driving simulation in CARLA with traffic, pedestrians, and advanced sensor processing including camera, LIDAR, and semantic segmentation.

## Requirements

- CARLA 0.9.13 or later
- Python 3.7 or later
- NumPy
- OpenCV

## Setting Up CARLA

1. Download and install CARLA from the [official website](https://carla.org/download/)
2. Start the CARLA server by running `CarlaUE4.exe` (Windows) or `./CarlaUE4.sh` (Linux)

## Running the Simulation

Run the main script:
```
python simulation.py
```
Additional command-line options:
- `--host`: CARLA server host (default: localhost)
- `--port`: CARLA server port (default: 2000)
- `--timeout`: Client connection timeout in seconds (default: 20.0)

Example:
```
python simulation.py --host localhost --port 2000
```

## Features

- Ego vehicle control with collision avoidance
- Traffic vehicle and pedestrian simulation
- Pothole detection using computer vision
- Obstacle detection using LIDAR point cloud processing
- Lane keeping using semantic segmentation
- Visualization of camera feed and detection results

## How It Works

1. The simulation starts by connecting to the CARLA server and setting up synchronous mode
2. An ego vehicle is spawned along with sensors (camera, LIDAR, semantic camera)
3. NPC vehicles and pedestrians are spawned to create a realistic environment
4. During each simulation tick:
   - Sensor data is processed to detect obstacles, potholes, and lane markings
   - The controller computes appropriate steering, throttle, and brake values
   - Visualization windows show the camera view with detections
5. The simulation continues until interrupted by the user (Ctrl+C)
