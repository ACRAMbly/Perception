# ACRAMbly-Perception

**Real-time 6DoF Object Pose Estimation for Robotic Assembly**

[![ROS2](https://img.shields.io/badge/ROS2-Jazzy-blue.svg)](https://docs.ros.org/en/jazzy/)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

**ACRAMbly-Perception** is a robust perception pipeline developed for the ACRAMbly Master Project at the University of Bremen. This system enables autonomous robots to accurately detect and estimate 6DoF (6 Degrees of Freedom) poses of objects in real-time assembly scenarios.

The pipeline integrates:  **FoundationPose**,  **Grounded SAM**,  **ROS2 Jazzy**,  **Intel RealSense**

### Installation

Quick setup:
```bash
# Clone the repository
git clone https://github.com/ACRAMbly/ACRAMbly-Perception.git
cd ACRAMbly-Perception

# Install dependencies
pip install -r requirements.txt

# Build extensions (required for first-time setup)
bash build_all.sh
```

### Running the Perception Pipeline

1. **Start the RealSense camera**:
```bash
source /opt/ros/jazzy/setup.bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

2. **Launch the perception node**:
```bash
source /opt/ros/jazzy/setup.bash
cd /path/to/ACRAMbly-Perception

python3 integration_test.py \
  --mesh_path demo_data/cube/mesh/textured.obj \
  --prompt_text "cube" \
  --debug 1
```

3. **View results**:
- RViz2: `ros2 run rviz2 rviz2`
- Console logs and visualization window

---

## 🎮 Usage Examples

### Example 1: Single Object Tracking

```bash
python3 main.py \
  --mesh_path demo_data/mustard0/mesh/textured.obj \
  --prompt_text "mustard bottle" \
  --debug 2 \
  --publish_pose \
  --publish_tf
```

### Example 2: Multi-Object Detection

```bash
python3 multi_thread_multi_object.py \
  --mesh_dir demo_data/ycbv/models/ \
  --prompt_text "objects on table" \
  --debug 1
```

### Example 3: With Custom Camera Topics

```bash
python3 main.py \
  --rgb_topic /my_camera/rgb \
  --depth_topic /my_camera/depth \
  --camera_info_topic /my_camera/info \
  --mesh_path path/to/mesh.obj \
  --prompt_text "target object"
```

---

## 📊 Performance

Tested on: NVIDIA RTX 3090, Intel i9-12900K, 64GB RAM

| Metric | Performance |
|--------|-------------|
| Detection FPS | ~10-15 Hz |
| Pose Estimation FPS | ~8-12 Hz |
| Pose Accuracy (YCB-Video) | <2cm translation error |
| Latency (camera to pose) | ~100-150ms |

---

## 🐛 Troubleshooting

### Camera not detected
```bash
# Check RealSense device
rs-enumerate-devices

# Verify ROS2 topics
ros2 topic list | grep camera
```

### CUDA out of memory
- Reduce image resolution
- Use smaller SAM model (`sam_b` instead of `sam_l`)
- Close other GPU applications

### No detections
- Adjust `--box_threshold` and `--text_threshold`
- Improve prompt text (be more specific)
- Check lighting conditions and object visibility

## 🤝 Contributing

This is an academic project developed as part of the ACRAMbly Master Project. Contributions, suggestions, and improvements are welcome!
## 📄 License

This project builds upon:
- **FoundationPose** - NVIDIA Corporation ([Paper](https://arxiv.org/abs/2312.08344))
- **Grounded SAM** - IDEA Research ([GitHub](https://github.com/IDEA-Research/Grounded-Segment-Anything))

Please respect the original licenses of these components.

---

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{acramblyperc2025,
  title = Real-time 6DoF Pose Estimation for Robotic Assembly: ACRAMbly Perception Pipeline,
  school = University of Bremen, AICOR-Institute of AI,
  year = 2025,
  type = Master's Project
}
```
**Made with ❤️ at the Institute for Artificial Intelligence, University of Bremen**
