# ACRAMbly-Perception

**Real-time 6DoF Object Pose Estimation for Robotic Assembly**

[![ROS2](https://img.shields.io/badge/ROS2-Jazzy-blue.svg)](https://docs.ros.org/en/jazzy/)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

**ACRAMbly-Perception** is a robust perception pipeline developed for the ACRAMbly Master Project at the University of Bremen. This system enables autonomous robots to accurately detect and estimate 6DoF (6 Degrees of Freedom) poses of objects in real-time assembly scenarios.

The pipeline integrates:
- **FoundationPose**: State-of-the-art 6D pose estimation and tracking
- **Grounded SAM**: Text-prompted object detection and segmentation  
- **ROS2 Jazzy**: Native integration for seamless robotic system communication
- **Intel RealSense**: RGB-D camera support for depth-aware perception

### 🎯 Project Context

This work is part of the **ACRAMbly Master Project** (12 ECTS, SoSe 2025 & WiSe 2025/2026) at the University of Bremen, supervised by Prof. Michael Beetz. The project aims to build a cognitive architecture for autonomous robotic assembly using PyCRAM, combining industrial automation with AI-driven robot control.

**Key Project Goals:**
- Integrate an assembly station into a cognitive architecture
- Plan action sequences for complete assembly tasks
- Precisely estimate 6D poses of assembly components
- Control dual UR10 robot arms with high precision

---

## ✨ Features

- ✅ **Real-time 6DoF Pose Estimation**: Track objects with millimeter precision
- ✅ **Text-Prompted Detection**: Natural language object detection ("red cube", "metal gear", etc.)
- ✅ **Multi-Object Support**: Concurrent tracking of multiple objects with thread-safe operations
- ✅ **ROS2 Native**: Full ROS2 Jazzy integration with message passing and TF transforms
- ✅ **RealSense Ready**: Out-of-the-box support for Intel RealSense D400 series
- ✅ **Mesh-Based Tracking**: Support for custom CAD models (OBJ, PLY formats)
- ✅ **Visualization Tools**: Real-time 3D bounding boxes, coordinate axes, and pose overlays
- ✅ **Debug Modes**: Comprehensive logging and image saving for development

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  RealSense Camera                       │
│              (RGB-D Image Acquisition)                  │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│              ROS2 Jazzy Topics                          │
│  /camera/color/image_raw                                │
│  /camera/depth/image_rect_raw                           │
│  /camera/color/camera_info                              │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         FoundationPose ROS2 Node                        │
│         (integration_test.py)                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │  1. Grounded SAM Detection                      │   │
│  │     - Text prompt processing                    │   │
│  │     - Bounding box detection                    │   │
│  │     - Mask segmentation                         │   │
│  └────────────────┬────────────────────────────────┘   │
│                   ▼                                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  2. FoundationPose Estimation                   │   │
│  │     - Initial pose estimation                   │   │
│  │     - Pose refinement                           │   │
│  │     - Pose tracking across frames               │   │
│  └────────────────┬────────────────────────────────┘   │
└───────────────────┼──────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              Output Topics                              │
│  /foundationpose/object_pose (PoseStamped)              │
│  /foundationpose/visualization (Image)                  │
│  /tf (TransformStamped)                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Ubuntu 24.04 (or compatible)
- ROS2 Jazzy
- CUDA-capable GPU (recommended: RTX 3090 or better)
- Intel RealSense Camera (D435, D455, or similar)
- Python 3.9+

### Installation

**See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions.**

Quick setup:
```bash
# Clone the repository
git clone https://github.com/ACRAMbly/ACRAMbly-Perception.git
cd ACRAMbly-Perception

# Install dependencies
pip install -r requirements.txt

# Build extensions (required for first-time setup)
bash build_all.sh

# Download model weights (see INSTALLATION.md for links)
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

## 📁 Project Structure

```
ACRAMbly-Perception/
├── integration_test.py           # Main ROS2 perception node
├── estimater.py                   # FoundationPose estimator wrapper
├── datareader.py                  # Data loading utilities
├── Utils.py                       # Visualization and helper functions
│
├── GroundedSAM_demo/             # Grounded SAM integration
│   ├── grounded_sam.py           # Main GroundedSAM class
│   ├── ros2_realsense_groundedsam.py
│   └── utils.py
│
├── ros2_integration/              # ROS2 bridge components
│   ├── ros2_bridge.py            # Alternative bridge implementation
│   ├── foundationpose_node.py    # Node wrapper
│   └── README.md
│
├── rosbag_testing/               # ROS bag testing utilities
│   ├── fixed_demo.py             # Offline testing with bags
│   └── COMPLETE_WORKFLOW_DOCUMENTATION.md
│
├── demo_data/                    # Sample data and test meshes
│   ├── cube/
│   ├── mustard0/
│   └── YCB_Video/
│
├── weights/                      # Model checkpoints (download separately)
├── checkpoints/                  # GroundingDINO checkpoints
│
├── learning/                     # Training scripts (original FoundationPose)
├── bundlesdf/                    # BundleSDF integration
└── mycpp/                        # C++ extensions
```

---

## 🎮 Usage Examples

### Example 1: Single Object Tracking

```bash
python3 integration_test.py \
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
python3 integration_test.py \
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

## 🔧 Configuration

Key parameters in `integration_test.py`:

```python
--prompt_text          # Natural language object description
--mesh_path           # Path to object CAD model (OBJ/PLY)
--mesh_obj_id         # Object ID for YCB/BOP datasets
--debug               # Debug level (0: none, 1: basic, 2: full)
--box_threshold       # GroundingDINO detection threshold (default: 0.25)
--text_threshold      # Text prompt confidence threshold (default: 0.25)
--sam_vit_model       # SAM variant (sam_b, sam_l, sam_h)
--publish_pose        # Publish PoseStamped messages
--publish_tf          # Broadcast TF transforms
```

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

---

## 📚 Documentation

- [Installation Guide](INSTALLATION.md)
- [ROS2 Integration Details](ros2_integration/README.md)
- [Complete Workflow](rosbag_testing/COMPLETE_WORKFLOW_DOCUMENTATION.md)
- [FoundationPose Paper](https://arxiv.org/abs/2312.08344)
- [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)

---

## 🤝 Contributing

This is an academic project developed as part of the ACRAMbly Master Project. Contributions, suggestions, and improvements are welcome!

### Development Team

**Author**: [Ahtasham Ilyas]  
**Institution**: University of Bremen  
**Supervisors**: Prof. Michael Beetz, Jonas Dech, Tom Schierenbeck, Malte Huerkamp

---

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
  author = {Ahtasham Ilyas},
  title = {Real-time 6DoF Pose Estimation for Robotic Assembly: ACRAMbly Perception Pipeline},
  school = {University of Bremen},
  year = {2025},
  type = {Master's Project}
}
```

Also cite the original FoundationPose work:
```bibtex
@InProceedings{foundationposewen2024,
  author    = {Bowen Wen and Wei Yang and Jan Kautz and Stan Birchfield},
  title     = {{FoundationPose}: Unified 6D Pose Estimation and Tracking of Novel Objects},
  booktitle = {CVPR},
  year      = {2024},
}
```

---

## 🔗 Related Projects

- [PyCRAM](https://github.com/cram2/pycram) - Cognitive Robot Abstract Machine
- [FoundationPose](https://github.com/NVlabs/FoundationPose) - Original implementation
- [Isaac ROS Pose Estimation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation) - Production ROS version

---

## 📧 Contact

For questions or collaboration:
- **GitHub Issues**: [Open an issue](https://github.com/ACRAMbly/ACRAMbly-Perception/issues)
- **Project Website**: [ACRAMbly Project Page](https://ai.uni-bremen.de/teaching/)

---

**Made with ❤️ at the Institute for Artificial Intelligence, University of Bremen**
