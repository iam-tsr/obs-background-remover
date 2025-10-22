## OBS - Background Remover [In Progress]

An initiative to support creators and streamers who use Linux operating systems, with a focus on providing high-quality tools to simplify their workflows through this project.

Currently, this project is standalone and supports <u>**Fedora only**</u>.

Our ongoing development aims to continually enhance the model and user experience while optimizing system resource usage.
​
### Features

The current version introduces a virtual camera with **real-time background blur** that can be used as a source in OBS Studio. We acknowledge that installation is not seamless yet; although packaging options like Flatpak are under consideration, this release is intended solely for testing.

### Getting Started

#### Quick Start Guide - Background Blur for OBS

#### 1. Install Dependencies

For installing dependencies you can either create a python virtual environment (Recommended) or install all the dependencies globally in your system. Make sure your system has python v3.11 installed because some dependencies will not work on higher version of python.

All you have to do is to run below cmd in your terminal:

```bash
source run.sh
```

#### 2. Run command

**To start the virtual camera, with no preview:**

```python
python ./virtual_camera.py
```

**With preview**

```python
python ./virtual_camera.py --preview
```

**Other parameters**

```bash
--input N       Use /dev/videoN as input (default: 0)
--blur N        Set blur strength 5-101 (default: 55)
--preview       Show preview window
--width N       Set output width (default: 640)
--height N      Set output height (default: 480)
--fps N         Set output FPS (default: 30)

Example:

   ./virtual_camera.py --input 0 --blur 75
```

### Contribution

We welcome engineers and researchers interested in computer vision and video segmentation models. Open pull requests or contact tusharsoni.info@gmail.com, or send a direct message on [LinkedIn](https://www.linkedin.com/in/imtsr/).

### Acknowledgments

- Special thanks to **v4l2loopback** for making such great kernel module accessable on Linux.
- Google's **MediaPipe**
- **OpenCV**
