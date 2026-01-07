# Thermal Camera Viewer for Mega-Idea Super IR Cam 2S Pro

A feature-rich thermal camera viewer for Linux with real-time temperature display, markers, recording, and multiple visualization modes.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Linux-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This viewer was developed specifically for the **Mega-Idea Super IR Cam 2S Pro** thermal camera, commonly used for PCB diagnostics and electronics repair. It provides real-time thermal imaging with temperature measurements without requiring Windows software.

<img width="1133" height="677" alt="Screenshot example" src="https://github.com/user-attachments/assets/c29b17da-a3e2-4e93-83f1-62362bd881e6" />


### Key Features

- **Real-time thermal imaging** with multiple color palettes
- **Temperature markers** — click to add measurement points
- **Auto/Manual temperature scaling**
- **Min/Max/Average temperature display**
- **Isotherm highlighting** — highlight specific temperature ranges
- **Screenshot and video recording**
- **Freeze frame** for detailed analysis
- **Temperature histogram**
- **Celsius/Fahrenheit toggle**
- **Resizable window**

## Tested Configuration

| Component | Version/Model |
|-----------|---------------|
| Camera | Mega-Idea Super IR Cam 2S Pro |
| OS | Ubuntu 24.04 LTS |
| Python | 3.10+ |
| Kernel | 6.x |

> ⚠️ **Note**: This software has been tested **only** with the Mega-Idea Super IR Cam 2S Pro on Ubuntu. Other configurations may require modifications.

## Requirements

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt install python3 python3-pip v4l-utils
```

### Python Dependencies

```bash
sudo apt install python3-opencv python3-numpy python3-pil
```

Or using virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy pillow
```

### Fonts (Optional)

For best text quality, ensure you have TrueType fonts installed:

```bash
sudo apt install fonts-dejavu fonts-liberation
```

## Installation

1. Clone or download the repository:
```bash
https://github.com/SkyEng1neering/thermal-camera-viewer.git
cd thermal-camera-viewer
```

2. Install dependencies:
```bash
sudo apt install python3-opencv python3-numpy python3-pil
```

3. Connect your thermal camera via USB

4. Verify the camera is detected:
```bash
lsusb | grep -i camera
v4l2-ctl --list-devices
```

## Usage

### Basic Usage

```bash
python3 thermal_viewer.py
```

### Command Line Options

```
Usage: python3 thermal_viewer.py [options]

Options:
  -h, --help            Show help message
  -d, --device DEVICE   Video device (default: /dev/video0)
  -r, --resolution WxH  Frame resolution (default: 256x384)
```

### Examples

```bash
# Default device and resolution
python3 thermal_viewer.py

# Specify video device
python3 thermal_viewer.py -d /dev/video1

# Custom resolution
python3 thermal_viewer.py -r 320x480

# Both device and resolution
python3 thermal_viewer.py -d /dev/video1 -r 256x384
```

### Find Your Camera Device

```bash
# List all video devices
v4l2-ctl --list-devices

# Check camera capabilities and supported resolutions
v4l2-ctl -d /dev/video0 --list-formats-ext
```

## Controls

### Mouse Controls

| Action | Description |
|--------|-------------|
| **Left Click** | Add temperature marker |
| **Shift + Left Click** | Remove nearest marker |
| **Ctrl + Left Click** | Remove nearest marker |
| **Middle Click** | Clear all markers |
| **Drag** | Move existing marker |

### Keyboard Controls

| Key | Description |
|-----|-------------|
| `q` / `ESC` | Quit application |
| `s` | Save screenshot + temperature data (.npy) |
| `r` | Start/stop video recording |
| `Del` / `Backspace` | Remove last marker |
| `x` | Clear all markers |
| `c` | Cycle through colormaps |
| `1`-`6` | Quick select colormap |
| `a` | Toggle auto-scale |
| `+` / `-` | Adjust max temperature |
| `[` / `]` | Adjust min temperature |
| `0` | Reset scale to auto |
| `m` | Cycle marker mode (All/Max/Min/None) |
| `i` | Toggle isotherm mode |
| `↑` `↓` `←` `→` | Adjust isotherm range |
| `f` | Freeze/unfreeze frame |
| `h` | Toggle histogram |
| `t` | Toggle °C / °F |
| `p` | Toggle side panel |

### Available Colormaps

1. **Inferno** — Default, good contrast
2. **Jet** — Classic rainbow thermal
3. **Hot** — Black-red-yellow-white
4. **Turbo** — Improved rainbow
5. **Gray** — Grayscale
6. **Iron** — Iron/bone palette

## Output Files

### Screenshots

```
thermal_20250107_143052.png      # Visual screenshot
thermal_20250107_143052_temps.npy  # Raw temperature data (NumPy array)
```

### Video Recording

```
thermal_rec_20250107_143052.mp4  # H.264 encoded video
```

### Loading Temperature Data in Python

```python
import numpy as np

# Load temperature array (192x256, values in Celsius)
temps = np.load('thermal_20250107_143052_temps.npy')

print(f"Min: {temps.min():.1f}°C")
print(f"Max: {temps.max():.1f}°C")
print(f"Shape: {temps.shape}")
```

## Compatible Cameras

This viewer **may work** with other thermal cameras that:

1. Use **UVC (USB Video Class)** interface
2. Output **YUYV** format at **256×384** resolution
3. Pack **16-bit temperature data** in the lower half of the frame
4. Use **InfiRay** or compatible sensor with formula: `T(°C) = raw_value / 64 - 273.15`

### Potentially Compatible Models

- Mega-Idea thermal cameras (Super IR Cam series)
- InfiRay-based thermal cameras
- Topdon TC-series (some models)
- Other Chinese thermal cameras using InfiRay sensors

### Known to NOT Work

- FLIR cameras (different protocol)
- Seek Thermal (different protocol)
- Cameras without UVC support
- Cameras with different temperature encoding

## Resolution Support

The viewer supports custom resolutions via command line argument:

```bash
python3 thermal_viewer.py -r 320x480
```

### Resolution Format

The viewer expects the frame to be split into two halves:
- **Upper half**: Thermal image (grayscale)
- **Lower half**: 16-bit temperature data

For example, with resolution `256x384`:
- Total frame: 256×384
- Thermal image: 256×192 (upper half)
- Temperature data: 256×192 (lower half)

### Default Resolution

If not specified, the default resolution is **256×384**, which is standard for Mega-Idea Super IR Cam 2S Pro.

### Finding Supported Resolutions

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

Look for YUYV format entries — those are the supported resolutions.

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| **Ubuntu 24.04** | ✅ Tested | Primary development platform |
| **Ubuntu 22.04** | ⚠️ Should work | Not tested |
| **Debian 12** | ⚠️ Should work | Not tested |
| **Fedora** | ⚠️ Should work | Not tested |
| **Arch Linux** | ⚠️ Should work | Not tested |
| **Windows** | ❌ Not supported | Uses V4L2 (Linux-specific) |
| **macOS** | ❌ Not supported | Uses V4L2 (Linux-specific) |

### Windows/macOS Users

For Windows or macOS, consider:
- Using the official MIIR software
- Running Linux in a VM with USB passthrough
- Porting the viewer to use platform-specific camera APIs

## Troubleshooting

### Camera not detected

```bash
# Check USB connection
lsusb

# Check video devices
ls -la /dev/video*

# Check kernel messages
dmesg | tail -20
```

### Permission denied

```bash
# Add user to video group
sudo usermod -aG video $USER
# Log out and log back in
```

### Wrong resolution

```bash
# Check supported formats
v4l2-ctl -d /dev/video0 --list-formats-ext

# Set correct resolution manually
v4l2-ctl -d /dev/video0 --set-fmt-video=width=256,height=384,pixelformat=YUYV
```

### Black/empty image

The camera may need a few frames to "wake up". The viewer automatically skips initial frames, but if you still see issues:

```bash
# Test with ffplay first
ffplay -f v4l2 -video_size 256x384 /dev/video0
```

### Font issues

If text looks pixelated, install better fonts:

```bash
sudo apt install fonts-dejavu-core fonts-liberation
```

## Technical Details

### Temperature Formula

```
Temperature (°C) = raw_16bit_value / 64.0 - 273.15
```

This formula is specific to InfiRay sensors and converts the raw 16-bit value to Celsius.

### Frame Structure

```
┌─────────────────────────┐
│                         │
│   Thermal Image (Y)     │  256 × 192 pixels
│                         │  8-bit grayscale
│                         │
├─────────────────────────┤
│                         │
│   Temperature Data      │  256 × 192 pixels
│   (16-bit Little-Endian)│  Raw sensor values
│                         │
└─────────────────────────┘
        256 pixels
```

### Data Format

- **YUYV (YUV 4:2:2)**: 2 bytes per pixel
- **Temperature encoding**: Low byte in Y channel, High byte in UV channel
- **Byte order**: Little-endian (low byte first)

## License

MIT License — feel free to use, modify, and distribute.

## Contributing

Contributions are welcome! If you:
- Test with a different camera model
- Port to another platform
- Add new features
- Fix bugs

Please open an issue or pull request.

## Acknowledgments

- Temperature formula derived from InfiRay sensor documentation
- Built with OpenCV, NumPy, and Pillow

---

**Disclaimer**: This is an unofficial third-party tool. The author is not affiliated with Mega-Idea or InfiRay. Use at your own risk.
