#!/usr/bin/env python3
"""
Mega-Idea Super IR Cam 2S Pro - Advanced Thermal Viewer

Usage: 
  python3 thermal_viewer.py [options]

Options:
  -d, --device DEVICE       Video device (default: /dev/video0)
  -r, --resolution WxH      Frame resolution, e.g. 256x384 (default: 256x384)

Examples:
  python3 thermal_viewer.py
  python3 thermal_viewer.py -d /dev/video1
  python3 thermal_viewer.py --resolution 320x480
  python3 thermal_viewer.py -d /dev/video0 -r 256x384

Mouse Controls:
  Left Click         - Add temperature marker
  Shift + Left Click - Remove nearest marker
  Ctrl + Left Click  - Remove nearest marker (alternative)
  Middle Click       - Clear all markers
  Drag               - Move marker (click near existing marker and drag)

Keyboard Controls:
  q / ESC            - Quit
  s                  - Save screenshot + temperature data
  r                  - Start/stop video recording
  Delete / Backspace - Remove last added marker
  x                  - Clear all markers
  
  c                  - Cycle colormap
  1-6                - Quick select colormap
  a                  - Toggle auto-scale
  m                  - Cycle marker mode
  i                  - Toggle isotherm
  f                  - Freeze frame
  h                  - Toggle histogram
  t                  - Toggle C/F
  p                  - Toggle side panel
  
  +/-                - Adjust max temp
  [/]                - Adjust min temp
  0                  - Reset scale
  Arrows             - Adjust isotherm range

Requirements:
  pip3 install opencv-python numpy pillow
"""

import sys
import argparse

def check_dependencies():
    missing = []
    try:
        import cv2
    except ImportError:
        missing.append('opencv-python')
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        missing.append('pillow')
    
    if missing:
        print("=" * 60)
        print("Missing dependencies:", ', '.join(missing))
        print("Install: pip3 install " + ' '.join(missing) + " --break-system-packages")
        print("=" * 60)
        sys.exit(1)

check_dependencies()

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time
import os

# =============================================================================
# Argument parsing
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Thermal Camera Viewer for Mega-Idea Super IR Cam 2S Pro',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                          Use default device and resolution
  %(prog)s -d /dev/video1           Use specific device
  %(prog)s -r 320x480               Use custom resolution
  %(prog)s -d /dev/video1 -r 256x384
        '''
    )
    parser.add_argument('-d', '--device', default='/dev/video0',
                        help='Video device path (default: /dev/video0)')
    parser.add_argument('-r', '--resolution', default='256x384',
                        help='Frame resolution WxH (default: 256x384)')
    return parser.parse_args()

def parse_resolution(res_str: str) -> tuple:
    """Parse resolution string like '256x384' into (width, height)."""
    try:
        parts = res_str.lower().split('x')
        if len(parts) != 2:
            raise ValueError()
        width = int(parts[0])
        height = int(parts[1])
        if width <= 0 or height <= 0:
            raise ValueError()
        if height % 2 != 0:
            print(f"Warning: Height {height} is not even. Thermal data may not align correctly.")
        return width, height
    except:
        print(f"Error: Invalid resolution format '{res_str}'. Expected WxH (e.g., 256x384)")
        sys.exit(1)

# Parse arguments early
ARGS = parse_args()
DEVICE = ARGS.device
WIDTH, HEIGHT = parse_resolution(ARGS.resolution)
THERMAL_HEIGHT = HEIGHT // 2  # Thermal image is top half, temp data is bottom half

# =============================================================================
# Other constants
# =============================================================================
DEFAULT_SCALE = 3
PANEL_WIDTH = 280

COLORMAPS = [
    (cv2.COLORMAP_INFERNO, 'Inferno'),
    (cv2.COLORMAP_JET, 'Jet'),
    (cv2.COLORMAP_HOT, 'Hot'),
    (cv2.COLORMAP_TURBO, 'Turbo'),
    (None, 'Gray'),
    (cv2.COLORMAP_BONE, 'Iron'),
]

# =============================================================================
# Font setup - try to find a good system font
# =============================================================================
def find_font():
    """Find a suitable TTF font on the system."""
    font_paths = [
        # Ubuntu/Debian
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/droid/DroidSans.ttf",
        # Arch/Fedora
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            return path
    return None

FONT_PATH = find_font()

def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Get font at specified size with proper rendering."""
    if FONT_PATH:
        try:
            return ImageFont.truetype(FONT_PATH, size)
        except Exception as e:
            print(f"Font load error: {e}")
    # Fallback - try to load any available font
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except:
        pass
    return ImageFont.load_default()

# Pre-load fonts at different sizes (larger for better quality)
FONTS = {
    'title': get_font(20),
    'large': get_font(18),
    'medium': get_font(16),
    'small': get_font(14),
    'tiny': get_font(13),
}

# =============================================================================
# Data classes
# =============================================================================
@dataclass
class Marker:
    x: int
    y: int
    id: int
    color: Tuple[int, int, int] = (255, 255, 255)

@dataclass
class AppState:
    temp_min: float = 15.0
    temp_max: float = 100.0
    auto_scale: bool = True
    auto_scale_margin: float = 5.0
    
    colormap_idx: int = 0
    display_scale: int = DEFAULT_SCALE
    show_histogram: bool = False
    show_panel: bool = True
    use_fahrenheit: bool = False
    
    markers: List[Marker] = field(default_factory=list)
    marker_counter: int = 0
    marker_mode: int = 0
    marker_modes: Tuple[str, ...] = ('All', 'Max', 'Min', 'None')
    
    dragging_marker: Optional[Marker] = None
    
    isotherm_enabled: bool = False
    isotherm_low: float = 40.0
    isotherm_high: float = 60.0
    
    frozen: bool = False
    frozen_temps: Optional[np.ndarray] = None
    
    recording: bool = False
    video_writer: Optional[cv2.VideoWriter] = None
    record_start_time: float = 0.0
    
    fps: float = 0.0
    thermal_display_width: int = WIDTH * DEFAULT_SCALE

# Global state for mouse callback
g_state: Optional[AppState] = None

# =============================================================================
# Helper functions
# =============================================================================
def raw_to_celsius(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32) / 64.0 - 273.15

def format_temp(c: float, use_f: bool) -> str:
    if use_f:
        return f"{c * 9/5 + 32:.1f}°F"
    return f"{c:.1f}°C"

def apply_colormap(gray: np.ndarray, idx: int) -> np.ndarray:
    cmap, _ = COLORMAPS[idx]
    if cmap is not None:
        return cv2.applyColorMap(gray, cmap)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def find_nearest_marker(markers: List[Marker], x: int, y: int, threshold: int = 20) -> Optional[Marker]:
    nearest = None
    min_dist = float(threshold)
    for m in markers:
        d = np.sqrt((m.x - x)**2 + (m.y - y)**2)
        if d < min_dist:
            min_dist = d
            nearest = m
    return nearest

def marker_color(idx: int) -> Tuple[int, int, int]:
    colors = [
        (255, 255, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (0, 255, 0), (0, 165, 255), (255, 100, 100), (100, 100, 255),
    ]
    return colors[idx % len(colors)]

def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR image."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def bgr_to_rgb(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Convert BGR color to RGB."""
    return (color[2], color[1], color[0])

# =============================================================================
# PIL-based text drawing
# =============================================================================
def draw_text(draw: ImageDraw.ImageDraw, pos: Tuple[int, int], text: str, 
              font_key: str = 'medium', color: Tuple[int, int, int] = (255, 255, 255)):
    """Draw text using PIL."""
    font = FONTS.get(font_key, FONTS['medium'])
    # Convert BGR to RGB for PIL
    rgb_color = bgr_to_rgb(color)
    draw.text(pos, text, font=font, fill=rgb_color)

def draw_text_with_bg(draw: ImageDraw.ImageDraw, pos: Tuple[int, int], text: str,
                      font_key: str = 'medium', color: Tuple[int, int, int] = (255, 255, 255),
                      bg_color: Tuple[int, int, int] = (0, 0, 0), padding: int = 3):
    """Draw text with background rectangle."""
    font = FONTS.get(font_key, FONTS['medium'])
    bbox = draw.textbbox(pos, text, font=font)
    # Draw background
    draw.rectangle([bbox[0] - padding, bbox[1] - padding, 
                    bbox[2] + padding, bbox[3] + padding], fill=bgr_to_rgb(bg_color))
    # Draw text
    draw.text(pos, text, font=font, fill=bgr_to_rgb(color))

# =============================================================================
# Drawing functions
# =============================================================================
def draw_marker_on_pil(draw: ImageDraw.ImageDraw, marker: Marker, temp: float, 
                       state: AppState, scale: int):
    """Draw a temperature marker - thin crosshair."""
    sx, sy = marker.x * scale, marker.y * scale
    color_rgb = bgr_to_rgb(marker.color)
    
    # Thin crosshair only
    r = 10
    draw.line([sx - r, sy, sx + r, sy], fill=color_rgb, width=1)
    draw.line([sx, sy - r, sx, sy + r], fill=color_rgb, width=1)
    
    # Label
    label = f"#{marker.id}: {format_temp(temp, state.use_fahrenheit)}"
    draw_text_with_bg(draw, (sx + 12, sy - 10), label, 'small', marker.color, (0, 0, 0), 2)

def draw_auto_markers_on_pil(draw: ImageDraw.ImageDraw, display: np.ndarray,
                              temps: np.ndarray, state: AppState, scale: int):
    """Draw automatic min/max markers - thin crosshairs."""
    if state.marker_mode == 3:
        return
    
    max_t, min_t = temps.max(), temps.min()
    max_loc = np.unravel_index(np.argmax(temps), temps.shape)
    min_loc = np.unravel_index(np.argmin(temps), temps.shape)
    
    if state.marker_mode in [0, 1]:
        mx, my = int(max_loc[1] * scale), int(max_loc[0] * scale)
        # Thin + marker for max (red)
        r = 8
        draw.line([mx - r, my, mx + r, my], fill=(255, 0, 0), width=1)
        draw.line([mx, my - r, mx, my + r], fill=(255, 0, 0), width=1)
        draw_text_with_bg(draw, (mx + 10, my - 8), f"MAX:{format_temp(max_t, state.use_fahrenheit)}", 
                         'small', (0, 0, 255), (0, 0, 0), 2)
    
    if state.marker_mode in [0, 2]:
        mx, my = int(min_loc[1] * scale), int(min_loc[0] * scale)
        # Thin + marker for min (cyan)
        r = 8
        draw.line([mx - r, my, mx + r, my], fill=(0, 200, 255), width=1)
        draw.line([mx, my - r, mx, my + r], fill=(0, 200, 255), width=1)
        draw_text_with_bg(draw, (mx + 10, my - 8), f"MIN:{format_temp(min_t, state.use_fahrenheit)}",
                         'small', (255, 200, 0), (0, 0, 0), 2)

def draw_color_scale(display: np.ndarray, state: AppState, x_offset: int):
    """Draw temperature color scale bar (OpenCV for gradient)."""
    bar_x = x_offset - 60
    bar_h = display.shape[0] - 50
    bar_top = 25
    bar_w = 20
    
    # Draw gradient
    for i in range(bar_h):
        val = int(255 * (1 - i / bar_h))
        c = apply_colormap(np.array([[val]], dtype=np.uint8), state.colormap_idx)[0, 0]
        cv2.line(display, (bar_x, bar_top + i), (bar_x + bar_w, bar_top + i), tuple(map(int, c)), 1)
    
    # Border
    cv2.rectangle(display, (bar_x, bar_top), (bar_x + bar_w, bar_top + bar_h), (180, 180, 180), 1)
    
    return bar_x, bar_top, bar_h  # Return for PIL text overlay

def draw_color_scale_labels(draw: ImageDraw.ImageDraw, state: AppState, bar_x: int, bar_top: int, bar_h: int):
    """Draw scale labels with PIL."""
    draw_text(draw, (bar_x - 48, bar_top - 2), format_temp(state.temp_max, state.use_fahrenheit), 'small', (255, 255, 255))
    mid = (state.temp_max + state.temp_min) / 2
    draw_text(draw, (bar_x - 48, bar_top + bar_h // 2 - 6), format_temp(mid, state.use_fahrenheit), 'tiny', (160, 160, 160))
    draw_text(draw, (bar_x - 48, bar_top + bar_h - 12), format_temp(state.temp_min, state.use_fahrenheit), 'small', (255, 255, 255))

def draw_info_panel(display: np.ndarray, temps: np.ndarray, state: AppState):
    """Draw info panel background (OpenCV), return coords for PIL text."""
    max_t, min_t, avg_t = temps.max(), temps.min(), temps.mean()
    
    h = 125 + (22 if state.recording else 0)
    cv2.rectangle(display, (8, 8), (180, h), (0, 0, 0), -1)
    cv2.rectangle(display, (8, 8), (180, h), (60, 60, 60), 1)
    
    return max_t, min_t, avg_t, h

def draw_info_panel_text(draw: ImageDraw.ImageDraw, max_t: float, min_t: float, avg_t: float, state: AppState):
    """Draw info panel text with PIL."""
    y = 16
    draw_text(draw, (14, y), f"Max: {format_temp(max_t, state.use_fahrenheit)}", 'medium', (0, 100, 255))
    y += 22
    draw_text(draw, (14, y), f"Min: {format_temp(min_t, state.use_fahrenheit)}", 'medium', (100, 255, 100))
    y += 22
    draw_text(draw, (14, y), f"Avg: {format_temp(avg_t, state.use_fahrenheit)}", 'medium', (255, 255, 255))
    y += 20
    
    _, name = COLORMAPS[state.colormap_idx]
    mode = "Auto" if state.auto_scale else "Manual"
    draw_text(draw, (14, y), f"[{name}] {mode}", 'small', (140, 140, 140))
    y += 18
    draw_text(draw, (14, y), f"FPS: {state.fps:.0f}  Markers: {len(state.markers)}", 'tiny', (100, 100, 100))
    
    if state.recording:
        y += 22
        elapsed = time.time() - state.record_start_time
        draw_text(draw, (14, y), f"● REC {elapsed:.1f}s", 'medium', (0, 0, 255))

def draw_histogram(display: np.ndarray, temps: np.ndarray, state: AppState, x_offset: int):
    """Draw temperature histogram."""
    if not state.show_histogram:
        return None
    
    hw, hh = 150, 80
    hx, hy = x_offset - hw - 70, display.shape[0] - hh - 20
    
    cv2.rectangle(display, (hx, hy), (hx + hw, hy + hh), (0, 0, 0), -1)
    cv2.rectangle(display, (hx, hy), (hx + hw, hy + hh), (60, 60, 60), 1)
    
    hist, _ = np.histogram(temps.flatten(), bins=35, range=(state.temp_min, state.temp_max))
    hist = hist.astype(np.float32)
    if hist.max() > 0:
        hist = hist / hist.max() * (hh - 18)
    
    bw = hw // len(hist)
    for i, h in enumerate(hist):
        x1 = hx + i * bw
        y1 = hy + hh - int(h) - 5
        y2 = hy + hh - 5
        c = apply_colormap(np.array([[int(255 * i / len(hist))]], dtype=np.uint8), state.colormap_idx)[0, 0]
        cv2.rectangle(display, (x1, y1), (x1 + bw - 1, y2), tuple(map(int, c)), -1)
    
    return hx, hy

def draw_isotherm(display: np.ndarray, temps: np.ndarray, state: AppState, scale: int):
    """Draw isotherm overlay."""
    if not state.isotherm_enabled:
        return
    
    mask = ((temps >= state.isotherm_low) & (temps <= state.isotherm_high)).astype(np.uint8) * 255
    mask_scaled = cv2.resize(mask, (WIDTH * scale, THERMAL_HEIGHT * scale), interpolation=cv2.INTER_NEAREST)
    
    contours, _ = cv2.findContours(mask_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pulse = int(127 + 127 * np.sin(time.time() * 4))
    cv2.drawContours(display, contours, -1, (0, pulse, 255), 2, cv2.LINE_AA)

def draw_side_panel(display: np.ndarray, state: AppState, panel_x: int):
    """Draw side panel with all controls info."""
    h = display.shape[0]
    
    # Background
    cv2.rectangle(display, (panel_x, 0), (panel_x + PANEL_WIDTH, h), (25, 25, 25), -1)
    cv2.line(display, (panel_x, 0), (panel_x, h), (60, 60, 60), 1)
    
    # Convert to PIL for text
    pil_img = cv2_to_pil(display)
    draw = ImageDraw.Draw(pil_img)
    
    x = panel_x + 12
    y = 18
    lh = 22
    
    # Title
    draw_text(draw, (x, y), "THERMAL VIEWER", 'title', (0, 255, 255))
    y += lh + 14
    
    # Mouse section
    draw_text(draw, (x, y), "MOUSE:", 'medium', (80, 180, 255))
    y += lh
    mouse_help = [
        ("Click", "Add marker"),
        ("Shift+Click", "Remove marker"),
        ("Ctrl+Click", "Remove marker"),
        ("Mid-Click", "Clear all"),
        ("Drag", "Move marker"),
    ]
    for key, desc in mouse_help:
        draw_text(draw, (x, y), key, 'small', (200, 200, 200))
        draw_text(draw, (x + 95, y), desc, 'small', (140, 140, 140))
        y += lh - 4
    
    y += 10
    
    # Keyboard section
    draw_text(draw, (x, y), "KEYBOARD:", 'medium', (80, 180, 255))
    y += lh
    kb_help = [
        ("q / ESC", "Quit"),
        ("s", "Screenshot"),
        ("r", "Record"),
        ("Del / x", "Remove markers"),
        ("c / 1-6", "Colormap"),
        ("a", "Auto scale"),
        ("+/-", "Max temp"),
        ("[/]", "Min temp"),
        ("0", "Reset scale"),
        ("m", "Marker mode"),
        ("i", "Isotherm"),
        ("Arrows", "Iso range"),
        ("f", "Freeze"),
        ("h", "Histogram"),
        ("t", "°C / °F"),
        ("p", "Panel"),
    ]
    for key, desc in kb_help:
        draw_text(draw, (x, y), key, 'small', (200, 200, 200))
        draw_text(draw, (x + 80, y), desc, 'small', (140, 140, 140))
        y += lh - 5
    
    y += 10
    
    # Settings section
    draw_text(draw, (x, y), "SETTINGS:", 'medium', (80, 180, 255))
    y += lh
    
    _, cmap = COLORMAPS[state.colormap_idx]
    scale_s = "Auto" if state.auto_scale else f"{state.temp_min:.0f}–{state.temp_max:.0f}"
    iso_s = f"{state.isotherm_low:.0f}–{state.isotherm_high:.0f}" if state.isotherm_enabled else "Off"
    
    settings = [
        ("Map:", cmap),
        ("Scale:", scale_s),
        ("Markers:", f"{len(state.markers)} ({state.marker_modes[state.marker_mode]})"),
        ("Unit:", "°F" if state.use_fahrenheit else "°C"),
        ("Iso:", iso_s),
    ]
    for name, val in settings:
        draw_text(draw, (x, y), name, 'small', (140, 140, 140))
        draw_text(draw, (x + 70, y), str(val), 'small', (255, 255, 255))
        y += lh - 4
    
    # Status at bottom
    if state.frozen:
        draw_text(draw, (x, h - 50), "★ FROZEN ★", 'large', (0, 255, 255))
    
    if state.recording:
        elapsed = time.time() - state.record_start_time
        draw_text(draw, (x, h - 25), f"● REC {elapsed:.1f}s", 'large', (0, 0, 255))
    
    # Convert back to OpenCV
    result = pil_to_cv2(pil_img)
    np.copyto(display, result)

# =============================================================================
# Mouse callback
# =============================================================================
def mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
    global g_state
    if g_state is None:
        return
    
    state = g_state
    scale = state.display_scale
    
    # Ignore clicks outside thermal area
    if state.show_panel and x >= state.thermal_display_width:
        return
    
    orig_x = x // scale
    orig_y = y // scale
    
    if not (0 <= orig_x < WIDTH and 0 <= orig_y < THERMAL_HEIGHT):
        return
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Shift+Click or Ctrl+Click = remove nearest
        if (flags & cv2.EVENT_FLAG_SHIFTKEY) or (flags & cv2.EVENT_FLAG_CTRLKEY):
            nearest = find_nearest_marker(state.markers, orig_x, orig_y, threshold=30)
            if nearest:
                state.markers.remove(nearest)
                print(f"Removed marker #{nearest.id}")
            return
        
        # Near existing marker = start drag
        nearest = find_nearest_marker(state.markers, orig_x, orig_y, threshold=15)
        if nearest:
            state.dragging_marker = nearest
            return
        
        # Add new marker
        state.marker_counter += 1
        color = marker_color(state.marker_counter)
        m = Marker(x=orig_x, y=orig_y, id=state.marker_counter, color=color)
        state.markers.append(m)
        print(f"Added marker #{m.id} at ({orig_x}, {orig_y})")
    
    elif event == cv2.EVENT_LBUTTONUP:
        state.dragging_marker = None
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if state.dragging_marker is not None:
            state.dragging_marker.x = max(0, min(WIDTH - 1, orig_x))
            state.dragging_marker.y = max(0, min(THERMAL_HEIGHT - 1, orig_y))
    
    elif event == cv2.EVENT_MBUTTONDOWN:
        n = len(state.markers)
        state.markers.clear()
        state.marker_counter = 0
        print(f"Cleared {n} markers")

# =============================================================================
# Recording
# =============================================================================
def start_recording(state: AppState, size: Tuple[int, int]):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fn = f'thermal_rec_{ts}.mp4'
    state.video_writer = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, size)
    state.recording = True
    state.record_start_time = time.time()
    print(f"Recording: {fn}")

def stop_recording(state: AppState):
    if state.video_writer:
        state.video_writer.release()
        state.video_writer = None
    state.recording = False
    print("Recording stopped")

# =============================================================================
# Main
# =============================================================================
def main():
    global g_state
    
    print(f"Opening {DEVICE}...")
    print(f"Resolution: {WIDTH}x{HEIGHT} (thermal: {WIDTH}x{THERMAL_HEIGHT})")
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"Error: Cannot open {DEVICE}")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print(f"Capture: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"Font: {FONT_PATH if FONT_PATH else 'default (no TTF found)'}")
    
    state = AppState()
    g_state = state
    
    thermal_w = WIDTH * state.display_scale
    thermal_h = THERMAL_HEIGHT * state.display_scale
    state.thermal_display_width = thermal_w
    
    win = 'Mega-Idea Thermal Camera'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(win, thermal_w + PANEL_WIDTH, thermal_h)
    cv2.setMouseCallback(win, mouse_callback, None)
    
    print("=" * 50)
    print("Started! Click to add markers, Shift+Click to remove")
    print("Press 'q' to quit, 'p' to toggle panel")
    print("=" * 50)
    
    temps = np.zeros((THERMAL_HEIGHT, WIDTH), dtype=np.float32)
    frame_count = 0
    fps_time = time.time()
    
    while True:
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except:
            break
        
        # Read frame
        if not state.frozen:
            ret, frame = cap.read()
            if not ret:
                continue
            
            if len(frame.shape) == 2 and frame.shape[1] == WIDTH * 2:
                frame = frame.reshape(HEIGHT, WIDTH, 2)
            
            if frame.shape[0] != HEIGHT or len(frame.shape) < 3:
                continue
            
            lo = frame[THERMAL_HEIGHT:, :, 0].astype(np.uint16)
            hi = frame[THERMAL_HEIGHT:, :, 1].astype(np.uint16)
            temps = raw_to_celsius(lo | (hi << 8))
            state.frozen_temps = temps.copy()
        else:
            if state.frozen_temps is not None:
                temps = state.frozen_temps
        
        # Auto scale
        if state.auto_scale:
            state.temp_min = float(temps.min() - state.auto_scale_margin)
            state.temp_max = float(temps.max() + state.auto_scale_margin)
        
        # Normalize & colorize
        rng = max(state.temp_max - state.temp_min, 1.0)
        norm = np.clip((temps - state.temp_min) / rng, 0, 1)
        gray = (norm * 255).astype(np.uint8)
        colored = apply_colormap(gray, state.colormap_idx)
        
        # Scale
        thermal = cv2.resize(colored, (thermal_w, thermal_h), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas
        if state.show_panel:
            display = np.zeros((thermal_h, thermal_w + PANEL_WIDTH, 3), dtype=np.uint8)
            display[:, :thermal_w] = thermal
            draw_side_panel(display, state, thermal_w)
        else:
            display = thermal.copy()
        
        # Draw OpenCV elements first
        draw_isotherm(display, temps, state, state.display_scale)
        bar_info = draw_color_scale(display, state, thermal_w)
        max_t, min_t, avg_t, info_h = draw_info_panel(display, temps, state)
        hist_pos = draw_histogram(display, temps, state, thermal_w)
        
        # Convert to PIL for text and markers
        pil_img = cv2_to_pil(display)
        draw = ImageDraw.Draw(pil_img)
        
        # Draw PIL text elements
        draw_color_scale_labels(draw, state, *bar_info)
        draw_info_panel_text(draw, max_t, min_t, avg_t, state)
        
        if hist_pos and state.show_histogram:
            draw_text(draw, (hist_pos[0] + 5, hist_pos[1] + 4), "Histogram", 'tiny', (180, 180, 180))
        
        # Draw markers
        draw_auto_markers_on_pil(draw, display, temps, state, state.display_scale)
        for m in state.markers:
            if 0 <= m.y < temps.shape[0] and 0 <= m.x < temps.shape[1]:
                draw_marker_on_pil(draw, m, temps[m.y, m.x], state, state.display_scale)
        
        # Isotherm label
        if state.isotherm_enabled:
            label = f"ISO: {format_temp(state.isotherm_low, state.use_fahrenheit)} – {format_temp(state.isotherm_high, state.use_fahrenheit)}"
            draw_text_with_bg(draw, (12, thermal_h - 28), label, 'medium', (0, 200, 255), (0, 0, 0), 3)
        
        # Frozen indicator
        if state.frozen:
            draw_text_with_bg(draw, (thermal_w // 2 - 50, 8), "★ FROZEN ★", 'large', (0, 255, 255), (0, 0, 0), 4)
        
        # Convert back
        display = pil_to_cv2(pil_img)
        
        # Record
        if state.recording and state.video_writer:
            rec_frame = display[:, :thermal_w] if state.show_panel else display
            state.video_writer.write(rec_frame)
        
        # FPS
        frame_count += 1
        if frame_count % 10 == 0:
            now = time.time()
            if now - fps_time > 0:
                state.fps = 10.0 / (now - fps_time)
            fps_time = now
        
        cv2.imshow(win, display)
        
        # Keys - FIXED: check for -1 (no key) before processing
        key = cv2.waitKey(1)
        if key == -1:
            continue  # No key pressed
        
        key = key & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('p'):
            state.show_panel = not state.show_panel
        elif key == ord('c'):
            state.colormap_idx = (state.colormap_idx + 1) % len(COLORMAPS)
        elif ord('1') <= key <= ord('6'):
            idx = key - ord('1')
            if idx < len(COLORMAPS):
                state.colormap_idx = idx
        elif key == ord('a'):
            state.auto_scale = not state.auto_scale
            print(f"Auto-scale: {'ON' if state.auto_scale else 'OFF'}")
        elif key == ord('m'):
            state.marker_mode = (state.marker_mode + 1) % len(state.marker_modes)
        elif key == ord('i'):
            state.isotherm_enabled = not state.isotherm_enabled
            if state.isotherm_enabled:
                mid = (state.temp_max + state.temp_min) / 2
                state.isotherm_low = mid - 10
                state.isotherm_high = mid + 10
        elif key == ord('f'):
            state.frozen = not state.frozen
            print(f"Freeze: {'ON' if state.frozen else 'OFF'}")
        elif key == ord('h'):
            state.show_histogram = not state.show_histogram
        elif key == ord('t'):
            state.use_fahrenheit = not state.use_fahrenheit
        elif key in [ord('+'), ord('=')]:
            state.auto_scale = False
            state.temp_max = min(300, state.temp_max + 5)
        elif key == ord('-'):
            state.auto_scale = False
            state.temp_max = max(state.temp_min + 10, state.temp_max - 5)
        elif key == ord('['):
            state.auto_scale = False
            state.temp_min = max(-50, state.temp_min - 5)
        elif key == ord(']'):
            state.auto_scale = False
            state.temp_min = min(state.temp_max - 10, state.temp_min + 5)
        elif key == ord('0'):
            state.auto_scale = True
        elif key == 82:  # Up
            if state.isotherm_enabled:
                state.isotherm_high += 2
                state.isotherm_low += 2
        elif key == 84:  # Down
            if state.isotherm_enabled:
                state.isotherm_high -= 2
                state.isotherm_low -= 2
        elif key == 83:  # Right
            if state.isotherm_enabled:
                state.isotherm_high += 2
        elif key == 81:  # Left
            if state.isotherm_enabled:
                state.isotherm_low -= 2
        elif key == ord('s'):
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            img = display[:, :thermal_w] if state.show_panel else display
            cv2.imwrite(f'thermal_{ts}.png', img)
            np.save(f'thermal_{ts}_temps.npy', temps)
            print(f"Saved: thermal_{ts}.png")
        elif key == ord('r'):
            if state.recording:
                stop_recording(state)
            else:
                start_recording(state, (thermal_w, thermal_h))
        elif key in [8, 127]:  # Backspace, Delete (NOT 255!)
            if state.markers:
                rm = state.markers.pop()
                print(f"Removed marker #{rm.id}")
        elif key == ord('x'):
            n = len(state.markers)
            state.markers.clear()
            state.marker_counter = 0
            print(f"Cleared {n} markers")
    
    if state.recording:
        stop_recording(state)
    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")

if __name__ == "__main__":
    main()
