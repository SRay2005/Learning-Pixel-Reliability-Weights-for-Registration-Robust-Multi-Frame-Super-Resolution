"""
Download and prepare Football and Bus sequences from Xiph.org for
qualitative SR evaluation.

Downloads CIF-resolution Y4M files, extracts frames as grayscale PNGs.
These are standard test sequences used in the SR literature.

Usage:
    python experiments/download_sequences.py
"""

import os
import sys
import urllib.request
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Xiph.org Derf's collection (CIF = 352x288)
SEQUENCES = {
    'bus': 'https://media.xiph.org/video/derf/y4m/bus_cif.y4m',
    'football': 'https://media.xiph.org/video/derf/y4m/football_cif.y4m',
}


def download_file(url, dest):
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        print(f"    Already exists: {dest}")
        return

    print(f"    Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"    Done ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"    Failed: {e}")
        # Try alternate URL format
        alt_url = url.replace('y4m/', '')
        try:
            print(f"    Trying alternate: {alt_url}")
            urllib.request.urlretrieve(alt_url, dest)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"    Done ({size_mb:.1f} MB)")
        except Exception as e2:
            print(f"    Also failed: {e2}")
            print(f"    Please download manually from https://media.xiph.org/video/derf/")


def parse_y4m_frames(y4m_path, output_dir, max_frames=30):
    """Parse Y4M file and extract frames as grayscale PNGs.

    Y4M format: header line + frame headers + raw YCbCr data.
    We extract only the Y (luminance) channel.
    """
    import numpy as np
    import cv2

    os.makedirs(output_dir, exist_ok=True)

    with open(y4m_path, 'rb') as f:
        # Read header line
        header = b''
        while True:
            c = f.read(1)
            if c == b'\n':
                break
            header += c

        header_str = header.decode('ascii')
        print(f"    Y4M header: {header_str}")

        # Parse width and height from header
        # Format: YUV4MPEG2 W352 H288 F30:1 Ip A0:0 C420jpeg
        parts = header_str.split()
        width = height = 0
        for p in parts:
            if p.startswith('W'):
                width = int(p[1:])
            elif p.startswith('H'):
                height = int(p[1:])

        if width == 0 or height == 0:
            print(f"    ERROR: Could not parse dimensions from header")
            return []

        print(f"    Frame size: {width}x{height}")

        # Determine chroma subsampling (assume 4:2:0 if not specified)
        chroma_format = '420'
        for p in parts:
            if p.startswith('C'):
                chroma_format = p[1:].split()[0] if len(p) > 1 else '420'

        # Frame size in bytes (Y plane only for 4:2:0)
        y_size = width * height
        if '420' in chroma_format:
            uv_size = (width // 2) * (height // 2) * 2
        elif '422' in chroma_format:
            uv_size = (width // 2) * height * 2
        else:
            uv_size = (width // 2) * (height // 2) * 2  # default 420

        frame_paths = []
        frame_idx = 0

        while frame_idx < max_frames:
            # Read frame header "FRAME\n" (possibly with parameters)
            frame_header = b''
            while True:
                c = f.read(1)
                if c == b'' or c == b'\n':
                    break
                frame_header += c

            if not frame_header.startswith(b'FRAME'):
                break  # End of file or invalid

            # Read Y plane
            y_data = f.read(y_size)
            if len(y_data) < y_size:
                break

            # Skip UV planes
            f.read(uv_size)

            # Convert to numpy and save
            y_frame = np.frombuffer(y_data, dtype=np.uint8).reshape(height, width)

            frame_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
            cv2.imwrite(frame_path, y_frame)
            frame_paths.append(frame_path)
            frame_idx += 1

        print(f"    Extracted {frame_idx} frames")
        return frame_paths


def main():
    print("\n=== Downloading Test Video Sequences ===")

    seq_dir = os.path.join(PROJECT_ROOT, 'data', 'sequences')
    os.makedirs(seq_dir, exist_ok=True)

    for name, url in SEQUENCES.items():
        print(f"\n  Sequence: {name}")

        y4m_path = os.path.join(seq_dir, f'{name}_cif.y4m')
        download_file(url, y4m_path)

        if os.path.exists(y4m_path):
            frames_dir = os.path.join(seq_dir, name)
            frames = parse_y4m_frames(y4m_path, frames_dir, max_frames=20)
            if frames:
                print(f"    Frames saved to {frames_dir}/")

    print("\n  [DONE]")


if __name__ == '__main__':
    main()
