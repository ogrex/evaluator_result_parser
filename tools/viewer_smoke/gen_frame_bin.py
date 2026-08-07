"""Generate a synthetic T4V3D002 frame.bin fixture for browser smoke tests.

Usage: .venv/bin/python tools/viewer_smoke/gen_frame_bin.py
Writes tools/viewer_smoke/frame.bin (gitignored; regenerate at will).
"""
from pathlib import Path

import numpy as np

from t4_visualizer.server import pack_viewer_frame_binary


class _Box:
    label = "car"

    def corners(self):
        return np.array(
            [
                [0.0, 4.0, 4.0, 0.0, 0.0, 4.0, 4.0, 0.0],
                [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.6, 1.6, 1.6, 1.6],
            ],
            dtype=np.float32,
        )


def main() -> None:
    rng = np.random.default_rng(42)
    pts = rng.random((500, 4), dtype=np.float32)
    pts[:, :3] *= 20
    pts[:, 3] = np.linspace(0.0, 1.0, pts.shape[0], dtype=np.float32)
    blob = pack_viewer_frame_binary(
        frame_index=0,
        sample_token="smoke_token",
        timestamp_us=1,
        points_xyz_i=pts,
        boxes_3d=[_Box()],
    )
    out = Path(__file__).parent / "frame.bin"
    out.write_bytes(blob)
    print(f"wrote {out} ({len(blob)} bytes)")


if __name__ == "__main__":
    main()
