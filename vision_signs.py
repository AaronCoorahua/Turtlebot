"""
B&W sign detection using ORB template matching.

Two-stage pipeline:
  1. HoughCircles  — finds the circular sign border in the frame
  2. ORB matching  — classifies the cropped ROI against known templates

Templates are loaded from fotos/ at startup. Both stop variants are used
to improve robustness against lighting/angle variation.

If HoughCircles finds no circle (sign too close or poor contrast),
the center 60% of the frame is used as fallback ROI.
"""
import cv2
import numpy as np

_CLASSES: dict[str, list[str]] = {
    "DER":  ["fotos/der.jpeg"],
    "IZQ":  ["fotos/izq.jpeg"],
    "STOP": ["fotos/stop1.jpeg", "fotos/stop2.jpeg"],
}

_ROI_SIZE   = (200, 200)
_DETECT_W   = 640   # resize frame to this width before HoughCircles


class SignDetector:
    def __init__(self, cfg: dict):
        self._orb     = cv2.ORB_create(nfeatures=500)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._min_matches = cfg.get("orb_min_matches", 15)
        self._templates   = self._load_templates()

    def _load_templates(self) -> dict[str, list[np.ndarray]]:
        templates: dict[str, list[np.ndarray]] = {}
        for cls, paths in _CLASSES.items():
            descs = []
            for p in paths:
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, _ROI_SIZE)
                _, d = self._orb.detectAndCompute(img, None)
                if d is not None:
                    descs.append(d)
            templates[cls] = descs
        return templates

    def _find_circle_roi(self, gray: np.ndarray) -> np.ndarray:
        """
        Returns a 200×200 crop of the most prominent circle found.
        Falls back to the center 60% of the frame if no circle is detected.
        """
        h_orig, w_orig = gray.shape
        scale = _DETECT_W / w_orig
        small = cv2.resize(gray, (_DETECT_W, int(h_orig * scale)))
        h_s, w_s = small.shape

        blurred = cv2.GaussianBlur(small, (9, 9), 2)
        min_r   = max(20, int(w_s * 0.05))
        max_r   = int(w_s * 0.48)

        circles = None
        for p2 in (30, 20, 15, 10):
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                minDist=int(w_s * 0.1),
                param1=80, param2=p2,
                minRadius=min_r, maxRadius=max_r,
            )
            if circles is not None:
                break

        if circles is not None:
            cx, cy, r = (v / scale for v in map(float, circles[0][0]))
            cx, cy, r = int(cx), int(cy), int(r)
            x1 = max(0, cx - r)
            y1 = max(0, cy - r)
            x2 = min(w_orig, cx + r)
            y2 = min(h_orig, cy + r)
            roi = gray[y1:y2, x1:x2]
        else:
            # Fallback: center 60% crop
            m = 0.20
            y1 = int(h_orig * m); y2 = int(h_orig * (1 - m))
            x1 = int(w_orig * m); x2 = int(w_orig * (1 - m))
            roi = gray[y1:y2, x1:x2]

        if roi.size == 0:
            roi = gray
        return cv2.resize(roi, _ROI_SIZE)

    def detect(self, frame: np.ndarray) -> str:
        """Returns 'STOP', 'DER', 'IZQ', or 'NONE'."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi  = self._find_circle_roi(gray)

        _, desc_roi = self._orb.detectAndCompute(roi, None)
        if desc_roi is None:
            return "NONE"

        best_cls   = "NONE"
        best_count = self._min_matches - 1
        for cls, descs in self._templates.items():
            for desc_tmpl in descs:
                raw  = self._matcher.knnMatch(desc_tmpl, desc_roi, k=2)
                good = [m for m, n in raw if m.distance < 0.75 * n.distance]
                if len(good) > best_count:
                    best_count = len(good)
                    best_cls   = cls
        return best_cls
