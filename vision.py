"""
Vision module: traffic light detection, arrow direction, QR checkpoint reading.
All thresholds come from config.yaml — never hardcode here.
"""
import cv2
import numpy as np
from enum import Enum, auto
from typing import Optional

try:
    from pyzbar import pyzbar
    _PYZBAR_AVAILABLE = True
except ImportError:
    _PYZBAR_AVAILABLE = False


class TrafficState(Enum):
    UNKNOWN = auto()
    RED     = auto()
    GREEN   = auto()


class ArrowDir(Enum):
    UNKNOWN  = auto()
    LEFT     = auto()
    RIGHT    = auto()
    STRAIGHT = auto()


class VisionResult:
    __slots__ = ("traffic", "arrow", "qr_data", "debug_frame")

    def __init__(self):
        self.traffic: TrafficState = TrafficState.UNKNOWN
        self.arrow: ArrowDir = ArrowDir.UNKNOWN
        self.qr_data: Optional[str] = None
        self.debug_frame: Optional[np.ndarray] = None


class VisionProcessor:
    def __init__(self, cfg: dict):
        tl = cfg["traffic_light"]
        self._tl_min_area = tl["min_contour_area"]
        self._red_lower1  = np.array(tl["hsv_red_lower1"], dtype=np.uint8)
        self._red_upper1  = np.array(tl["hsv_red_upper1"], dtype=np.uint8)
        self._red_lower2  = np.array(tl["hsv_red_lower2"], dtype=np.uint8)
        self._red_upper2  = np.array(tl["hsv_red_upper2"], dtype=np.uint8)
        self._grn_lower   = np.array(tl["hsv_green_lower"], dtype=np.uint8)
        self._grn_upper   = np.array(tl["hsv_green_upper"], dtype=np.uint8)

        ar = cfg["arrow"]
        self._ar_min_area     = ar["min_contour_area"]
        self._ar_split_thresh = ar["split_threshold"]
        self._ar_lower = np.array(ar["hsv_lower"], dtype=np.uint8)
        self._ar_upper = np.array(ar["hsv_upper"], dtype=np.uint8)

        self._qr_detector   = cv2.QRCodeDetector()
        self._max_checkpoints = cfg["qr"]["max_checkpoints"]
        self.checkpoints: list[str] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process(self, frame: np.ndarray, debug: bool = False) -> VisionResult:
        result = VisionResult()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        result.traffic = self._detect_traffic_light(hsv)
        result.arrow   = self._detect_arrow(frame)
        result.qr_data = self._detect_qr(frame)

        if debug:
            result.debug_frame = self._draw_debug(frame.copy(), result)

        return result

    @property
    def checkpoint_count(self) -> int:
        return len(self.checkpoints)

    # ------------------------------------------------------------------ #
    #  Traffic light                                                       #
    # ------------------------------------------------------------------ #

    def _detect_traffic_light(self, hsv: np.ndarray) -> TrafficState:
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, self._red_lower1, self._red_upper1),
            cv2.inRange(hsv, self._red_lower2, self._red_upper2),
        )
        grn_mask = cv2.inRange(hsv, self._grn_lower, self._grn_upper)

        red_area = self._largest_contour_area(red_mask)
        grn_area = self._largest_contour_area(grn_mask)

        if red_area < self._tl_min_area and grn_area < self._tl_min_area:
            return TrafficState.UNKNOWN
        return TrafficState.RED if red_area >= grn_area else TrafficState.GREEN

    @staticmethod
    def _largest_contour_area(mask: np.ndarray) -> float:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        return max(cv2.contourArea(c) for c in contours)

    # ------------------------------------------------------------------ #
    #  Arrow detection                                                     #
    # ------------------------------------------------------------------ #

    def _detect_arrow(self, frame: np.ndarray) -> ArrowDir:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._ar_lower, self._ar_upper)

        # Limpiar ruido y cerrar huecos dentro del blob de la flecha
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        candidates = [c for c in contours if cv2.contourArea(c) > self._ar_min_area]
        if not candidates:
            return ArrowDir.UNKNOWN

        cnt = max(candidates, key=cv2.contourArea)
        return self._classify_arrow_defects(cnt)

    def _classify_arrow_defects(self, cnt: np.ndarray) -> ArrowDir:
        """
        Primary method: convex hull defects.
        The deepest defect is the V-notch at the tail of the arrow.
        The tail is on the OPPOSITE side of where the arrow points.

        Fallback: pixel-distribution comparison (left vs right half of bbox).
        """
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        if hull_idx is not None and len(hull_idx) >= 3:
            defects = cv2.convexityDefects(cnt, hull_idx)
            if defects is not None and len(defects) > 0:
                # Deepest defect → tail notch
                deepest = max(defects, key=lambda d: d[0][3])
                far_pt_x = cnt[deepest[0][2]][0][0]
                x, _, w, _ = cv2.boundingRect(cnt)
                box_cx = x + w / 2.0
                offset = far_pt_x - box_cx

                if abs(offset) > w * 0.10:
                    # Notch on right → tail right → arrow points LEFT, and vice versa
                    return ArrowDir.LEFT if offset > 0 else ArrowDir.RIGHT

        # Fallback: compare pixel mass in left vs right half of the bounding box
        return self._classify_arrow_pixel_split(cnt)

    def _classify_arrow_pixel_split(self, cnt: np.ndarray) -> ArrowDir:
        x, y, w, h = cv2.boundingRect(cnt)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt - [x, y]], -1, 255, cv2.FILLED)

        left  = int(mask[:, : w // 2].sum())
        right = int(mask[:, w // 2 :].sum())
        total = left + right
        if total == 0:
            return ArrowDir.UNKNOWN

        ratio = (right - left) / total   # >0 → more pixels right → points RIGHT
        if ratio >  self._ar_split_thresh:
            return ArrowDir.RIGHT
        if ratio < -self._ar_split_thresh:
            return ArrowDir.LEFT
        return ArrowDir.STRAIGHT

    # ------------------------------------------------------------------ #
    #  QR checkpoints                                                      #
    # ------------------------------------------------------------------ #

    def _detect_qr(self, frame: np.ndarray) -> Optional[str]:
        if len(self.checkpoints) >= self._max_checkpoints:
            return None

        # Try pyzbar first (more robust), fall back to OpenCV
        if _PYZBAR_AVAILABLE:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            codes = pyzbar.decode(gray)
            for code in codes:
                data = code.data.decode("utf-8", errors="ignore").strip()
                return self._register_checkpoint(data)
        else:
            data, _, _ = self._qr_detector.detectAndDecode(frame)
            if data:
                return self._register_checkpoint(data.strip())
        return None

    def _register_checkpoint(self, data: str) -> Optional[str]:
        if data and data not in self.checkpoints:
            self.checkpoints.append(data)
            return data
        return None

    # ------------------------------------------------------------------ #
    #  Debug overlay                                                       #
    # ------------------------------------------------------------------ #

    def _draw_debug(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        color_map = {
            TrafficState.RED:     (0, 0, 255),
            TrafficState.GREEN:   (0, 255, 0),
            TrafficState.UNKNOWN: (128, 128, 128),
        }
        tl_color = color_map[result.traffic]
        cv2.putText(frame, f"TL: {result.traffic.name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, tl_color, 2)
        cv2.putText(frame, f"Arrow: {result.arrow.name}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if result.qr_data:
            cv2.putText(frame, f"QR: {result.qr_data}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"CPs: {self.checkpoints}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return frame
