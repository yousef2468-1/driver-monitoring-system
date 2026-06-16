"""
Unit Tests — Driver Monitoring System
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import pytest


# ─────────────────────────────────────────────────────────────────────────────
class TestDrowsinessDetector:

    def setup_method(self):
        from models.drowsiness_detector import DrowsinessDetector
        self.detector = DrowsinessDetector()

    def test_detector_initializes(self):
        assert self.detector is not None
        assert self.detector.frame_counter == 0
        assert self.detector.drowsy == False

    def test_detect_returns_dict(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)   # black frame
        result = self.detector.detect(frame)
        assert isinstance(result, dict)
        assert "drowsy"   in result
        assert "ear"      in result
        assert "message"  in result
        assert "frame"    in result

    def test_detect_on_black_frame(self):
        # Black frame — no face — should not crash
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        assert result["drowsy"] == False
        assert result["ear"] == 0.0

    def test_reset(self):
        self.detector.frame_counter = 30
        self.detector.drowsy = True
        self.detector.reset()
        assert self.detector.frame_counter == 0
        assert self.detector.drowsy == False


# ─────────────────────────────────────────────────────────────────────────────
class TestAlertManager:

    def setup_method(self):
        from models.alert_manager import AlertManager
        self.mgr = AlertManager()

    def test_initializes(self):
        assert self.mgr is not None
        assert len(self.mgr.alert_log) == 0

    def test_trigger_logs_alert(self):
        triggered = self.mgr.trigger("drowsiness", "Test alert", play_sound=False)
        assert triggered == True
        assert len(self.mgr.alert_log) == 1
        assert self.mgr.alert_log[0]["type"] == "drowsiness"

    def test_cooldown_prevents_duplicate(self):
        self.mgr.trigger("phone", "Alert 1", play_sound=False)
        triggered = self.mgr.trigger("phone", "Alert 2", play_sound=False)
        assert triggered == False    # cooldown
        assert len(self.mgr.alert_log) == 1

    def test_different_types_allowed(self):
        self.mgr.trigger("phone",      "Phone alert",    play_sound=False)
        self.mgr.trigger("drowsiness", "Drowsy alert",   play_sound=False)
        self.mgr.trigger("cigarette",  "Smoking alert",  play_sound=False)
        assert len(self.mgr.alert_log) == 3

    def test_get_stats(self):
        self.mgr.trigger("drowsiness", "d", play_sound=False)
        self.mgr.trigger("phone",      "p", play_sound=False)
        stats = self.mgr.get_stats()
        assert stats["drowsiness"] == 1
        assert stats["phone"]      == 1
        assert stats["cigarette"]  == 0
        assert stats["total"]      == 2

    def test_reset(self):
        self.mgr.trigger("phone", "p", play_sound=False)
        self.mgr.reset()
        assert len(self.mgr.alert_log) == 0


# ─────────────────────────────────────────────────────────────────────────────
class TestObjectDetector:

    def setup_method(self):
        from models.object_detector import ObjectDetector
        self.detector = ObjectDetector()    # uses COCO fallback

    def test_initializes(self):
        assert self.detector is not None

    def test_detect_returns_dict(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        assert isinstance(result, dict)
        assert "phone_detected"     in result
        assert "cigarette_detected" in result
        assert "detections"         in result
        assert "frame"              in result
        assert "message"            in result

    def test_detect_on_blank_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        assert result["phone_detected"]     == False
        assert result["cigarette_detected"] == False
        assert result["detections"]         == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
