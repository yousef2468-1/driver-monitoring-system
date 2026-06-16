"""
Driver Safety Score Module
Calculates real-time safety score 0-100
"""

import time


class SafetyScore:
    def __init__(self):
        self.score        = 100.0
        self.start_time   = time.time()
        self.deductions   = []
        self.grade_map    = [
            (90, "A", "Excellent"),
            (75, "B", "Good"),
            (60, "C", "Fair"),
            (40, "D", "Poor"),
            (0,  "F", "Dangerous")
        ]
        # Deduction per alert type
        self.deductions_map = {
            "drowsiness": 5.0,
            "yawning":    2.0,
            "phone":      4.0,
            "cigarette":  3.0
        }

    def deduct(self, alert_type: str):
        amount = self.deductions_map.get(alert_type, 2.0)
        self.score = max(0.0, self.score - amount)
        self.deductions.append({
            "time":   time.strftime("%H:%M:%S"),
            "type":   alert_type,
            "amount": amount,
            "score":  round(self.score, 1)
        })

    def get_grade(self):
        for threshold, grade, label in self.grade_map:
            if self.score >= threshold:
                return grade, label
        return "F", "Dangerous"

    def get_color(self):
        if self.score >= 90: return "00CC66"   # green
        if self.score >= 75: return "88CC00"   # yellow-green
        if self.score >= 60: return "FFAA00"   # orange
        if self.score >= 40: return "FF6600"   # dark orange
        return "FF0000"                         # red

    def get_stats(self):
        grade, label = self.get_grade()
        elapsed      = int(time.time() - self.start_time)
        return {
            "score":    round(self.score, 1),
            "grade":    grade,
            "label":    label,
            "color":    self.get_color(),
            "elapsed":  elapsed,
            "history":  self.deductions[-10:]
        }

    def reset(self):
        self.score      = 100.0
        self.start_time = time.time()
        self.deductions = []
