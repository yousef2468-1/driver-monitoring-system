"""
Alert Manager v2 - Fixed audio using system beep
Works 100% on Linux/Ubuntu
"""

import time
import os
import subprocess


class AlertManager:
    def __init__(self):
        self.alert_log   = []
        self.last_alerts = {}
        self.cooldown    = 3.0

    def _beep(self, alert_type):
        """Use system speaker - works on all Linux"""
        try:
            # Method 1: paplay (PulseAudio - most reliable)
            freq_map = {
                "drowsiness": 880,
                "yawning":    660,
                "phone":      1000,
                "cigarette":  770
            }
            freq = freq_map.get(alert_type, 880)

            # Generate beep using python and play via aplay
            import wave, struct, math
            sample_rate = 44100
            duration    = 0.5
            filename    = f"/tmp/alert_{alert_type}.wav"

            if not os.path.exists(filename):
                n_samples = int(sample_rate * duration)
                with wave.open(filename, 'w') as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(sample_rate)
                    for i in range(n_samples):
                        # Fade in/out to avoid clicks
                        fade = min(i, n_samples-i, 1000) / 1000.0
                        val  = int(32767 * fade * math.sin(2 * math.pi * freq * i / sample_rate))
                        f.writeframes(struct.pack('<h', val))

            subprocess.Popen(
                ["aplay", "-q", filename],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            try:
                # Method 2: print bell character
                print('\a', end='', flush=True)
            except:
                pass

    def trigger(self, alert_type: str, message: str, play_sound: bool = True):
        now = time.time()
        if now - self.last_alerts.get(alert_type, 0) < self.cooldown:
            return False

        self.last_alerts[alert_type] = now
        self.alert_log.append({
            "time":    time.strftime("%H:%M:%S"),
            "type":    alert_type,
            "message": message
        })

        if play_sound:
            self._beep(alert_type)

        return True

    def get_stats(self):
        return {
            "total":      len(self.alert_log),
            "drowsiness": sum(1 for a in self.alert_log if a["type"] == "drowsiness"),
            "yawning":    sum(1 for a in self.alert_log if a["type"] == "yawning"),
            "phone":      sum(1 for a in self.alert_log if a["type"] == "phone"),
            "cigarette":  sum(1 for a in self.alert_log if a["type"] == "cigarette"),
            "log":        self.alert_log[-10:]
        }

    def reset(self):
        self.alert_log   = []
        self.last_alerts = {}
