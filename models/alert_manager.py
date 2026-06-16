"""
Alert Manager
Handles all alerts: visual, audio, logging
"""

import time
import pygame
import numpy as np
import os


class AlertManager:
    def __init__(self):
        self.alert_log   = []          # list of (timestamp, alert_type, message)
        self.last_alerts = {}          # cooldown tracker per alert type
        self.cooldown    = 3.0         # seconds between same alert
        self._init_audio()

    # ── Audio init ────────────────────────────────────────────────────────────
    def _init_audio(self):
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self.audio_ok = True
        except Exception:
            self.audio_ok = False

    # ── Generate beep ─────────────────────────────────────────────────────────
    def _beep(self, freq=880, duration_ms=500):
        if not self.audio_ok:
            return
        try:
            sample_rate = 22050
            n_samples   = int(sample_rate * duration_ms / 1000)
            t           = np.linspace(0, duration_ms / 1000, n_samples, False)
            wave        = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            wave        = np.column_stack([wave, wave])   # stereo
            sound       = pygame.sndarray.make_sound(wave)
            sound.play()
        except Exception:
            pass

    # ── Trigger alert ─────────────────────────────────────────────────────────
    def trigger(self, alert_type: str, message: str, play_sound: bool = True):
        """
        Args:
            alert_type : "drowsiness" | "phone" | "cigarette"
            message    : human-readable alert text
            play_sound : whether to beep
        """
        now = time.time()
        # Cooldown check
        if now - self.last_alerts.get(alert_type, 0) < self.cooldown:
            return False

        self.last_alerts[alert_type] = now
        self.alert_log.append({
            "time":       time.strftime("%H:%M:%S"),
            "type":       alert_type,
            "message":    message
        })

        if play_sound:
            freq_map = {"drowsiness": 660, "phone": 880, "cigarette": 770}
            self._beep(freq=freq_map.get(alert_type, 880))

        return True

    # ── Get stats ─────────────────────────────────────────────────────────────
    def get_stats(self):
        drowsy_count = sum(1 for a in self.alert_log if a["type"] == "drowsiness")
        phone_count  = sum(1 for a in self.alert_log if a["type"] == "phone")
        cig_count    = sum(1 for a in self.alert_log if a["type"] == "cigarette")
        return {
            "total":      len(self.alert_log),
            "drowsiness": drowsy_count,
            "phone":      phone_count,
            "cigarette":  cig_count,
            "log":        self.alert_log[-10:]   # last 10
        }

    def reset(self):
        self.alert_log   = []
        self.last_alerts = {}
