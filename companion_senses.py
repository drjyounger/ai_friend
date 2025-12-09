"""
Companion Senses Module (Improved)
==================================
Fixes for speech cutoff and responsiveness:
1. Rolling pre-buffer to capture speech onset
2. Adaptive noise threshold
3. Smoothed voice activity detection
4. Longer silence tolerance for natural pauses
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Tuple
from collections import deque
import sounddevice as sd
from scipy.io import wavfile
import tempfile
import os


class CompanionEyes:
    """
    Webcam handler with face detection.
    (Unchanged from original - focus is on audio fixes)
    """
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.face_cascade = None
        self.is_running = False
        self._last_frame = None
        self._face_detected = False
        
    def start(self):
        if self.cap is not None:
            return
            
        print(f"👁  Initializing webcam (index {self.camera_index})...")
        
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {self.camera_index}. "
                "Check that no other application is using it."
            )
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.is_running = True
        print("👁  Webcam active. Watching for presence...")
        
    def stop(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("👁  Webcam stopped.")
        
    def check_for_presence(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None or not self.is_running:
            return False, None
            
        ret, frame = self.cap.read()
        
        if not ret:
            return False, None
            
        self._last_frame = frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        self._face_detected = len(faces) > 0
        
        return self._face_detected, frame
    
    def get_frame_for_gemini(self, frame: np.ndarray = None) -> Optional[bytes]:
        if frame is None:
            frame = self._last_frame
            
        if frame is None:
            return None
            
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        
        return buffer.tobytes()
    
    def get_frame_base64(self, frame: np.ndarray = None) -> Optional[str]:
        import base64
        
        jpeg_bytes = self.get_frame_for_gemini(frame)
        if jpeg_bytes is None:
            return None
            
        return base64.b64encode(jpeg_bytes).decode('utf-8')


class CompanionEars:
    """
    Improved microphone handler with:
    - Rolling pre-buffer (captures speech onset)
    - Adaptive noise floor detection
    - Smoothed VAD to prevent premature cutoff
    - Configurable silence tolerance
    """
    
    def __init__(self, 
                 device_index: int = None,
                 sample_rate: int = 16000,
                 base_threshold: float = 0.015,
                 pre_buffer_seconds: float = 0.8,
                 silence_duration: float = 1.2,
                 smoothing_window: int = 5):
        """
        Initialize the microphone with improved settings.
        
        Args:
            device_index: Which microphone to use (None = default)
            sample_rate: Audio sample rate in Hz
            base_threshold: Base voice activity threshold (adapts to noise)
            pre_buffer_seconds: How much audio to keep before speech detection
            silence_duration: How long of silence ends recording
            smoothing_window: Number of frames to average for VAD
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.base_threshold = base_threshold
        self.pre_buffer_seconds = pre_buffer_seconds
        self.silence_duration = silence_duration
        self.smoothing_window = smoothing_window
        
        self.is_running = False
        self.is_listening = False
        self._stream = None
        
        # Rolling pre-buffer to capture speech onset
        pre_buffer_samples = int(pre_buffer_seconds * sample_rate)
        self._pre_buffer = deque(maxlen=pre_buffer_samples)
        
        # Audio recording buffer
        self._audio_buffer = []
        
        # Smoothed RMS for stable VAD
        self._rms_history = deque(maxlen=smoothing_window)
        self._latest_rms = 0.0
        self._smoothed_rms = 0.0
        self._current_rms = 0.0  # For audio level display
        
        # Adaptive noise floor
        self._noise_floor = 0.005
        self._noise_samples = deque(maxlen=100)  # Track ambient noise
        self._calibration_complete = False
        
        # Effective threshold (adapts based on noise)
        self._effective_threshold = base_threshold
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for continuous audio stream."""
        if status:
            print(f"⚠️  Audio status: {status}")
        
        # Flatten to 1D
        audio_chunk = indata[:, 0].copy()
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        self._latest_rms = rms
        
        # Update RMS history for smoothing
        self._rms_history.append(rms)
        self._smoothed_rms = np.mean(self._rms_history)
        
        # Track current RMS for audio level display
        self._current_rms = rms
        
        # Update noise floor during quiet periods
        if not self.is_listening and rms < self._effective_threshold * 1.5:
            self._noise_samples.append(rms)
            if len(self._noise_samples) >= 50:
                self._noise_floor = np.percentile(list(self._noise_samples), 75)
                # Threshold = noise floor * multiplier
                # Speech should be significantly louder than ambient noise
                # Use 5x multiplier - truly relative to calibrated noise
                self._effective_threshold = self._noise_floor * 5.0
                
                # Absolute minimum floor (for very quiet environments)
                if self._effective_threshold < 0.003:
                    self._effective_threshold = 0.003
                    
                if not self._calibration_complete:
                    print(f"🎚️  Noise floor calibrated: {self._noise_floor:.4f}, "
                          f"threshold: {self._effective_threshold:.4f}")
                    self._calibration_complete = True
        
        # Always add to pre-buffer (rolling window)
        self._pre_buffer.extend(audio_chunk)
        
        # Add to recording buffer if actively listening
        if self.is_listening:
            self._audio_buffer.append(audio_chunk)
        
    def start(self):
        """Start the continuous audio stream."""
        print(f"👂 Initializing microphone (improved VAD)...")
        
        devices = sd.query_devices()
        print(f"   Available audio devices: {len(devices)}")
        
        try:
            # Larger block size for more stable RMS
            block_size = int(self.sample_rate * 0.1)  # 100ms blocks
            
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=self.device_index,
                blocksize=block_size,
                callback=self._audio_callback
            )
            self._stream.start()
            self.is_running = True
            print(f"👂 Microphone active. Calibrating noise floor...")
            print(f"   (Stay quiet for 2 seconds for best results)")
        except Exception as e:
            print(f"⚠️  Could not start microphone: {e}")
            self.is_running = False
        
    def stop(self):
        """Stop the audio stream."""
        self.is_running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("👂 Microphone stopped.")
        
    def check_for_voice(self, duration: float = 0.1) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if voice activity is detected using smoothed RMS.
        
        Returns:
            Tuple of (voice_detected: bool, audio_chunk: None)
        """
        if not self.is_running or self._stream is None:
            return False, None
        
        # Use smoothed RMS for more stable detection
        voice_detected = self._smoothed_rms > self._effective_threshold
        
        return voice_detected, None
    
    def record_until_silence(self, 
                            silence_duration: float = None,
                            max_duration: float = 60.0,
                            min_speech_duration: float = 0.5) -> Optional[np.ndarray]:
        """
        Record audio with pre-buffer inclusion and patient silence detection.
        
        Args:
            silence_duration: Override default silence duration
            max_duration: Maximum recording duration
            min_speech_duration: Minimum speech before silence counts
            
        Returns:
            Numpy array of recorded audio, or None if nothing recorded
        """
        if silence_duration is None:
            silence_duration = self.silence_duration
            
        print(f"🎙️  Recording (threshold: {self._effective_threshold:.4f})...")
        
        # Start with pre-buffer content (captures speech onset!)
        self._audio_buffer = [np.array(list(self._pre_buffer))]
        pre_buffer_duration = len(self._pre_buffer) / self.sample_rate
        print(f"   Pre-buffer: {pre_buffer_duration:.2f}s of audio")
        
        self.is_listening = True
        
        silence_start = None
        speech_started = False
        speech_duration = 0.0
        start_time = time.time()
        
        # More detailed status tracking
        consecutive_speech_frames = 0
        consecutive_silence_frames = 0
        frames_for_confirmation = 4  # Need 4 frames to confirm state change (more stable)
        
        while (time.time() - start_time) < max_duration:
            current_time = time.time()
            is_speech = self._smoothed_rms > self._effective_threshold
            
            if is_speech:
                consecutive_speech_frames += 1
                consecutive_silence_frames = 0
                
                if consecutive_speech_frames >= frames_for_confirmation:
                    if not speech_started:
                        speech_started = True
                        print("   📢 Speech detected")
                    silence_start = None
                    
            else:
                consecutive_silence_frames += 1
                consecutive_speech_frames = 0
                
                if speech_started and consecutive_silence_frames >= frames_for_confirmation:
                    # Only start counting silence after minimum speech
                    speech_duration = current_time - start_time
                    
                    if speech_duration >= min_speech_duration:
                        if silence_start is None:
                            silence_start = current_time
                            
                        silence_elapsed = current_time - silence_start
                        
                        # Adaptive silence tolerance based on speech length
                        # Longer utterances get more patience for pauses
                        adaptive_silence = silence_duration
                        if speech_duration > 2.0:
                            adaptive_silence = silence_duration * 1.25  # 2.5s silence
                        if speech_duration > 5.0:
                            adaptive_silence = silence_duration * 1.5   # 3s silence
                        if speech_duration > 10.0:
                            adaptive_silence = silence_duration * 2.0   # 4s silence for long statements
                        
                        if silence_elapsed > adaptive_silence:
                            print(f"   🔇 End of speech (silence: {silence_elapsed:.1f}s)")
                            break
            
            time.sleep(0.05)  # 50ms check interval
        
        self.is_listening = False
        
        if not self._audio_buffer:
            return None
        
        # Concatenate all chunks
        try:
            full_audio = np.concatenate(self._audio_buffer)
        except ValueError:
            return None
            
        duration = len(full_audio) / self.sample_rate
        
        if duration < 0.3:  # Too short to be meaningful
            print(f"   (Recording too short: {duration:.2f}s)")
            return None
            
        print(f"🎙️  Captured {duration:.1f} seconds")
        
        return full_audio
    
    def save_audio_to_file(self, audio: np.ndarray, filename: str = None) -> str:
        """Save audio to a temporary WAV file."""
        if filename is None:
            fd, filename = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
        # Ensure audio is in correct format
        if audio.dtype == np.float32:
            audio_int16 = (audio * 32767).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)
        
        wavfile.write(filename, self.sample_rate, audio_int16)
        
        return filename
    
    def get_audio_level(self) -> float:
        """Get current audio level (0.0 to 1.0 normalized)."""
        if self._effective_threshold > 0:
            return min(1.0, self._smoothed_rms / (self._effective_threshold * 3))
        return 0.0
    
    def recalibrate(self):
        """
        Reset and recalibrate noise floor.
        Call this when moving to a new room/environment.
        """
        print("\n🎚️  Recalibrating noise floor...")
        print("   (Stay quiet for 2 seconds)")
        
        # Reset calibration state
        self._noise_samples.clear()
        self._calibration_complete = False
        self._noise_floor = 0.005
        self._effective_threshold = self.base_threshold
        
        # The callback will automatically recalibrate as it collects new samples


class CompanionSentinel:
    """
    Combined presence detection system.
    """
    
    def __init__(self):
        self.eyes = CompanionEyes()
        self.ears = CompanionEars(
            pre_buffer_seconds=1.0,   # Capture 1s before speech detection
            silence_duration=2.0,     # Wait 2s of silence before cutting (more patient)
            smoothing_window=7        # Smooth over 7 frames (~700ms) for stability
        )
        self._user_present = False
        self._user_speaking = False
        
    def start(self):
        """Start all sensors."""
        self.eyes.start()
        self.ears.start()
        
    def stop(self):
        """Stop all sensors."""
        self.eyes.stop()
        self.ears.stop()
        
    def check_presence(self) -> dict:
        """Comprehensive presence check."""
        face_detected, frame = self.eyes.check_for_presence()
        voice_detected, audio = self.ears.check_for_voice()
        
        self._user_present = face_detected
        self._user_speaking = voice_detected
        
        return {
            "face_detected": face_detected,
            "voice_detected": voice_detected,
            "should_respond": face_detected or voice_detected,
            "frame": frame,
            "audio": audio,
            "audio_level": self.ears.get_audio_level()
        }
    
    @property
    def user_is_here(self) -> bool:
        return self._user_present or self._user_speaking


if __name__ == "__main__":
    print("Testing Improved Companion Senses Module")
    print("=" * 50)
    
    sentinel = CompanionSentinel()
    
    try:
        sentinel.start()
        
        print("\n🔊 Testing voice capture...")
        print("   Speak a complete sentence when ready.\n")
        
        # Wait for calibration
        time.sleep(2)
        
        print("   Listening now - speak!")
        
        # Wait for voice
        while True:
            result = sentinel.check_presence()
            level_bar = "█" * int(result["audio_level"] * 20)
            print(f"\r   Level: [{level_bar:<20}]", end="", flush=True)
            
            if result["voice_detected"]:
                print("\n")
                audio = sentinel.ears.record_until_silence()
                
                if audio is not None:
                    path = sentinel.ears.save_audio_to_file(audio)
                    print(f"   Saved to: {path}")
                    print("   ✅ Voice capture test complete!")
                break
                
            time.sleep(0.05)
        
    except KeyboardInterrupt:
        print("\n   Interrupted")
        
    finally:
        sentinel.stop()