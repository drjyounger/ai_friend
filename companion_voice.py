"""
Companion Voice Module
======================
Handles the AI companion's voice output using ElevenLabs TTS.

Configure your preferred voice using ELEVENLABS_VOICE_ID in your .env file.
You can choose any voice from the ElevenLabs library that matches
your companion's personality.
"""

import os
import io
import tempfile
import time
import numpy as np
from typing import Optional, Callable, List
import threading

# Expense tracking - helps track API usage and costs
try:
    import expense_tracker
    _expense_tracking_enabled = True
except ImportError:
    _expense_tracking_enabled = False


class CompanionVoice:
    """
    Text-to-Speech handler using ElevenLabs.
    Gives the companion their distinctive voice.
    """
    
    def __init__(self, 
                 api_key: str = None,
                 voice_id: str = None,
                 model_id: str = "eleven_turbo_v2"):
        """
        Initialize the voice module.
        
        Args:
            api_key: ElevenLabs API key (defaults to env var)
            voice_id: ElevenLabs voice ID for the companion
            model_id: Which ElevenLabs model to use
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID")
        self.model_id = model_id
        
        self._client = None
        self._pygame_initialized = False
        
        # Current amplitude for visualization (thread-safe via simple assignment)
        self._current_amplitude = 0.0
        self._is_speaking = False
        
        if not self.api_key:
            print("⚠️  ELEVENLABS_API_KEY not set. Voice will be disabled.")
            print("   Set it in your .env file to enable the companion's voice.")
    
    def get_current_amplitude(self) -> float:
        """Get current audio amplitude for visualization (0.0 to 1.0)."""
        return self._current_amplitude
    
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking
            
    def _init_elevenlabs(self):
        """Lazy initialization of ElevenLabs client."""
        if self._client is not None:
            return
            
        if not self.api_key:
            return
            
        try:
            from elevenlabs import ElevenLabs
            self._client = ElevenLabs(api_key=self.api_key)
            print("🎤 ElevenLabs initialized.")
        except ImportError:
            print("⚠️  elevenlabs package not installed. Run: pip install elevenlabs")
            
    def _init_pygame(self):
        """Initialize pygame for audio playback."""
        if self._pygame_initialized:
            return
            
        try:
            import pygame
            pygame.mixer.init()
            self._pygame_initialized = True
        except ImportError:
            print("⚠️  pygame not installed. Run: pip install pygame")
            
    def speak(self, text: str, wait: bool = True, amplitude_callback: Callable[[float], None] = None):
        """
        Convert text to speech and play it.
        
        Args:
            text: What the companion should say
            wait: Whether to wait for playback to complete
            amplitude_callback: Optional callback that receives amplitude (0.0-1.0) for visualization
        """
        if not text:
            return
            
        # Try ElevenLabs first
        if self.api_key and self.voice_id:
            self._speak_elevenlabs(text, wait, amplitude_callback)
        else:
            # Fallback to system TTS (no amplitude support)
            self._speak_system(text)
            
    def _speak_elevenlabs(self, text: str, wait: bool = True, amplitude_callback: Callable[[float], None] = None):
        """Use ElevenLabs for high-quality TTS with streaming for lower latency."""
        self._init_elevenlabs()
        self._init_pygame()
        
        if self._client is None:
            self._speak_system(text)
            return
        
        self._is_speaking = True
        self._current_amplitude = 0.0
        
        # Track expense (for budgeting)
        if _expense_tracking_enabled:
            expense_tracker.log_elevenlabs(len(text))
            
        try:
            import pygame
            
            print(f"🗣️  Companion: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            
            # Use turbo model for speed
            audio_generator = self._client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",  # Fastest model
                output_format="mp3_22050_32"   # Lower quality = faster
            )
            
            # Collect audio bytes from generator
            audio_bytes = b''.join(audio_generator)
            
            # Extract amplitude envelope for visualization
            amplitude_envelope = self._extract_amplitude_envelope(audio_bytes)
            
            # Play using pygame
            audio_io = io.BytesIO(audio_bytes)
            pygame.mixer.music.load(audio_io)
            pygame.mixer.music.play()
            
            if wait:
                start_time = time.time()
                
                while pygame.mixer.music.get_busy():
                    # Update amplitude for visualization
                    if amplitude_envelope is not None:
                        elapsed = time.time() - start_time
                        envelope_idx = int(elapsed * 30)  # 30 samples per second
                        if envelope_idx < len(amplitude_envelope):
                            self._current_amplitude = amplitude_envelope[envelope_idx]
                        else:
                            self._current_amplitude = 0.0
                    else:
                        # Fallback: simulate amplitude from elapsed time
                        self._current_amplitude = 0.5 + 0.3 * np.sin(time.time() * 8)
                    
                    # Also call callback if provided
                    if amplitude_callback is not None:
                        amplitude_callback(self._current_amplitude)
                    
                    pygame.time.Clock().tick(30)  # 30fps for smooth visualization
                    
        except Exception as e:
            print(f"⚠️  ElevenLabs error: {e}")
            self._speak_system(text)
        finally:
            self._is_speaking = False
            self._current_amplitude = 0.0
    
    def _extract_amplitude_envelope(self, mp3_bytes: bytes) -> Optional[List[float]]:
        """
        Extract amplitude envelope from MP3 audio for visualization.
        Returns a list of amplitude values (0.0 to 1.0) at ~30 samples per second.
        """
        try:
            from pydub import AudioSegment
            
            # Load MP3 from bytes
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
            
            # Convert to raw samples
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            
            # If stereo, convert to mono
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            
            # Calculate RMS amplitude in chunks (~30 per second)
            # audio.frame_rate is samples per second
            chunk_size = audio.frame_rate // 30  # ~30 chunks per second
            
            envelope = []
            for i in range(0, len(samples), chunk_size):
                chunk = samples[i:i + chunk_size]
                if len(chunk) > 0:
                    # RMS amplitude
                    rms = np.sqrt(np.mean(chunk ** 2))
                    envelope.append(rms)
            
            # Normalize to 0-1 range
            if envelope:
                max_amp = max(envelope) if max(envelope) > 0 else 1.0
                envelope = [min(1.0, amp / max_amp) for amp in envelope]
            
            return envelope
            
        except ImportError:
            # pydub not available - fall back to simulated amplitude
            return None
        except Exception as e:
            print(f"   ⚠️ Could not extract amplitude: {e}")
            return None
            
    def _speak_system(self, text: str):
        """Fallback to system TTS (macOS 'say' command)."""
        import subprocess
        import platform
        
        print(f"🗣️  [System TTS] Companion: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        
        if platform.system() == "Darwin":  # macOS
            # Use macOS 'say' command with system voice
            try:
                subprocess.run(
                    ["say", "-v", "Samantha", text],
                    check=True
                )
            except subprocess.CalledProcessError:
                # Fallback to default voice
                subprocess.run(["say", text])
        else:
            print("   (System TTS not available on this platform)")
            
    def speak_async(self, text: str):
        """
        Speak without blocking.
        
        Args:
            text: What the companion should say
        """
        thread = threading.Thread(target=self.speak, args=(text, True))
        thread.daemon = True
        thread.start()
        return thread
        
    def list_available_voices(self) -> list:
        """
        List available ElevenLabs voices.
        Useful for finding the perfect voice for your companion.
        
        Returns:
            List of voice dictionaries
        """
        self._init_elevenlabs()
        
        if self._client is None:
            return []
            
        try:
            voices = self._client.voices.get_all()
            return [
                {
                    "voice_id": v.voice_id,
                    "name": v.name,
                    "labels": v.labels if hasattr(v, 'labels') else {}
                }
                for v in voices.voices
            ]
        except Exception as e:
            print(f"⚠️  Error listing voices: {e}")
            return []
            
    def preview_voice(self, voice_id: str, text: str = None):
        """
        Preview a specific voice.
        
        Args:
            voice_id: The voice ID to preview
            text: Test text (defaults to a sample greeting)
        """
        if text is None:
            text = "Hello! How are you doing today? It's nice to meet you."
            
        old_voice = self.voice_id
        self.voice_id = voice_id
        self.speak(text)
        self.voice_id = old_voice


class VoiceFinder:
    """
    Utility class to help find the perfect voice for your companion.
    Search ElevenLabs voice library for voices matching your criteria.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        
    def search_voices(self, 
                      gender: str = None,
                      accent: str = None) -> list:
        """
        Search for voices matching your criteria.
        
        Args:
            gender: Voice gender to search for (optional)
            accent: Accent to search for (optional)
            
        Returns:
            List of matching voices
        """
        if not self.api_key:
            print("⚠️  Need ELEVENLABS_API_KEY to search voices")
            return []
            
        try:
            from elevenlabs import ElevenLabs
            client = ElevenLabs(api_key=self.api_key)
            
            # Get all voices
            all_voices = client.voices.get_all()
            
            # Filter for criteria
            matches = []
            for voice in all_voices.voices:
                labels = voice.labels if hasattr(voice, 'labels') else {}
                
                # Check if voice matches criteria
                voice_gender = labels.get('gender', '').lower()
                voice_accent = labels.get('accent', '').lower()
                
                # If no filters specified, return all voices
                if gender is None and accent is None:
                    matches.append({
                        "voice_id": voice.voice_id,
                        "name": voice.name,
                        "labels": labels,
                        "preview_url": voice.preview_url if hasattr(voice, 'preview_url') else None
                    })
                elif (gender and gender.lower() in voice_gender) or (accent and accent.lower() in voice_accent):
                    matches.append({
                        "voice_id": voice.voice_id,
                        "name": voice.name,
                        "labels": labels,
                        "preview_url": voice.preview_url if hasattr(voice, 'preview_url') else None
                    })
                    
            return matches
            
        except Exception as e:
            print(f"⚠️  Error searching voices: {e}")
            return []


# Example voices you might try (search ElevenLabs for more)
SUGGESTED_VOICES = [
    {
        "name": "Charlotte",
        "description": "British, warm, conversational",
        "voice_id": "XB0fDUnXU5powFXDhCwa"
    },
    {
        "name": "Adam",
        "description": "American, clear, narrator-style",
        "voice_id": "pNInz6obpgDQGcFmaJgB"
    },
    {
        "name": "Antoni",
        "description": "Calm, well-rounded",
        "voice_id": "ErXwobaYiN019PkySvjV"
    }
]


if __name__ == "__main__":
    # Test the voice module
    print("Testing Companion Voice Module")
    print("=" * 50)
    
    voice = CompanionVoice()
    
    # Test system TTS (always available on macOS)
    print("\n🎤 Testing system TTS...")
    voice._speak_system("Hello! This is a test of the system voice.")
    
    # Test ElevenLabs if configured
    if voice.api_key and voice.voice_id:
        print("\n🎤 Testing ElevenLabs...")
        voice.speak("Welcome back. Did you get coffee? You look like you need coffee.")
        
        print("\n📋 Available voices:")
        voices = voice.list_available_voices()
        for v in voices[:10]:  # Show first 10
            print(f"   - {v['name']}: {v['voice_id']}")
    else:
        print("\n⚠️  ElevenLabs not configured.")
        print("   Set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env")
        print("\n   Suggested voices to try:")
        for v in SUGGESTED_VOICES:
            print(f"   - {v['name']}: {v['description']}")
            print(f"     voice_id: {v['voice_id']}")
            
    print("\n✅ Voice module test complete!")

