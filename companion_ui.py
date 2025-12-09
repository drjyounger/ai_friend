"""
Companion Visual Interface
==========================
A visual presence for your AI companion featuring:
- Her avatar/profile image
- Status indicator (Listening, Thinking, Speaking)
- Audio waveform visualization when speaking
- Debug/Terminal log panel with toggle
- Dock-friendly windowed mode (not true fullscreen)
"""

import pygame
import numpy as np
import threading
import queue
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
from enum import Enum
from collections import deque

# Import mode status for UI display
try:
    from companion_mode import is_restricted_mode, get_mode_display_name
except ImportError:
    def is_restricted_mode(): return False
    def get_mode_display_name(): return ""


class _TeeStdout:
    """
    A stdout wrapper that captures output to a buffer while still printing to original stdout.
    This allows the UI to display terminal output without blocking normal print statements.
    """
    def __init__(self, original_stdout, buffer: deque):
        self.original = original_stdout
        self.buffer = buffer
        self._current_line = ""
    
    def write(self, text):
        # Write to original stdout
        self.original.write(text)
        
        # Capture for UI display
        if text:
            self._current_line += text
            # Process complete lines
            while '\n' in self._current_line:
                line, self._current_line = self._current_line.split('\n', 1)
                if line.strip():  # Only add non-empty lines
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self.buffer.append({
                        "time": timestamp,
                        "message": line.strip()[:80],  # Truncate long lines
                        "level": "terminal"
                    })
            # Handle carriage return (overwrite line) - common for progress bars
            if '\r' in self._current_line:
                self._current_line = self._current_line.split('\r')[-1]
    
    def flush(self):
        self.original.flush()
    
    def isatty(self):
        return self.original.isatty() if hasattr(self.original, 'isatty') else False
    
    def fileno(self):
        return self.original.fileno() if hasattr(self.original, 'fileno') else -1


class CompanionState(Enum):
    """The companion's current state for display."""
    IDLE = "..."
    LISTENING = "Listening..."
    THINKING = "Thinking..."
    SPEAKING = "Speaking..."
    STARTUP = "Waking up..."


class CompanionUI:
    """
    Full-screen visual interface for the AI companion.
    
    Runs in a separate thread to not block the main conversation loop.
    """
    
    def __init__(self, 
                 avatar_path: str = "companion_art/companion_profile.png",
                 fullscreen: bool = True):
        """
        Initialize the visual interface.
        
        Args:
            avatar_path: Path to the companion's profile image
            fullscreen: Whether to run in fullscreen mode
        """
        self.avatar_path = avatar_path
        self.fullscreen = fullscreen
        
        # State
        self._state = CompanionState.STARTUP
        self._subtitle = ""  # Optional subtitle text
        self._audio_levels = np.zeros(64)  # For waveform visualization
        
        # Thread communication
        self._running = False
        self._ui_thread: Optional[threading.Thread] = None
        self._command_queue = queue.Queue()
        
        # Colors (elegant dark theme)
        self.BG_COLOR = (15, 15, 20)  # Near black
        self.TEXT_COLOR = (200, 200, 210)  # Soft white
        self.ACCENT_COLOR = (100, 150, 255)  # Soft blue
        self.WAVE_COLOR = (80, 120, 200)  # Wave blue
        self.GLOW_COLOR = (60, 100, 180)  # Subtle glow
        
        # State-specific colors
        self.STATE_COLORS = {
            CompanionState.IDLE: (100, 100, 110),
            CompanionState.LISTENING: (100, 200, 150),  # Green
            CompanionState.THINKING: (200, 180, 100),   # Amber
            CompanionState.SPEAKING: (100, 150, 255),   # Blue
            CompanionState.STARTUP: (150, 100, 200),    # Purple
        }
        
        # Avatar visibility (can be dismissed with X button)
        self._show_avatar = True  # Hidden once dismissed, until restart
        self._avatar_close_button_rect: Optional[pygame.Rect] = None
        
        # Debug log panel
        self._log_buffer: deque = deque(maxlen=15)  # Last 15 log lines
        self._terminal_buffer: deque = deque(maxlen=50)  # Last 50 terminal lines
        self._debug_expanded = True  # Collapsible with chevron
        self._showing_terminal = False  # Toggle between debug and terminal logs
        self.LOG_BG_COLOR = (20, 20, 25, 200)  # Semi-transparent dark
        self.LOG_TEXT_COLOR = (150, 150, 160)
        self.LOG_HIGHLIGHT_COLOR = (100, 180, 255)  # For API timing
        self.LOG_WARNING_COLOR = (255, 180, 100)  # For warnings
        self.LOG_ERROR_COLOR = (255, 100, 100)  # For errors
        self._debug_chevron_rect: Optional[pygame.Rect] = None
        self._log_toggle_rect: Optional[pygame.Rect] = None  # Toggle button rect
        
        # Terminal capture
        self._stdout_capture = None
        self._original_stdout = None
        
        # API latency tracking
        self._api_latencies: deque = deque(maxlen=10)  # Last 10 API calls
        self._current_api_start: Optional[float] = None
        self._last_api_latency: Optional[float] = None
    
    def start(self):
        """Initialize the UI (must be called from main thread on macOS)."""
        if self._running:
            return
        
        self._running = True
        
        # Start capturing stdout for terminal view
        self._start_stdout_capture()
        
        # Initialize pygame on main thread (required for macOS)
        pygame.init()
        
        # Get display info
        display_info = pygame.display.Info()
        
        if self.fullscreen:
            # DOCK-FRIENDLY WINDOW: Leave space for dock (~70px) and menu bar (~25px)
            # This allows accessing the Mac dock and menu without blocking
            dock_margin = 80  # Space for dock at bottom
            menu_margin = 30  # Space for menu bar at top
            
            window_width = display_info.current_w
            window_height = display_info.current_h - dock_margin - menu_margin
            
            # Use NOFRAME for borderless look, but NOT FULLSCREEN
            self.screen = pygame.display.set_mode(
                (window_width, window_height),
                pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            
            # Position window below menu bar (move window to y=menu_margin)
            # Note: pygame doesn't have a direct method, but we can use SDL2 env var
            import os
            os.environ['SDL_VIDEO_WINDOW_POS'] = f'0,{menu_margin}'
            
            # Reinitialize with the position hint
            pygame.display.quit()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (window_width, window_height),
                pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF
            )
        else:
            self.screen = pygame.display.set_mode((1280, 720))
        
        pygame.display.set_caption("AI Companion")
        
        self.width, self.height = self.screen.get_size()
        self.clock = pygame.time.Clock()
        
        # Load fonts
        try:
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)
        except:
            self.font_large = pygame.font.SysFont('arial', 48)
            self.font_medium = pygame.font.SysFont('arial', 32)
            self.font_small = pygame.font.SysFont('arial', 24)
        
        # Load avatar
        self.avatar = self._load_avatar()
        
        # Animation state
        self.glow_phase = 0
        
        print("🖥️  Visual interface started (dock-friendly mode)")
    
    def stop(self):
        """Stop the UI."""
        self._running = False
        self._stop_stdout_capture()
        pygame.quit()
        print("🖥️  Visual interface stopped")
    
    def _start_stdout_capture(self):
        """Start capturing stdout for terminal view."""
        self._original_stdout = sys.stdout
        self._stdout_capture = _TeeStdout(self._original_stdout, self._terminal_buffer)
        sys.stdout = self._stdout_capture
    
    def _stop_stdout_capture(self):
        """Stop capturing stdout and restore original."""
        if self._original_stdout:
            sys.stdout = self._original_stdout
            self._original_stdout = None
            self._stdout_capture = None
    
    def update(self):
        """
        Update the UI - must be called periodically from the main loop.
        Returns False if the UI was closed (ESC pressed or window closed).
        """
        if not self._running:
            return True
        
        self._handle_events()
        self._update()
        self._draw()
        
        # Don't limit framerate here - let the main loop control timing
        return self._running
    
    def set_state(self, state: CompanionState, subtitle: str = ""):
        """Update the companion's displayed state."""
        self._state = state
        self._subtitle = subtitle
    
    def set_audio_levels(self, levels: np.ndarray):
        """Update audio visualization levels (0.0 to 1.0 array)."""
        if len(levels) > 0:
            # Resample to 64 bars if needed
            if len(levels) != 64:
                indices = np.linspace(0, len(levels) - 1, 64).astype(int)
                self._audio_levels = np.clip(levels[indices], 0, 1)
            else:
                self._audio_levels = np.clip(levels, 0, 1)
    
    def pulse_audio(self, intensity: float = 0.5):
        """Create a simple pulse effect for audio (fallback when we don't have real data)."""
        # Generate a smooth wave pattern
        t = time.time() * 3  # Speed of animation
        x = np.linspace(0, 4 * np.pi, 64)
        wave = (np.sin(x + t) + 1) / 2  # 0 to 1
        wave *= intensity
        # Add some randomness for life
        wave += np.random.random(64) * 0.1 * intensity
        self._audio_levels = np.clip(wave, 0, 1)
    
    def set_amplitude(self, amplitude: float):
        """
        Set the audio visualization based on a real amplitude value (0.0 to 1.0).
        Creates a dynamic equalizer-style visualization that responds to the amplitude.
        """
        # Clamp amplitude
        amplitude = max(0.0, min(1.0, amplitude))
        
        # Create an equalizer-style visualization
        # - Center bars are taller (like a speech pattern)
        # - Random variation for natural look
        # - Amplitude controls overall height
        
        t = time.time() * 5  # For subtle animation
        
        # Create base shape - center-weighted like speech frequencies
        x = np.linspace(-1, 1, 64)
        base_shape = 1 - np.abs(x) ** 0.8  # Center-weighted curve
        
        # Add some frequency-like variation
        freq_variation = np.sin(np.linspace(0, 8 * np.pi, 64) + t) * 0.3 + 0.7
        
        # Random per-bar variation for organic look
        random_factor = np.random.random(64) * 0.4 + 0.6
        
        # Combine: base shape * frequency variation * random * amplitude
        levels = base_shape * freq_variation * random_factor * amplitude
        
        # Add a minimum "floor" when there's any sound
        if amplitude > 0.05:
            levels = np.maximum(levels, 0.1 * amplitude)
        
        # Smooth transition from previous levels (prevents jarring jumps)
        self._audio_levels = self._audio_levels * 0.3 + np.clip(levels, 0, 1) * 0.7
    
    def clear_audio(self):
        """Clear the audio visualization."""
        self._audio_levels = np.zeros(64)
    
    def log(self, message: str, level: str = "info"):
        """
        Add a message to the debug log.
        
        Args:
            message: Log message
            level: "info", "warning", "error", "api", or "success"
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_buffer.append({
            "time": timestamp,
            "message": message,
            "level": level
        })
    
    def log_api_start(self, endpoint: str = "gemini"):
        """Mark the start of an API call for latency tracking."""
        self._current_api_start = time.time()
        self.log(f"→ API call: {endpoint}", "api")
    
    def log_api_end(self, endpoint: str = "gemini", success: bool = True):
        """Mark the end of an API call and record latency."""
        if self._current_api_start:
            latency = time.time() - self._current_api_start
            self._last_api_latency = latency
            self._api_latencies.append(latency)
            self._current_api_start = None
            
            level = "success" if success else "error"
            self.log(f"← API response: {latency:.2f}s", level)
    
    def get_avg_latency(self) -> Optional[float]:
        """Get average API latency from recent calls."""
        if self._api_latencies:
            return sum(self._api_latencies) / len(self._api_latencies)
        return None
    
    def toggle_debug(self):
        """Toggle debug panel visibility."""
        self._show_debug = not self._show_debug
    
    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                elif event.key == pygame.K_f:
                    # Toggle fullscreen (not used in dock-friendly mode)
                    pygame.display.toggle_fullscreen()
                elif event.key == pygame.K_d:
                    # Toggle debug panel expanded/collapsed
                    self._debug_expanded = not self._debug_expanded
                elif event.key == pygame.K_l:
                    # Toggle between debug and terminal logs
                    self._showing_terminal = not self._showing_terminal
                    mode = "terminal" if self._showing_terminal else "debug"
                    self.log(f"Switched to {mode} logs", "info")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_pos = event.pos
                    # Check if clicked on avatar close button
                    if self._avatar_close_button_rect and self._show_avatar:
                        if self._avatar_close_button_rect.collidepoint(mouse_pos):
                            self._show_avatar = False
                            self.log("Avatar dismissed", "info")
                    # Check if clicked on debug chevron
                    if self._debug_chevron_rect:
                        if self._debug_chevron_rect.collidepoint(mouse_pos):
                            self._debug_expanded = not self._debug_expanded
                    # Check if clicked on log toggle button
                    if self._log_toggle_rect:
                        if self._log_toggle_rect.collidepoint(mouse_pos):
                            self._showing_terminal = not self._showing_terminal
    
    def _update(self):
        """Update animation state."""
        self.glow_phase += 0.05
        if self.glow_phase > 2 * np.pi:
            self.glow_phase -= 2 * np.pi
        
        # Auto-decay audio levels for smooth animation
        self._audio_levels *= 0.9
    
    def _draw(self):
        """Draw the interface."""
        # Clear screen
        self.screen.fill(self.BG_COLOR)

        # Draw subtle background gradient/vignette
        self._draw_vignette()
        
        # Draw mode indicator (top of screen)
        self._draw_mode_indicator()

        # Draw avatar with glow
        self._draw_avatar()

        # Draw status
        self._draw_status()

        # Draw audio waveform
        self._draw_waveform()

        # Draw subtitle if present
        if self._subtitle:
            self._draw_subtitle()

        # Draw debug panel (bottom right) - always visible, collapsible with chevron
        self._draw_debug_panel()

        pygame.display.flip()
    
    def _draw_mode_indicator(self):
        """Draw the current mode indicator at the top of the screen."""
        mode_text = get_mode_display_name()
        if not mode_text:
            return
        
        # Color based on mode
        if is_restricted_mode():
            # Green for restricted/safe mode
            color = (100, 200, 100)
            bg_color = (40, 80, 40)
        else:
            # Blue for full mode
            color = (130, 180, 220)
            bg_color = (40, 60, 80)
        
        # Render mode text
        text_surface = self.font_small.render(mode_text, True, color)
        text_rect = text_surface.get_rect()
        
        # Position at top center with padding
        padding = 10
        bg_rect = pygame.Rect(
            (self.width - text_rect.width) // 2 - padding,
            15,
            text_rect.width + padding * 2,
            text_rect.height + padding
        )
        
        # Draw background pill
        pygame.draw.rect(self.screen, bg_color, bg_rect, border_radius=15)
        pygame.draw.rect(self.screen, color, bg_rect, width=1, border_radius=15)
        
        # Draw text
        text_rect.center = bg_rect.center
        self.screen.blit(text_surface, text_rect)
    
    def _draw_vignette(self):
        """Draw a subtle vignette effect."""
        # Simple radial gradient approximation using circles
        center = (self.width // 2, self.height // 2)
        max_radius = max(self.width, self.height)
        
        for i in range(5):
            radius = max_radius - (i * max_radius // 10)
            alpha = 10 + i * 5
            color = (self.BG_COLOR[0] + 5, self.BG_COLOR[1] + 5, self.BG_COLOR[2] + 10)
            # Skip drawing circles for performance - vignette is optional
    
    def _draw_avatar(self):
        """Draw the companion's avatar with a subtle glow effect and X close button."""
        if self.avatar is None or not self._show_avatar:
            self._avatar_close_button_rect = None
            return
        
        # Calculate position (centered, upper portion of screen)
        avatar_rect = self.avatar.get_rect()
        avatar_x = (self.width - avatar_rect.width) // 2
        avatar_y = (self.height - avatar_rect.height) // 2 - 80
        
        # Draw glow effect behind avatar
        glow_intensity = (np.sin(self.glow_phase) + 1) / 2 * 0.3 + 0.7
        state_color = self.STATE_COLORS.get(self._state, self.ACCENT_COLOR)
        glow_color = tuple(int(c * glow_intensity * 0.5) for c in state_color)
        
        # Draw glow circles
        for i in range(3):
            glow_radius = min(avatar_rect.width, avatar_rect.height) // 2 + 20 + i * 15
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            alpha = int(30 * glow_intensity * (3 - i) / 3)
            pygame.draw.circle(
                glow_surface, 
                (*glow_color, alpha),
                (glow_radius, glow_radius), 
                glow_radius
            )
            self.screen.blit(
                glow_surface, 
                (avatar_x + avatar_rect.width // 2 - glow_radius,
                 avatar_y + avatar_rect.height // 2 - glow_radius)
            )
        
        # Draw avatar
        self.screen.blit(self.avatar, (avatar_x, avatar_y))
        
        # Draw X close button (top-right corner of avatar)
        button_size = 28
        button_x = avatar_x + avatar_rect.width - button_size // 2
        button_y = avatar_y - button_size // 2
        
        # Store button rect for click detection
        self._avatar_close_button_rect = pygame.Rect(button_x, button_y, button_size, button_size)
        
        # Draw button background (circle)
        pygame.draw.circle(self.screen, (60, 60, 70), 
                          (button_x + button_size // 2, button_y + button_size // 2), 
                          button_size // 2)
        pygame.draw.circle(self.screen, (100, 100, 110), 
                          (button_x + button_size // 2, button_y + button_size // 2), 
                          button_size // 2, 2)
        
        # Draw X
        x_padding = 8
        x_color = (180, 180, 190)
        pygame.draw.line(self.screen, x_color, 
                        (button_x + x_padding, button_y + x_padding),
                        (button_x + button_size - x_padding, button_y + button_size - x_padding), 2)
        pygame.draw.line(self.screen, x_color,
                        (button_x + button_size - x_padding, button_y + x_padding),
                        (button_x + x_padding, button_y + button_size - x_padding), 2)
    
    def _draw_status(self):
        """Draw the current status text."""
        state_color = self.STATE_COLORS.get(self._state, self.TEXT_COLOR)
        
        # Pulsing effect for active states
        if self._state in [CompanionState.LISTENING, CompanionState.THINKING, CompanionState.SPEAKING]:
            pulse = (np.sin(self.glow_phase * 2) + 1) / 2 * 0.3 + 0.7
            state_color = tuple(int(c * pulse) for c in state_color)
        
        text = self.font_large.render(self._state.value, True, state_color)
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2 + 180))
        self.screen.blit(text, text_rect)
    
    def _draw_waveform(self):
        """Draw audio waveform visualization."""
        # Waveform area at bottom of screen
        wave_height = 100
        wave_y = self.height - 150
        wave_width = min(800, self.width - 100)
        wave_x = (self.width - wave_width) // 2
        
        bar_count = len(self._audio_levels)
        bar_width = wave_width // bar_count - 2
        bar_spacing = wave_width // bar_count
        
        state_color = self.STATE_COLORS.get(self._state, self.WAVE_COLOR)
        
        for i, level in enumerate(self._audio_levels):
            bar_height = int(level * wave_height)
            if bar_height < 2:
                bar_height = 2  # Minimum bar height for visibility
            
            x = wave_x + i * bar_spacing
            y = wave_y + (wave_height - bar_height) // 2
            
            # Color intensity based on level
            intensity = 0.3 + level * 0.7
            color = tuple(int(c * intensity) for c in state_color)
            
            pygame.draw.rect(
                self.screen, 
                color,
                (x, y, bar_width, bar_height),
                border_radius=2
            )
    
    def _draw_subtitle(self):
        """Draw subtitle text (e.g., what the companion is saying)."""
        # Truncate long text
        max_chars = 80
        text = self._subtitle
        if len(text) > max_chars:
            text = text[:max_chars-3] + "..."
        
        rendered = self.font_medium.render(text, True, self.TEXT_COLOR)
        text_rect = rendered.get_rect(center=(self.width // 2, self.height - 50))
        
        # Draw background for readability
        padding = 10
        bg_rect = text_rect.inflate(padding * 2, padding)
        pygame.draw.rect(self.screen, (*self.BG_COLOR, 200), bg_rect, border_radius=5)
        
        self.screen.blit(rendered, text_rect)
    
    def _draw_debug_panel(self):
        """Draw the debug/terminal log panel in the bottom right corner with collapsible chevron."""
        padding = 10
        line_height = 16
        header_height = 30
        
        # Panel dimensions depend on expanded state
        panel_width = 500  # Wider to show more terminal output
        if self._debug_expanded:
            panel_height = 320  # Taller for more logs
        else:
            panel_height = header_height + 10  # Collapsed: just header
        
        # Position (bottom right with margin)
        panel_x = self.width - panel_width - 20
        panel_y = self.height - panel_height - 20
        
        # Create semi-transparent surface
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((20, 20, 25, 230))  # Slightly more opaque
        
        # Draw border
        pygame.draw.rect(panel_surface, (50, 50, 60), 
                        (0, 0, panel_width, panel_height), 1, border_radius=5)
        
        # Draw chevron button (left side of header)
        chevron_size = 20
        chevron_x = padding
        chevron_y = padding - 2
        
        # Store chevron rect for click detection (in screen coordinates)
        self._debug_chevron_rect = pygame.Rect(
            panel_x + chevron_x, 
            panel_y + chevron_y, 
            chevron_size, 
            chevron_size
        )
        
        # Draw chevron background
        pygame.draw.rect(panel_surface, (40, 40, 50), 
                        (chevron_x, chevron_y, chevron_size, chevron_size), 
                        border_radius=3)
        
        # Draw chevron arrow (▼ when expanded, ▶ when collapsed)
        chevron_color = (120, 120, 140)
        cx, cy = chevron_x + chevron_size // 2, chevron_y + chevron_size // 2
        if self._debug_expanded:
            # Down arrow ▼
            points = [(cx - 5, cy - 2), (cx + 5, cy - 2), (cx, cy + 4)]
        else:
            # Right arrow ▶
            points = [(cx - 2, cy - 5), (cx - 2, cy + 5), (cx + 4, cy)]
        pygame.draw.polygon(panel_surface, chevron_color, points)
        
        # Header text - shows which mode we're in
        if self._showing_terminal:
            header_text = "TERMINAL"
            header_color = (150, 200, 255)  # Blue tint for terminal
        else:
            header_text = "DEBUG LOG"
            header_color = (100, 100, 120)
        header = self.font_small.render(header_text, True, header_color)
        panel_surface.blit(header, (chevron_x + chevron_size + 8, padding))
        
        # Toggle button (between debug and terminal)
        toggle_width = 50
        toggle_height = 18
        toggle_x = chevron_x + chevron_size + 85
        toggle_y = padding
        
        # Store toggle rect for click detection
        self._log_toggle_rect = pygame.Rect(
            panel_x + toggle_x,
            panel_y + toggle_y,
            toggle_width,
            toggle_height
        )
        
        # Draw toggle button
        toggle_bg_color = (60, 80, 100) if self._showing_terminal else (50, 50, 60)
        pygame.draw.rect(panel_surface, toggle_bg_color,
                        (toggle_x, toggle_y, toggle_width, toggle_height),
                        border_radius=3)
        pygame.draw.rect(panel_surface, (80, 80, 100),
                        (toggle_x, toggle_y, toggle_width, toggle_height),
                        1, border_radius=3)
        
        toggle_label = "Term" if not self._showing_terminal else "Debug"
        toggle_text = self.font_small.render(toggle_label, True, (140, 140, 160))
        text_x = toggle_x + (toggle_width - toggle_text.get_width()) // 2
        text_y = toggle_y + (toggle_height - toggle_text.get_height()) // 2
        panel_surface.blit(toggle_text, (text_x, text_y))
        
        # Keyboard hint
        hint_text = self.font_small.render("(L)", True, (80, 80, 100))
        panel_surface.blit(hint_text, (toggle_x + toggle_width + 5, padding))
        
        # API latency stats (always shown in header when in debug mode)
        if not self._showing_terminal:
            avg_latency = self.get_avg_latency()
            if avg_latency is not None:
                latency_text = f"Avg: {avg_latency:.2f}s"
                if self._last_api_latency:
                    latency_text += f" | Last: {self._last_api_latency:.2f}s"
                
                # Color based on latency
                if avg_latency < 2.0:
                    latency_color = (100, 200, 150)  # Green - fast
                elif avg_latency < 5.0:
                    latency_color = (200, 180, 100)  # Amber - moderate
                else:
                    latency_color = (255, 100, 100)  # Red - slow
                
                latency_rendered = self.font_small.render(latency_text, True, latency_color)
                panel_surface.blit(latency_rendered, (panel_width - latency_rendered.get_width() - padding, padding))
        
        # Only draw log entries if expanded
        if self._debug_expanded:
            # Divider line
            pygame.draw.line(panel_surface, (50, 50, 60), 
                            (padding, header_height), (panel_width - padding, header_height), 1)
            
            # Choose which buffer to display
            log_buffer = self._terminal_buffer if self._showing_terminal else self._log_buffer
            
            # Log entries
            y_offset = header_height + 10
            for entry in list(log_buffer):
                if y_offset > panel_height - line_height - padding:
                    break
                
                # Determine color based on level
                level = entry.get("level", "info")
                if level == "error":
                    text_color = self.LOG_ERROR_COLOR
                elif level == "warning":
                    text_color = self.LOG_WARNING_COLOR
                elif level == "api":
                    text_color = self.LOG_HIGHLIGHT_COLOR
                elif level == "success":
                    text_color = (100, 200, 150)
                elif level == "terminal":
                    text_color = (180, 180, 190)  # Slightly brighter for terminal
                else:
                    text_color = self.LOG_TEXT_COLOR
                
                # Format: [HH:MM:SS] message
                time_str = entry.get("time", "")
                message = entry.get("message", "")
                
                # Truncate long messages (wider panel allows more chars)
                max_msg_len = 58
                if len(message) > max_msg_len:
                    message = message[:max_msg_len-3] + "..."
                
                full_text = f"[{time_str}] {message}"
                
                text_rendered = self.font_small.render(full_text, True, text_color)
                panel_surface.blit(text_rendered, (padding, y_offset))
                
                y_offset += line_height
        
        # Blit panel to screen
        self.screen.blit(panel_surface, (panel_x, panel_y))
    
    def _load_avatar(self) -> Optional[pygame.Surface]:
        """Load and scale the avatar image."""
        try:
            avatar_path = Path(self.avatar_path)
            if not avatar_path.exists():
                print(f"   ⚠️ Avatar not found: {avatar_path}")
                return None
            
            avatar = pygame.image.load(str(avatar_path))
            
            # Scale to reasonable size (max 400px)
            max_size = 400
            width, height = avatar.get_size()
            scale = min(max_size / width, max_size / height)
            new_size = (int(width * scale), int(height * scale))
            
            avatar = pygame.transform.smoothscale(avatar, new_size)
            
            # Convert for faster blitting
            avatar = avatar.convert_alpha()
            
            print(f"   👤 Avatar loaded: {new_size[0]}x{new_size[1]}")
            return avatar
            
        except Exception as e:
            print(f"   ⚠️ Could not load avatar: {e}")
            return None


# Convenience functions for integration with main.py
_ui_instance: Optional[CompanionUI] = None

def start_ui(avatar_path: str = "companion_art/companion_profile.png", fullscreen: bool = True):
    """Start the visual interface."""
    global _ui_instance
    if _ui_instance is None:
        _ui_instance = CompanionUI(avatar_path=avatar_path, fullscreen=fullscreen)
    _ui_instance.start()
    return _ui_instance

def stop_ui():
    """Stop the visual interface."""
    global _ui_instance
    if _ui_instance:
        _ui_instance.stop()
        _ui_instance = None

def update_ui() -> bool:
    """Update the UI (call from main loop). Returns False if UI was closed."""
    if _ui_instance:
        return _ui_instance.update()
    return True

def get_ui() -> Optional[CompanionUI]:
    """Get the UI instance."""
    return _ui_instance

def set_state(state: CompanionState, subtitle: str = ""):
    """Set UI state (convenience function)."""
    if _ui_instance:
        _ui_instance.set_state(state, subtitle)

def set_speaking(text: str = ""):
    """Set speaking state with optional text."""
    set_state(CompanionState.SPEAKING, text)

def set_listening():
    """Set listening state."""
    set_state(CompanionState.LISTENING)

def set_thinking():
    """Set thinking state."""
    set_state(CompanionState.THINKING)

def set_idle():
    """Set idle state."""
    set_state(CompanionState.IDLE)

def pulse_audio(intensity: float = 0.5):
    """Pulse the audio visualization (fallback for when we don't have real amplitude)."""
    if _ui_instance:
        _ui_instance.pulse_audio(intensity)

def set_amplitude(amplitude: float):
    """Set audio visualization from real amplitude value (0.0 to 1.0)."""
    if _ui_instance:
        _ui_instance.set_amplitude(amplitude)

def log(message: str, level: str = "info"):
    """Log a message to the debug panel."""
    if _ui_instance:
        _ui_instance.log(message, level)

def log_api_start(endpoint: str = "gemini"):
    """Mark start of API call for latency tracking."""
    if _ui_instance:
        _ui_instance.log_api_start(endpoint)

def log_api_end(endpoint: str = "gemini", success: bool = True):
    """Mark end of API call and record latency."""
    if _ui_instance:
        _ui_instance.log_api_end(endpoint, success)

def clear_audio():
    """Clear audio visualization."""
    if _ui_instance:
        _ui_instance.clear_audio()


if __name__ == "__main__":
    # Test the UI
    print("Testing Companion Visual Interface")
    print("Press ESC to exit, D to toggle debug panel, L to toggle debug/terminal")
    
    ui = start_ui(fullscreen=True)  # Test dock-friendly mode
    
    try:
        # Test logging
        log("UI test started", "success")
        log("This is an info message", "info")
        log("This is a warning", "warning")
        log("API call started", "api")
        
        # Simulate states
        time.sleep(2)
        set_state(CompanionState.LISTENING)
        log("State: LISTENING", "info")
        time.sleep(2)
        set_state(CompanionState.THINKING)
        log("State: THINKING", "info")
        time.sleep(2)
        set_state(CompanionState.SPEAKING, "Hello! It's good to see you!")
        log("State: SPEAKING", "info")
        
        # Simulate audio
        for _ in range(50):
            pulse_audio(0.7)
            if not update_ui():
                break
            time.sleep(0.1)
        
        clear_audio()
        set_state(CompanionState.IDLE)
        log("Test complete - idle", "success")
        
        # Keep running until ESC
        while update_ui():
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        pass
    finally:
        stop_ui()
