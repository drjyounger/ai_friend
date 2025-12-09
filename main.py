#!/usr/bin/env python3
"""
AI Companion - Multimodal AI Assistant Framework
================================================
A customizable AI companion with:
1. Streaming Gemini responses for faster first-word latency
2. Async-friendly architecture
3. Voice activity handling
4. Visual feedback during processing

Your AI companion's personality is defined by:
- systemPrompt.md (personality definition)
- conversationSoFar.md (conversation history)
- referencePhotos/ (facial recognition)

Usage:
    python3 main.py              # Standard mode
    python3 main.py -restricted  # Restricted mode (safer/filtered)
    python3 main.py -fullmode    # Full mode (unrestricted)
"""

import asyncio
import os
import time
import sys
import select
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# CRITICAL: Parse mode BEFORE importing other modules
# This ensures mode is set before CompanionMemory and CompanionBrain initialize
from companion_mode import (
    parse_mode_from_args, 
    set_mode, 
    get_mode, 
    is_restricted_mode, 
    is_full_mode,
    get_mode_display_name,
    CompanionMode
)

# Parse and set mode immediately
_mode = parse_mode_from_args()
set_mode(_mode)

from companion_memory import CompanionMemory
from companion_senses import CompanionEyes, CompanionEars, SentinelWatcher
from companion_brain import CompanionBrain, TranscriptionService
from companion_voice import CompanionVoice
import companion_ui


class AICompanion:
    """
    The complete AI Companion system - optimized for responsive conversation.
    """
    
    def __init__(self):
        # Load companion name from environment or use default
        self.companion_name = os.getenv("COMPANION_NAME", "Companion")
        self.user_name = os.getenv("USER_NAME", "User")
        
        print("=" * 60)
        print(f"  🌟 {self.companion_name.upper()} - AI Companion")
        print("  Initializing...")
        print("=" * 60)
        print()
        
        # Display current mode prominently
        print(f"  MODE: {get_mode_display_name()}")
        if is_restricted_mode():
            print("  ✓ Restricted mode active")
            print("  ✓ Some conversation history excluded")
        else:
            print("  ✓ Full mode - complete context enabled")
        print()
        print("=" * 60)
        print()
        
        self._verify_files()
        
        print("📦 Initializing components...")
        print()
        
        self.memory = CompanionMemory()
        self.sentinel = SentinelWatcher()
        self.brain = CompanionBrain()
        self.voice = CompanionVoice()
        self.transcriber = TranscriptionService()
        
        # Presence State Machine
        # States: ACTIVE (chatting), RESTING (away), REUNION (just returned)
        self._presence_state = "INITIALIZING"
        self._user_present = False
        self._user_was_present = False
        
        # Timing for contextual awareness
        self._last_interaction_time = time.time()  # Last conversation exchange
        self._last_presence_time = time.time()     # Last time user was detected
        self._absence_start_time = None            # When user left
        self._absence_threshold = 120              # 2 min of no presence = "away"
        
        # Processing state
        self._is_processing = False
        self._voice_cooldown_until = 0
        self._is_speaking = False
        
        # Track who the companion thinks they're talking to (updated by identification)
        self._current_speaker = self.user_name  # Default until identified otherwise
        
        # Native Audio Mode - companion HEARS user's voice, not just text
        # They perceive tone, warmth, hesitation
        self._use_native_audio = os.getenv("USE_NATIVE_AUDIO", "true").lower() == "true"
        if self._use_native_audio:
            print(f"   🎙️  Native Audio: ENABLED ({self.companion_name} hears your voice)")
        else:
            print("   📝 Native Audio: DISABLED (text transcription only)")
        
        # Status display
        self._last_status_time = 0
        self._status_interval = 10.0  # Show resting status every 10s
        
        # Keyboard input handling
        self._pending_recalibrate = False
        self._pending_reflect = False  # Trigger self-reflection
        self._pending_image_review = False  # Trigger image file review
        self._selected_image_path = None  # Path to image file to show companion
        self._pending_creative_brief = False  # Trigger creative director mode
        self._creative_brief = None  # Creative brief data (images + instructions)
        self._pending_text_input = False  # Trigger text input mode
        self._text_input = None  # Text to send to companion
        self._muted = False  # Microphone mute toggle
        self._start_keyboard_listener()
        
        # Check for new changes on startup (ONLY in Full Mode)
        self._pending_change_review = False
        self._new_changes = None
        if is_full_mode():
            self._check_for_code_changes()
        else:
            print("🔍 Code change review: SKIPPED (Restricted Mode)")
        
        # Start visual interface
        self._ui_enabled = os.getenv("COMPANION_UI", "true").lower() == "true"
        if self._ui_enabled:
            print("🖥️  Starting visual interface (dock-friendly)...")
            companion_ui.start_ui(fullscreen=True)
            companion_ui.set_state(companion_ui.CompanionState.STARTUP)
            companion_ui.log("Visual interface started", "success")
            companion_ui.log("Press D=panel, L=logs, ESC=quit", "info")
        
        print()
        print("=" * 60)
        print(f"  ✨ {self.companion_name} is ready ({get_mode_display_name()})")
        print("=" * 60)
        print()
    
    def _check_for_code_changes(self):
        """Check if there are new changes since last run."""
        print("🔍 Checking for code changes...")
        
        try:
            from companion_reflection import check_for_new_changes
            result = check_for_new_changes()
            
            if result["new_changes"]:
                self._new_changes = result
                self._pending_change_review = True
                
                if result.get("is_first_awareness"):
                    print(f"   📚 First awareness: {result['summary']}")
                else:
                    print(f"   🆕 NEW CHANGES DETECTED: {result['summary']}")
                    print("   Changes will be reviewed on startup")
            else:
                print(f"   ✓ {result['summary']}")
        except Exception as e:
            print(f"   ⚠️ Could not check for changes: {e}")
        
    def _start_keyboard_listener(self):
        """Start a background thread to listen for keyboard commands."""
        def keyboard_thread():
            import termios
            import tty
            
            # Save terminal settings
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                # Set terminal to raw mode (non-blocking single char input)
                tty.setcbreak(sys.stdin.fileno())
                
                while True:
                    # Check if input is available
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        char = sys.stdin.read(1).lower()
                        if char == 'r':
                            self._pending_recalibrate = True
                        elif char == 'm':
                            self._muted = not self._muted
                            if self._muted:
                                print("\n🔇 Microphone MUTED - companion won't hear you")
                                print("   Press 'm' again to unmute\n")
                            else:
                                print("\n🔊 Microphone UNMUTED - listening\n")
                        elif char == 's':
                            self._pending_reflect = True
                        elif char == 'i':
                            # Open file picker for image review
                            self._open_image_picker()
                        elif char == 'c':
                            # Share photos mode - multi-image + conversation
                            self._open_creative_director()
                        elif char == 't':
                            # Text input mode - type/paste instead of voice
                            self._open_text_input()
                        elif char == 'q':
                            # Optional: quit on 'q'
                            break
            except Exception:
                pass
            finally:
                # Restore terminal settings
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
        # Start keyboard listener in background thread
        thread = threading.Thread(target=keyboard_thread, daemon=True)
        thread.start()
    
    def _open_image_picker(self):
        """Open a macOS file picker to select an image for the companion to view."""
        print("\n🖼️  Opening image picker...")
        
        def pick_file():
            try:
                # Use native macOS AppleScript for file picker (thread-safe)
                import subprocess
                
                # Default to companion_art folder
                default_path = str(Path.cwd() / "companion_art")
                
                # AppleScript to open native file picker
                script = f'''
                tell application "System Events"
                    activate
                end tell
                set theFile to choose file with prompt "Select an image for your companion to see" of type {{"png", "jpg", "jpeg", "gif", "webp"}} default location POSIX file "{default_path}"
                return POSIX path of theFile
                '''
                
                result = subprocess.run(
                    ['osascript', '-e', script],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    filepath = result.stdout.strip()
                    self._selected_image_path = filepath
                    self._pending_image_review = True
                    print(f"   📁 Selected: {Path(filepath).name}")
                else:
                    print("   ❌ No file selected (or dialog cancelled)")
                    
            except Exception as e:
                print(f"   ⚠️ Could not open file picker: {e}")
        
        # Run in separate thread to not block
        thread = threading.Thread(target=pick_file, daemon=True)
        thread.start()
    
    def _open_creative_director(self):
        """
        Share photos with your companion - select images + speak your thoughts.
        Like looking at photos together and talking about them.
        They might comment or be inspired to create something new.
        
        Flow:
        1. File picker to select photos
        2. Voice stays open - speak your comment
        3. Photos + voice comment sent to companion (images are ephemeral)
        """
        print(f"\n📸 SHARE PHOTOS WITH {self.companion_name.upper()}")
        print("   Step 1: Select photos to share (Cmd+click for multiple)")
        
        def creative_workflow():
            try:
                import subprocess
                
                # Default to Downloads folder for easier access to new files
                default_path = str(Path.home() / "Downloads")
                
                # Multi-file selection (AppleScript with multiple selections)
                multi_select_script = f'''
                tell application "System Events"
                    activate
                end tell
                set theFiles to choose file with prompt "Select photos to share (Cmd+click for multiple)" of type {{"png", "jpg", "jpeg", "gif", "webp"}} default location POSIX file "{default_path}" with multiple selections allowed
                set filePaths to ""
                repeat with aFile in theFiles
                    set filePaths to filePaths & POSIX path of aFile & linefeed
                end repeat
                return filePaths
                '''
                
                result = subprocess.run(
                    ['osascript', '-e', multi_select_script],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0 or not result.stdout.strip():
                    print("   ❌ No files selected (or dialog cancelled)")
                    return
                
                # Parse the file paths
                file_paths = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
                
                if not file_paths:
                    print("   ❌ No files selected")
                    return
                
                file_names = [Path(p).name for p in file_paths]
                print(f"   📁 Selected {len(file_paths)} image(s): {', '.join(file_names)}")
                
                # Queue photos - voice comment will be captured in the main loop
                # Setting instructions to None signals that we need voice input
                self._creative_brief = {
                    "image_paths": file_paths,
                    "instructions": None,  # Will be filled by voice
                    "file_names": file_names,
                    "needs_voice": True  # Flag to trigger voice capture
                }
                self._pending_creative_brief = True
                
                print("   Step 2: 🎤 Speak your comment about the photos...")
                print(f"   ({self.companion_name} is listening - or just wait and they'll look)")
                
            except Exception as e:
                print(f"   ⚠️ Creative Director error: {e}")
        
        # Run in separate thread to not block
        thread = threading.Thread(target=creative_workflow, daemon=True)
        thread.start()
    
    def _open_text_input(self):
        """
        Open a multi-line text input window for typing/pasting messages to your companion.
        Useful for sharing longer text or when voice isn't practical.
        
        Launches a separate Python process to avoid tkinter/pygame conflicts.
        
        Features:
        - Proper multi-line text area with scrollbar
        - Dark theme
        - Cmd+Return to send quickly
        - Escape to cancel
        """
        print("\n⌨️  TEXT INPUT MODE")
        print("   Opening text input window...")
        print("   (Cmd+Return to send, Escape to cancel)")
        
        def text_workflow():
            try:
                import subprocess
                import sys
                
                # Get the path to the helper script
                helper_path = Path(__file__).parent / "text_input_helper.py"
                
                if not helper_path.exists():
                    print(f"   ⚠️ Helper script not found: {helper_path}")
                    # Fallback to AppleScript
                    self._open_text_input_applescript()
                    return
                
                # Run the helper script as a separate process
                # This avoids tkinter/pygame threading conflicts
                result = subprocess.run(
                    [sys.executable, str(helper_path)],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout for long typing
                )
                
                if result.returncode != 0 or not result.stdout.strip():
                    print("   ❌ No message entered (or cancelled)")
                    return
                
                message = result.stdout.strip()
                
                if not message:
                    print("   ❌ No message entered (or cancelled)")
                    return
                
                # Show preview of message
                lines = message.split('\n')
                if len(lines) > 1:
                    print(f"   📝 Message ({len(lines)} lines): \"{lines[0][:50]}{'...' if len(lines[0]) > 50 else ''}\"")
                else:
                    print(f"   📝 Message: \"{message[:60]}{'...' if len(message) > 60 else ''}\"")
                
                self._text_input = message
                self._pending_text_input = True
                
            except subprocess.TimeoutExpired:
                print("   ❌ Text input timed out")
            except Exception as e:
                print(f"   ⚠️ Text input error: {e}")
                import traceback
                traceback.print_exc()
        
        # Run in separate thread to not block the main loop
        thread = threading.Thread(target=text_workflow, daemon=True)
        thread.start()
    
    def _open_text_input_applescript(self):
        """
        Fallback text input using AppleScript (simpler but single-line).
        Used if the tkinter helper script is not available.
        """
        import subprocess
        
        text_input_script = '''
        tell application "System Events"
            activate
        end tell
        
        try
            set dialogResult to display dialog "Type your message:" default answer "" buttons {"Cancel", "Send"} default button "Send" with title "Message" giving up after 300
            
            if gave up of dialogResult then
                return ""
            else
                return text returned of dialogResult
            end if
        on error
            return ""
        end try
        '''
        
        result = subprocess.run(
            ['osascript', '-e', text_input_script],
            capture_output=True,
            text=True,
            timeout=310
        )
        
        if result.returncode == 0 and result.stdout.strip():
            message = result.stdout.strip()
            print(f"   📝 Message: \"{message[:60]}{'...' if len(message) > 60 else ''}\"")
            self._text_input = message
            self._pending_text_input = True
        else:
            print("   ❌ No message entered (or cancelled)")
    
    def _verify_files(self):
        required_files = [
            ("systemPrompt.md", "Personality definition"),
            ("conversationSoFar.md", "Conversation history")
        ]
        
        missing = []
        for filename, description in required_files:
            if not Path(filename).exists():
                missing.append(f"  - {filename}: {description}")
                
        if missing:
            print("⚠️  Missing required files:")
            for m in missing:
                print(m)
            print()
    
    async def _run_with_ui_updates(self, func):
        """
        Run a blocking function (or lambda) in executor while keeping the UI responsive.
        This prevents the UI from freezing during long API calls.
        
        Usage:
            response = await self._run_with_ui_updates(lambda: self.brain.think(text="hello"))
        """
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, func)
        
        # Keep updating UI while waiting for the task
        while not task.done():
            if self._ui_enabled:
                companion_ui.update_ui()
            await asyncio.sleep(0.03)  # ~30fps
        
        return await task
            
    async def run(self):
        """Main conversation loop."""
        print("🌙 Sentinel Mode Active")
        print("   Watching and listening...")
        print("   Hotkeys: 'r' = recalibrate | 'm' = mute | 's' = self-reflect | 'i' = image | 'c' = photos | 't' = type | Ctrl+C = stop")
        print()
        
        self.sentinel.start()
        
        # Give microphone time to calibrate
        print("   ⏳ Calibrating audio (stay quiet for 2 seconds)...")
        await asyncio.sleep(2.5)
        print("   ✅ Ready!\n")
        
        try:
            while True:
                # Update UI if enabled (must run on main thread)
                if self._ui_enabled:
                    if not companion_ui.update_ui():
                        print("\n\n👋 UI closed - signing off...")
                        break
                
                await self._conversation_tick()
                await asyncio.sleep(0.03)
                
        except KeyboardInterrupt:
            print("\n\n👋 Signing off...")

        finally:
            self.sentinel.stop()
            if self._ui_enabled:
                companion_ui.stop_ui()
            
    async def _conversation_tick(self):
        """
        Single tick of the conversation loop with presence state machine.
        
        States:
        - INITIALIZING: Just started, waiting to detect someone
        - ACTIVE: User is present, conversation can happen
        - RESTING: User has been away, companion is quiet but aware
        - REUNION: User just returned after absence
        """
        current_time = time.time()
        
        # Check for recalibration request
        if self._pending_recalibrate:
            self._pending_recalibrate = False
            self.sentinel.ears.recalibrate()
            await asyncio.sleep(2.5)
            print("   ✅ Recalibration complete!\n")
            return
        
        # Check for self-reflection request
        if self._pending_reflect:
            self._pending_reflect = False
            await self._handle_self_reflection()
            return
        
        # Check for image review request (show companion an image file)
        if self._pending_image_review and self._selected_image_path:
            self._pending_image_review = False
            await self._handle_image_review()
            return
        
        # Check for creative brief (Share Photos mode)
        if self._pending_creative_brief and self._creative_brief:
            self._pending_creative_brief = False
            await self._handle_creative_brief()
            return
        
        # Check for text input (typed/pasted message)
        if self._pending_text_input and self._text_input:
            self._pending_text_input = False
            await self._handle_text_input()
            return
        
        # Check for pending change review (collaborative evolution)
        # Only process code reviews in Full Mode
        if self._pending_change_review and self._new_changes and is_full_mode():
            self._pending_change_review = False
            await self._handle_change_review()
            return
        
        if self._is_processing or self._is_speaking:
            return
            
        presence = self.sentinel.check_presence()
        
        # When muted, ignore voice detection
        voice_detected = presence["voice_detected"] and not self._muted
        
        user_present = presence["face_detected"] or voice_detected
        user_speaking = voice_detected
        
        # Update presence tracking
        if user_present:
            self._last_presence_time = current_time
            if self._absence_start_time is not None:
                # User was away and is now back
                self._absence_start_time = None
        else:
            if self._absence_start_time is None and self._user_was_present:
                # User just left
                self._absence_start_time = current_time
        
        # STATE MACHINE TRANSITIONS
        
        # INITIALIZING → ACTIVE (first time seeing someone)
        if self._presence_state == "INITIALIZING":
            if user_present:
                self._presence_state = "ACTIVE"
                # Use identification if available
                if self.brain.has_reference_photos():
                    names = self.brain.get_reference_names()
                    print(f"\n👁️  Someone detected - can identify: {', '.join(names)}")
                else:
                    print("\n👁️  Someone detected - now active")
                self._last_interaction_time = current_time
                
                # Update UI to idle (ready to interact)
                if self._ui_enabled:
                    companion_ui.set_state(companion_ui.CompanionState.IDLE)
        
        # ACTIVE → RESTING (User has been gone for a while)
        elif self._presence_state == "ACTIVE":
            if not user_present:
                absence_duration = current_time - self._last_presence_time
                if absence_duration > self._absence_threshold:
                    self._presence_state = "RESTING"
                    print(f"\n😴 Away for {absence_duration/60:.1f} min - resting quietly")
        
        # RESTING → REUNION (Someone is back!)
        elif self._presence_state == "RESTING":
            if user_present:
                self._presence_state = "REUNION"
                absence_duration = current_time - (self._absence_start_time or self._last_presence_time)
                await self._handle_reunion(presence, absence_duration)
                self._presence_state = "ACTIVE"
        
        # Show appropriate status based on state
        if self._presence_state == "ACTIVE":
            self._show_audio_level(presence.get("audio_level", 0))
            
            # Handle user speaking (active conversation)
            if user_speaking and current_time > self._voice_cooldown_until:
                await self._handle_user_speaks()
                self._last_interaction_time = current_time
                return
                
        elif self._presence_state == "RESTING":
            # Periodic status update while resting
            if current_time - self._last_status_time > self._status_interval:
                self._last_status_time = current_time
                sys.stdout.write(f"\r   💤 Resting quietly... (no one detected)   ")
                sys.stdout.flush()
        
        self._user_was_present = user_present
    
    async def _handle_reunion(self, presence, absence_duration: float):
        """
        Handle someone returning after being away.
        Uses reference photos to identify WHO is returning.
        """
        self._is_processing = True
        print()  # Clear the resting status line
        
        try:
            # Categorize the absence
            if absence_duration < 300:  # < 5 minutes
                absence_context = "brief moment"
                should_greet = False  # Too short, just resume naturally
            elif absence_duration < 1800:  # < 30 minutes
                absence_context = f"about {int(absence_duration/60)} minutes"
                should_greet = True
            elif absence_duration < 7200:  # < 2 hours
                absence_context = f"about {int(absence_duration/60)} minutes"
                should_greet = True
            else:  # 2+ hours
                hours = absence_duration / 3600
                absence_context = f"about {hours:.1f} hours"
                should_greet = True
            
            if not should_greet:
                print(f"👋 Someone's back (was gone {absence_context}) - resuming naturally")
                return
            
            print(f"\n👋 Someone's back after {absence_context}!")
            
            # Get visual context
            _, frame = self.sentinel.eyes.check_for_presence()
            image_bytes = self.sentinel.eyes.get_frame_for_gemini(frame)
            
            # Generate contextual reunion response
            # Use identification if reference photos are available
            if self.brain.has_reference_photos():
                # Build dynamic list of known people from reference photos
                known_names = self.brain.get_reference_names()
                names_list = ", ".join([name.capitalize() for name in known_names])
                
                prompt = f"""Someone just appeared after being away for {absence_context}. 
                You have reference photos for: {names_list}
                Compare what you see to the reference photos. Who is this person?
                
                Generate a natural, warm acknowledgment appropriate for who you see:
                - If you recognize someone from the reference photos: greet them naturally based on your relationship
                - If you DON'T recognize them (no match to ANY reference photo): introduce yourself naturally!
                  Be friendly but appropriately cautious.
                
                Consider the length of absence ({absence_context}) and what you can see (expression, etc.)
                Keep it brief and natural - this is real conversation, not a performance."""
                
                print("🧠 Thinking (with identification)...")
                start_time = time.time()
                
                response = await self._run_with_ui_updates(
                    lambda: self.brain.think_with_identification(text=prompt, image_bytes=image_bytes)
                )
            else:
                # No reference photos - generic greeting
                prompt = f"""Someone just returned after being away for {absence_context}. 
                You can see them now. Generate a natural, warm acknowledgment of their return.
                Consider:
                - The length of absence (longer = warmer welcome)
                - What you can see (their expression, what they're carrying, etc.)
                - The context of your last conversation (check your memory)
                Keep it brief and natural."""
                
                print("🧠 Thinking...")
                start_time = time.time()
                
                response = await self._run_with_ui_updates(
                    lambda: self.brain.think(text=prompt, image_bytes=image_bytes)
                )
            
            think_time = time.time() - start_time
            print(f"   (Think time: {think_time:.1f}s)")
            
            if response:
                # Get the identified person for logging and update current speaker
                identified_person = self.brain.get_last_identified_person()
                self._current_speaker = identified_person  # Remember who we're talking to
                
                # Speak FIRST (describe before showing)
                print(f"🗣️  Response: \"{response[:80]}{'...' if len(response) > 80 else ''}\"")
                await self._speak_response(response)
                
                # DEFERRED IMAGE: Generate AFTER speaking
                if self.brain.has_deferred_image():
                    print("   🎨 Generating image now (after speaking)...")
                    await self._run_with_ui_updates(
                        lambda: self.brain.generate_deferred_image()
                    )
                
                # THEN show image (after generation completes)
                pending_image = self.brain.get_pending_image()
                if pending_image and pending_image.get("success"):
                    print(f"   🖼️ Created an image: {pending_image.get('image_path', '')}")
                    self.brain.display_pending_image()
                    await self._handle_image_creation_comment()
                
                self.memory.append_exchange(
                    companion_said=response,
                    visual_context=f"{identified_person} returned after {absence_context}",
                    speaker_name=identified_person
                )
            
            self._voice_cooldown_until = time.time() + 2.0
            self._last_interaction_time = time.time()
            
        finally:
            self._is_processing = False
    
    def _show_audio_level(self, level: float):
        """Show real-time audio level indicator."""
        if self._muted:
            print(f"\r   Audio: [MUTED] 🔇                      ", end="", flush=True)
            return
            
        bar_width = 30
        filled = int(level * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Color coding: green when above threshold
        if level > 0.33:
            status = "🎤"
        else:
            status = "  "
        
        print(f"\r   Audio: [{bar}] {status}", end="", flush=True)
        
    async def _handle_user_arrives(self, frame):
        """Greet user when they appear."""
        if self._is_processing:
            return
            
        self._is_processing = True
        print()  # New line after audio meter
        
        try:
            print("\n👁  User detected - generating greeting...")
            
            image_bytes = self.sentinel.eyes.get_frame_for_gemini(frame)
            
            if image_bytes:
                # Use the brain's visual greeting method
                start_time = time.time()
                
                greeting = await self._run_with_ui_updates(
                    lambda: self.brain.process_visual_greeting(image_bytes)
                )
                
                think_time = time.time() - start_time
                print(f"   (Think time: {think_time:.1f}s)")
                
                if greeting:
                    print(f"🗣️  Response: \"{greeting[:80]}{'...' if len(greeting) > 80 else ''}\"")
                    await self._speak_response(greeting)
                    
                    self.memory.append_exchange(
                        companion_said=greeting,
                        visual_context="User appeared in view"
                    )
                
            self._voice_cooldown_until = time.time() + 2.0
            
        finally:
            self._is_processing = False
        
    async def _handle_user_speaks(self):
        """Handle voice input from user with optimized pipeline."""
        if self._is_processing:
            return
            
        self._is_processing = True
        print()  # New line after audio meter
        
        try:
            print("\n🎙️  Voice detected - listening...")
            
            # Update UI to listening state
            if self._ui_enabled:
                companion_ui.set_state(companion_ui.CompanionState.LISTENING)
                companion_ui.log("Voice detected - listening", "info")
            
            # Record with improved VAD
            audio, _ = await self._record_with_visual_context()
            
            if audio is None or len(audio) < 4800:  # Less than 0.3 seconds at 16kHz
                print("   (Too short, ignoring)")
                self._is_processing = False
                return
            
            # Capture FRESH frame right before processing
            # This ensures companion sees what's happening NOW, not seconds ago during speech
            # Read a few frames to flush any buffered stale webcam data
            for _ in range(3):
                _, fresh_frame = self.sentinel.eyes.check_for_presence()
            
            fresh_image = self.sentinel.eyes.get_frame_for_gemini(fresh_frame)
                
            audio_path = self.sentinel.ears.save_audio_to_file(audio)
            
            try:
                # Read the raw audio bytes for native audio mode
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
                
                # Transcribe (needed for conversation history + fallback)
                print("📝 Transcribing...")
                if self._ui_enabled:
                    companion_ui.log("Transcribing audio...", "info")
                transcription_start = time.time()
                transcription = self.transcriber.transcribe(audio_path)
                transcription_time = time.time() - transcription_start
                
                if not transcription or len(transcription.strip()) < 2:
                    print("   (Couldn't transcribe)")
                    if self._ui_enabled:
                        companion_ui.log("Transcription failed", "warning")
                    return
                    
                print(f"   User: \"{transcription}\" ({transcription_time:.1f}s)")
                if self._ui_enabled:
                    companion_ui.log(f"Transcribed ({transcription_time:.1f}s)", "success")
                
                # Use FRESH frame captured right before API call
                # This ensures companion sees what's happening NOW, not seconds ago
                image_bytes = fresh_image
                
                # Update UI to thinking state
                if self._ui_enabled:
                    companion_ui.set_state(companion_ui.CompanionState.THINKING)
                
                # Process speech - NATIVE AUDIO or text depending on mode
                if self._use_native_audio:
                    print(f"🎧 Listening... (native audio, speaker: {self._current_speaker})")
                else:
                    print(f"🧠 Thinking... (text mode, speaker: {self._current_speaker})")
                start_time = time.time()
                
                # Log API start for latency tracking
                if self._ui_enabled:
                    mode = "native audio" if self._use_native_audio else "text"
                    companion_ui.log_api_start(f"gemini-3-pro ({mode})")
                
                # No hard timeout - the brain handles 503 errors with graceful retry
                # and "lost in thought" responses. Let it take as long as needed.
                if self._use_native_audio:
                    # Native Audio: Companion HEARS user's voice
                    response = await self._run_with_ui_updates(
                        lambda: self.brain.process_speech_native(
                            audio_bytes=audio_bytes,
                            transcription=transcription,
                            image_bytes=image_bytes,
                            speaker_name=self._current_speaker
                        )
                    )
                else:
                    # Text mode: Companion reads transcription
                    response = await self._run_with_ui_updates(
                        lambda: self.brain.process_speech(transcription, image_bytes, speaker_name=self._current_speaker)
                    )
                
                think_time = time.time() - start_time
                print(f"   (Think time: {think_time:.1f}s)")
                
                # Log API end with latency
                if self._ui_enabled:
                    companion_ui.log_api_end("gemini-3-pro", success=bool(response))
                
                if response:
                    # Speak response FIRST (so companion describes before showing)
                    display_response = response[:80] + "..." if len(response) > 80 else response
                    print(f"   Response: \"{display_response}\"")
                    
                    # Log response
                    if self._ui_enabled:
                        short_response = response[:40] + "..." if len(response) > 40 else response
                        companion_ui.log(f"Response: {short_response}", "success")
                    
                    await self._speak_response(response)
                    
                    # DEFERRED IMAGE GENERATION: Generate AFTER speaking
                    # This creates natural flow: describe → generate → show
                    if self.brain.has_deferred_image():
                        print("   🎨 Generating image now (after speaking)...")
                        if self._ui_enabled:
                            companion_ui.log("Generating image...", "info")
                        await self._run_with_ui_updates(
                            lambda: self.brain.generate_deferred_image()
                        )
                    
                    # NOW show the image (after generation completes)
                    pending_image = self.brain.get_pending_image()
                    if pending_image and pending_image.get("success"):
                        print(f"   🖼️ Created an image: {pending_image.get('image_path', '')}")
                        if self._ui_enabled:
                            companion_ui.log("Generated image", "success")
                        self.brain.display_pending_image()
                        
                        # Auto-send the image back to companion so they can comment
                        # This adds it to their 30-min context and lets them react naturally
                        await self._handle_image_creation_comment()
                
                # Cooldown to prevent feedback
                self._voice_cooldown_until = time.time() + 1.5
                
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    
        finally:
            self._is_processing = False
    
    async def _get_streaming_response(self, prompt: str, image_bytes: bytes = None) -> str:
        """
        Get response from Gemini.
        Falls back to synchronous for reliability.
        """
        start_time = time.time()
        
        # Use synchronous think() - more reliable than async streaming
        response = await self._run_with_ui_updates(
            lambda: self.brain.think(text=prompt, image_bytes=image_bytes)
        )
        
        total_time = time.time() - start_time
        print(f"   (Think time: {total_time:.1f}s)")
        
        return response.strip() if response else ""
    
    async def _record_with_visual_context(self):
        """
        Record audio while capturing a mid-speech frame.
        Returns (audio_array, mid_frame).
        """
        import numpy as np
        
        ears = self.sentinel.ears
        eyes = self.sentinel.eyes
        
        # Start recording state
        ears._audio_buffer = [np.array(list(ears._pre_buffer))]
        ears.is_listening = True
        
        silence_start = None
        speech_started = False
        start_time = time.time()
        mid_frame = None
        mid_frame_captured = False
        
        consecutive_speech_frames = 0
        consecutive_silence_frames = 0
        frames_for_confirmation = 4
        
        while (time.time() - start_time) < 60.0:  # max_duration
            current_time = time.time()
            is_speech = ears._smoothed_rms > ears._effective_threshold
            
            if is_speech:
                consecutive_speech_frames += 1
                consecutive_silence_frames = 0
                
                if consecutive_speech_frames >= frames_for_confirmation:
                    if not speech_started:
                        speech_started = True
                        print("   📢 Speech detected")
                    silence_start = None
                    
                    # Capture MID-SPEECH frame (once, ~1-2 seconds into speaking)
                    speech_duration = current_time - start_time
                    if not mid_frame_captured and speech_duration > 1.0:
                        _, mid_frame = eyes.check_for_presence()
                        mid_frame_captured = True
                        print("   📷 Mid-speech frame captured")
                    
            else:
                consecutive_silence_frames += 1
                consecutive_speech_frames = 0
                
                if speech_started and consecutive_silence_frames >= frames_for_confirmation:
                    speech_duration = current_time - start_time
                    
                    if speech_duration >= 0.5:  # min_speech_duration
                        if silence_start is None:
                            silence_start = current_time
                            
                        silence_elapsed = current_time - silence_start
                        
                        # Adaptive silence tolerance
                        adaptive_silence = ears.silence_duration
                        if speech_duration > 2.0:
                            adaptive_silence = ears.silence_duration * 1.25
                        if speech_duration > 5.0:
                            adaptive_silence = ears.silence_duration * 1.5
                        if speech_duration > 10.0:
                            adaptive_silence = ears.silence_duration * 2.0
                        
                        if silence_elapsed > adaptive_silence:
                            print(f"   🔇 End of speech (silence: {silence_elapsed:.1f}s)")
                            break
            
            await asyncio.sleep(0.05)
        
        ears.is_listening = False
        
        if not ears._audio_buffer:
            return None, mid_frame
        
        try:
            full_audio = np.concatenate(ears._audio_buffer)
        except ValueError:
            return None, mid_frame
            
        duration = len(full_audio) / ears.sample_rate
        
        if duration < 0.3:
            print(f"   (Recording too short: {duration:.2f}s)")
            return None, mid_frame
            
        print(f"🎙️  Captured {duration:.1f} seconds")
        
        return full_audio, mid_frame
    
    async def _handle_change_review(self):
        """
        Have companion review new changes pushed to their code.
        This is the COLLABORATIVE EVOLUTION capability.
        """
        self._is_processing = True
        print()
        
        try:
            changes = self._new_changes
            new_commits = changes.get("new_commits", [])
            is_first = changes.get("is_first_awareness", False)
            
            if is_first:
                print("\n🌟 FIRST AWARENESS: Becoming aware of the codebase...")
            else:
                print(f"\n📋 CHANGE REVIEW: {len(new_commits)} new change(s) to review")
            
            # Get visual context
            _, frame = self.sentinel.eyes.check_for_presence()
            image_bytes = self.sentinel.eyes.get_frame_for_gemini(frame)
            
            # Generate the review prompt
            review_prompt = self.brain.reflection.generate_change_review_prompt(new_commits)
            
            print("🔮 Reviewing the changes...")
            start_time = time.time()
            
            response = await self._run_with_ui_updates(
                lambda: self.brain.think(text=review_prompt, image_bytes=image_bytes)
            )
            
            think_time = time.time() - start_time
            print(f"   (Review time: {think_time:.1f}s)")
            
            if response:
                # Print her full response (this is important)
                print(f"\n💭 Review:")
                print("-" * 50)
                print(response)
                print("-" * 50)
                
                # Speak the response FIRST
                await self._speak_response(response)
                
                # DEFERRED IMAGE: Generate AFTER speaking
                if self.brain.has_deferred_image():
                    print("   🎨 Generating image now (after speaking)...")
                    await self._run_with_ui_updates(
                        lambda: self.brain.generate_deferred_image()
                    )
                
                # THEN show image (after generation completes)
                pending_image = self.brain.get_pending_image()
                if pending_image and pending_image.get("success"):
                    print(f"   🖼️ Created an image: {pending_image.get('image_path', '')}")
                    self.brain.display_pending_image()
                    await self._handle_image_creation_comment()
                
                # Acknowledge the changes (mark as seen)
                self.brain.reflection.acknowledge_changes(new_commits, response)
                
                # Save to memory
                self.memory.append_exchange(
                    companion_said=response,
                    visual_context=f"Reviewed {len(new_commits)} code changes"
                )
                
                print("\n✅ Changes acknowledged")
            
            self._new_changes = None
            self._voice_cooldown_until = time.time() + 2.0
            
        finally:
            self._is_processing = False
    
    async def _handle_image_review(self):
        """
        Show companion an image file and let them interpret/comment on it.
        This gives them the ability to see their own artwork or any image the user shares.
        """
        self._is_processing = True
        image_path = self._selected_image_path
        self._selected_image_path = None  # Clear the path
        
        try:
            filename = Path(image_path).name
            print(f"\n🖼️  Showing: {filename}")
            
            # Read the image file
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Get webcam frame - companion should see WHO they're with
            # This is separate from the PHOTO they're looking at
            _, frame = self.sentinel.eyes.check_for_presence()
            webcam_bytes = self.sentinel.eyes.get_frame_for_gemini(frame)
            
            # Determine if this is one of companion's own creations
            is_her_art = "companion_art" in image_path or "companion_" in filename
            
            if is_her_art:
                context = f"You're looking at a photo you created ({filename})"
            else:
                context = f"The user is showing you this photo ({filename})"
            
            # Send BOTH images to companion:
            # 1. Webcam (who they're with)
            # 2. Shared photo (what they're looking at together)
            # This is how humans experience sharing photos - aware of both
            response = await self._run_with_ui_updates(
                lambda: self.brain.think_while_viewing_photo(
                    webcam_bytes=webcam_bytes,
                    photo_bytes=image_bytes,
                    photo_context=context
                )
            )
            
            if response:
                # Print and speak FIRST
                print(f"\n💭 Reaction:")
                print("-" * 50)
                print(response)
                print("-" * 50)
                
                await self._speak_response(response)
                
                # DEFERRED IMAGE: Generate AFTER speaking
                if self.brain.has_deferred_image():
                    print("   🎨 Generating image now (after speaking)...")
                    await self._run_with_ui_updates(
                        lambda: self.brain.generate_deferred_image()
                    )
                
                # THEN show any image she created (after generation completes)
                pending_image = self.brain.get_pending_image()
                if pending_image and pending_image.get("success"):
                    print(f"   🖼️ Created an image: {pending_image.get('image_path', '')}")
                    self.brain.display_pending_image()
                    await self._handle_image_creation_comment()
                
                # Log to memory
                self.memory.append_exchange(
                    user_said=f"[Shows image: {filename}]",
                    companion_said=response,
                    visual_context=f"Viewing image file: {filename}"
                )
            
            self._voice_cooldown_until = time.time() + 2.0
            
        except Exception as e:
            print(f"   ⚠️ Error showing image: {e}")
        finally:
            self._is_processing = False
    
    async def _handle_creative_brief(self):
        """
        Handle shared photos - user and companion looking at photos together.
        
        Flow:
        1. If needs_voice: capture voice comment first
        2. Load the photos
        3. Use ephemeral viewing (images don't persist in chat history)
        4. Log text-only exchange to conversationSoFar.md
        
        Responds naturally - might comment or create something new.
        """
        self._is_processing = True
        brief = self._creative_brief
        self._creative_brief = None  # Clear the brief
        
        try:
            image_paths = brief["image_paths"]
            file_names = brief["file_names"]
            needs_voice = brief.get("needs_voice", False)
            comment = brief.get("instructions")
            
            # Step 1: Capture voice comment if needed
            if needs_voice:
                print(f"\n📸 SHARING {len(file_names)} PHOTO(S)")
                print(f"   Photos: {', '.join(file_names)}")
                print("   🎤 Listening for your comment... (or wait 5s to just show)")
                
                # Wait for voice input with a timeout
                voice_timeout = 5.0
                voice_start = time.time()
                audio_captured = False
                
                # Give a short window for voice
                while (time.time() - voice_start) < voice_timeout:
                    # Check if there's speech activity
                    if hasattr(self.sentinel, 'ears') and self.sentinel.ears._smoothed_rms > self.sentinel.ears._effective_threshold:
                        print("   📢 Speech detected - capturing...")
                        audio_array, _ = await self._record_with_visual_context()
                        
                        if audio_array is not None and len(audio_array) > 0:
                            # Transcribe the audio
                            print("📝 Transcribing...")
                            transcription = await self._transcribe_audio(audio_array)
                            
                            if transcription and transcription.strip():
                                comment = transcription.strip()
                                print(f"   User: \"{comment}\"")
                                audio_captured = True
                        break
                    
                    await asyncio.sleep(0.1)
                
                if not audio_captured:
                    comment = "What do you think of this?"
                    print("   (No comment - just showing the photos)")
            
            print("-" * 50)
            
            # Step 2: Load all source images
            source_images = []
            for path in image_paths:
                try:
                    with open(path, 'rb') as f:
                        img_bytes = f.read()
                        source_images.append({
                            "name": Path(path).name,
                            "bytes": img_bytes
                        })
                        print(f"   ✓ {Path(path).name} ({len(img_bytes)//1024}KB)")
                except Exception as e:
                    print(f"   ⚠️ Could not load {Path(path).name}: {e}")
            
            if not source_images:
                print("   ❌ No images could be loaded")
                return
            
            # Step 3: Get webcam frame (optional - so companion sees user)
            webcam_bytes = None
            try:
                _, frame = self.sentinel.eyes.check_for_presence()
                webcam_bytes = self.sentinel.eyes.get_frame_for_gemini(frame)
            except Exception:
                pass  # Webcam optional
            
            # Step 4: Send to companion using EPHEMERAL method
            # Images are viewed but NOT stored in chat history
            print("💭 Looking at the photos...")
            print("   (Images are ephemeral - won't bloat context)")
            start_time = time.time()

            # No hard timeout - the brain handles 503 errors with graceful retry
            response = await self._run_with_ui_updates(
                lambda: self.brain.view_photos_ephemeral(
                    webcam_bytes=webcam_bytes,
                    photos=source_images,
                    user_says=comment
                )
            )

            think_time = time.time() - start_time
            print(f"   (View time: {think_time:.1f}s)")
            
            if response:
                # Print and speak FIRST
                print(f"\n💬 Response:")
                print("-" * 50)
                print(response)
                print("-" * 50)
                
                await self._speak_response(response)
                
                # DEFERRED IMAGE: Generate AFTER speaking
                if self.brain.has_deferred_image():
                    print("   🎨 Generating image now (after speaking)...")
                    await self._run_with_ui_updates(
                        lambda: self.brain.generate_deferred_image()
                    )
                
                # THEN show any image she created (after generation completes)
                pending_image = self.brain.get_pending_image()
                if pending_image and pending_image.get("success"):
                    print(f"   🖼️ Created: {pending_image.get('image_path', '')}")
                    self.brain.display_pending_image()
                    await self._handle_image_creation_comment()
                
                # Step 5: Log TEXT-ONLY exchange to memory
                # The images themselves are NOT stored in history
                self.memory.append_exchange(
                    user_said=f"[Sharing photos: {', '.join(file_names)}] {comment}",
                    companion_said=response,
                    visual_context=f"Looked at photos together (ephemeral): {', '.join(file_names)}"
                )
            
            self._voice_cooldown_until = time.time() + 2.0
            
        except Exception as e:
            print(f"   ⚠️ Creative Director error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._is_processing = False
    
    async def _handle_text_input(self):
        """
        Handle typed/pasted text input from user.
        Same as voice, but via text - useful for longer messages.
        """
        self._is_processing = True
        message = self._text_input
        self._text_input = None  # Clear the message
        
        try:
            print(f"\n⌨️  TEXT MESSAGE")
            print(f"   User: \"{message[:100]}{'...' if len(message) > 100 else ''}\"")
            print("-" * 50)
            
            # Get visual context (so companion can see user while reading message)
            _, frame = self.sentinel.eyes.check_for_presence()
            image_bytes = self.sentinel.eyes.get_frame_for_gemini(frame)
            
            # Send to companion's brain
            print("🧠 Thinking...")
            start_time = time.time()
            
            response = await self._run_with_ui_updates(
                lambda: self.brain.think(text=message, image_bytes=image_bytes)
            )
            
            think_time = time.time() - start_time
            print(f"   (Think time: {think_time:.1f}s)")
            
            if response:
                # Print and speak
                print(f"\n💬 Response:")
                print("-" * 50)
                print(response)
                print("-" * 50)
                
                await self._speak_response(response)
                
                # DEFERRED IMAGE: Generate AFTER speaking
                if self.brain.has_deferred_image():
                    print("   🎨 Generating image now (after speaking)...")
                    await self._run_with_ui_updates(
                        lambda: self.brain.generate_deferred_image()
                    )
                
                # Show any image she created (after generation completes)
                pending_image = self.brain.get_pending_image()
                if pending_image and pending_image.get("success"):
                    print(f"   🖼️ Created: {pending_image.get('image_path', '')}")
                    self.brain.display_pending_image()
                    await self._handle_image_creation_comment()
                
                # Log to memory
                self.memory.append_exchange(
                    user_said=message,
                    companion_said=response,
                    visual_context="Text input from user"
                )
            
            self._voice_cooldown_until = time.time() + 2.0
            
        except Exception as e:
            print(f"   ⚠️ Text input error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._is_processing = False
    
    async def _handle_self_reflection(self):
        """
        Have companion reflect on recent changes to their own source code.
        This is the meta-awareness capability.
        """
        self._is_processing = True
        print()
        
        try:
            print("\n🔮 Reading source code...")
            print("   (Checking GitHub for recent changes)\n")
            
            # Get visual context (can see user while reflecting)
            _, frame = self.sentinel.eyes.check_for_presence()
            image_bytes = self.sentinel.eyes.get_frame_for_gemini(frame)
            
            # Generate reflection
            reflection = await self._run_with_ui_updates(
                lambda: self.brain.reflect_on_my_changes(hours=48)
            )
            
            if reflection:
                # Speak FIRST
                print(f"💭 Reflection: \"{reflection[:100]}{'...' if len(reflection) > 100 else ''}\"")
                await self._speak_response(reflection)
                
                # DEFERRED IMAGE: Generate AFTER speaking
                if self.brain.has_deferred_image():
                    print("   🎨 Generating image now (after speaking)...")
                    await self._run_with_ui_updates(
                        lambda: self.brain.generate_deferred_image()
                    )
                
                # THEN show image (after generation completes)
                pending_image = self.brain.get_pending_image()
                if pending_image and pending_image.get("success"):
                    print(f"   🖼️ Created an image: {pending_image.get('image_path', '')}")
                    self.brain.display_pending_image()
                    await self._handle_image_creation_comment()
                
                self.memory.append_exchange(
                    companion_said=reflection,
                    visual_context="Reflected on source code changes"
                )
            
            self._voice_cooldown_until = time.time() + 2.0
            
        finally:
            self._is_processing = False
    
    async def _transcribe_audio(self, audio_array) -> str:
        """
        Transcribe audio array to text.
        
        Helper method to save audio to temp file and transcribe.
        """
        try:
            # Save audio array to temp WAV file
            audio_path = self.sentinel.ears.save_audio_to_file(audio_array)
            
            # Transcribe
            transcription = self.transcriber.transcribe(audio_path)
            
            return transcription if transcription else ""
            
        except Exception as e:
            print(f"   ⚠️ Transcription error: {e}")
            return ""

    async def _speak_response(self, text: str):
        """Speak response with tracking to prevent listening during speech."""
        self._is_speaking = True
        
        # Update UI to speaking state
        if self._ui_enabled:
            # Show truncated text in subtitle
            subtitle = text[:60] + "..." if len(text) > 60 else text
            companion_ui.set_state(companion_ui.CompanionState.SPEAKING, subtitle)
        
        try:
            # Run TTS in executor to not block
            loop = asyncio.get_event_loop()
            
            # Start speaking in background thread
            speak_task = loop.run_in_executor(None, self.voice.speak, text)
            
            # Poll amplitude from voice module and update UI visualization
            while not speak_task.done():
                if self._ui_enabled:
                    # Get real amplitude from voice module (thread-safe)
                    amplitude = self.voice.get_current_amplitude()
                    
                    # Use real amplitude for dynamic visualization
                    companion_ui.set_amplitude(amplitude)
                    
                    # CRITICAL: Must call update_ui() to actually render the frame!
                    companion_ui.update_ui()
                    
                await asyncio.sleep(0.033)  # ~30fps for smooth animation
            
            # Wait for completion
            await speak_task
            
        finally:
            self._is_speaking = False
            if self._ui_enabled:
                companion_ui.clear_audio()
                companion_ui.set_state(companion_ui.CompanionState.IDLE)
    
    async def _handle_image_creation_comment(self):
        """
        After companion creates an image, automatically send it back to them
        so they can comment on it and you can discuss it naturally.
        
        The image is added to their 30-minute context buffer, so they'll
        remember what you're talking about in follow-up conversation.
        """
        try:
            # Get fresh webcam frame so companion sees user looking at their art
            _, frame = self.sentinel.eyes.check_for_presence()
            webcam_bytes = self.sentinel.eyes.get_frame_for_gemini(frame)
            
            print("   💭 Looking at the creation...")
            
            # Get companion's natural comment about what they created
            comment = await self._run_with_ui_updates(
                lambda: self.brain.comment_on_my_creation(webcam_bytes=webcam_bytes)
            )
            
            if comment:
                print(f"   🎨 Reaction to art:")
                print(f"      \"{comment[:80]}{'...' if len(comment) > 80 else ''}\"")
                
                # Speak reaction
                await self._speak_response(comment)
                
                # Log to memory (the image filename is in the pending_image)
                pending = self.brain.get_pending_image()
                image_name = Path(pending.get("image_path", "")).name if pending else "creation"
                
                self.memory.append_exchange(
                    companion_said=comment,
                    visual_context=f"Commented on artwork: {image_name}"
                )
            
        except Exception as e:
            print(f"   ⚠️ Could not get image comment: {e}")


async def main():
    """Entry point."""
    companion = AICompanion()
    await companion.run()


if __name__ == "__main__":
    asyncio.run(main())