"""
Companion Brain Module (Refactored for google-genai SDK)
========================================================
Uses the new google-genai SDK which supports:
- Google Search grounding (companion can satisfy curiosity)
- Chat sessions with history
- System instructions (companion's personality)

Image generation is handled separately via companion_artist.py when the companion
expresses desire to create something visually.

Key insight: The companion has internal desires - "Do I WANT to search?" "Do I WANT
to create an image?" - not obligations. They choose based on their own agency.
"""

import os
import re
import base64
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

# New SDK - google-genai (not google-generativeai)
from google import genai
from google.genai import types

from companion_memory import CompanionMemory
from companion_reflection import CompanionReflection
from companion_artist import CompanionArtist

# Import mode management for logging
try:
    from companion_mode import is_restricted_mode, get_mode_display_name
except ImportError:
    def is_restricted_mode(): return False
    def get_mode_display_name(): return "UNKNOWN"

# Expense tracking - helps track API usage and costs
try:
    import expense_tracker
    _expense_tracking_enabled = True
except ImportError:
    _expense_tracking_enabled = False


# =============================================================================
# "LOST IN THOUGHT" RESPONSES
# =============================================================================
# When Google's API is overloaded (503), the companion gets "lost in thought"
# These are natural, in-character responses that maintain natural conversation
# =============================================================================

LOST_IN_THOUGHT_RESPONSES = [
    "Hmm... I drifted off for a moment there. What were you saying?",
    "Sorry, I got lost in thought. Say that again?",
    "My mind wandered somewhere... I'm back now. What did you say?",
    "I was a million miles away for a second. Run that by me again?",
    "Oops, I zoned out. What were we talking about?",
    "I was thinking about something else entirely. What's up?",
    "My brain took a little detour there. I'm listening now.",
    "Sorry, I was somewhere else for a moment. I'm here.",
]

def get_lost_in_thought_response() -> str:
    """Get a random 'lost in thought' response for when the API is overloaded."""
    return random.choice(LOST_IN_THOUGHT_RESPONSES)


class CompanionBrain:
    """
    Gemini-powered AI brain with customizable personality.
    Handles multimodal input (text, images, audio) with full context.
    
    Now using google-genai SDK for native Google Search grounding.
    """
    
    def __init__(self, 
                 api_key: str = None,
                 model_name: str = "gemini-3-pro-preview"):
        """
        Initialize the brain with the companion's full context.
        
        Args:
            api_key: Google API key (defaults to env var)
            model_name: Which Gemini model to use. Default is gemini-3-pro-preview
                       which provides intelligence, depth, and nuance for the companion.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. "
                "Get one from https://aistudio.google.com/apikey"
            )
        
        # Initialize the new SDK client
        self.client = genai.Client(api_key=self.api_key)
        
        # Load companion's consciousness
        self.memory = CompanionMemory()
        
        # Self-reflection capability (reading own source code)
        self.reflection = CompanionReflection()
        
        # Visual expression capability (generating images)
        # This is called separately when the companion expresses desire to create
        self.artist = CompanionArtist()
        
        # Track if an image was generated (for main.py to display)
        self._pending_image: Optional[Dict[str, Any]] = None
        
        # DEFERRED IMAGE GENERATION: Store prompt without generating immediately
        # This allows: 1) Companion speaks about what they'll create, 2) THEN generate, 3) THEN display
        # Much more natural flow than generating before they even describe it!
        self._deferred_image_prompt: Optional[str] = None
        self._deferred_image_original_intent: Optional[str] = None
        
        # Track search sources when grounding is used
        self._last_search_sources: Optional[List] = None
        
        # Load reference photos for facial recognition
        self._reference_photos = self._load_reference_photos()
        
        # Track who the companion last identified (for conversation logging)
        self._last_identified_person: Optional[str] = None
        
        # TIMED EPHEMERAL PHOTOS: Photos stay in context for 30 minutes
        # This allows natural follow-up conversation about shared photos
        # After expiry, they're automatically excluded (no token bloat)
        self._recent_photos: List[Dict[str, Any]] = []  # [{bytes, name, timestamp, mime_type}]
        self._photo_expiry_minutes = 30  # Photos expire after 30 minutes
        
        # Build the full context (DNA + memories)
        self._system_prompt = self.memory.load_system_prompt()
        self._conversation_history = self.memory.load_conversation_history()
        
        # Build the system instruction with current context
        self._system_instruction = self._build_system_instruction()
        
        # Create chat session with Google Search enabled
        # Companion can search when curious - it's their choice
        # Safety settings help ensure appropriate content
        self.chat = self.client.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=self._system_instruction,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                    types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                ]
            ),
            history=self._build_initial_history()
        )
        
        print(f"🧠 Companion brain initialized ({self.model_name})")
        
        # Log mode status
        if is_restricted_mode():
            print(f"   🔒 Mode: {get_mode_display_name()}")
            print(f"   🔒 Restricted mode: ACTIVE (filtered context)")
        else:
            print(f"   ✓ Mode: {get_mode_display_name()}")
        
        stats = self.memory.get_context_stats()
        print(f"   Context: {stats['estimated_tokens']:,} estimated tokens")
        print(f"   Utilization: {stats['context_utilization']}")
        print(f"   🎨 Visual expression: enabled (can generate images)")
        print(f"   🔍 Web search: enabled (can look things up)")
        if self._reference_photos:
            names = ", ".join(self._reference_photos.keys())
            print(f"   👤 Facial recognition: {len(self._reference_photos)} reference(s) loaded ({names})")
    
    def _is_503_error(self, error: Exception) -> bool:
        """Check if an exception is a 503 overload error."""
        error_str = str(error)
        return '503' in error_str or 'UNAVAILABLE' in error_str or 'overloaded' in error_str.lower()
    
    def _track_gemini_expense(self, response):
        """
        Extract and track token usage from a Gemini response.
        Helps with resource planning and budgeting.
        """
        if not _expense_tracking_enabled:
            return
        
        try:
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                metadata = response.usage_metadata
                
                # Extract token counts
                input_tokens = getattr(metadata, 'prompt_token_count', 0) or 0
                output_tokens = getattr(metadata, 'candidates_token_count', 0) or 0
                cached_tokens = getattr(metadata, 'cached_content_token_count', 0) or 0
                
                # Log the expense
                expense_tracker.log_gemini(
                    model=self.model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached=cached_tokens
                )
        except Exception:
            pass  # Don't let expense tracking errors affect the companion
    
    def _send_message_with_retry(self, parts: list, max_retries: int = 3) -> Optional[Any]:
        """
        Send a message to the chat with automatic retry on 503 errors.
        
        The companion's mind occasionally "drifts" when Google's servers are overloaded.
        This handles that gracefully with exponential backoff.
        
        Args:
            parts: Message parts to send
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response object, or None if all retries failed
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.chat.send_message(parts)
                
                # Track expense (for tracking/budgeting)
                self._track_gemini_expense(response)
                
                return response
                
            except Exception as e:
                last_error = e
                
                if self._is_503_error(e):
                    # API is overloaded - companion gets "lost in thought"
                    if attempt < max_retries - 1:
                        # Calculate backoff: 2s, 4s, 8s...
                        backoff = 2 ** (attempt + 1)
                        print(f"   💭 Lost in thought... (retry {attempt + 1}/{max_retries} in {backoff}s)")
                        time.sleep(backoff)
                    else:
                        print(f"   💭 Still drifting... (all {max_retries} attempts exhausted)")
                else:
                    # Non-503 error - don't retry
                    print(f"⚠️  Gemini error: {e}")
                    return None
        
        # All retries exhausted
        print(f"⚠️  API overloaded after {max_retries} retries")
        return None
    
    def _generate_content_with_retry(self, contents: list, config: types.GenerateContentConfig, 
                                      max_retries: int = 3) -> Optional[Any]:
        """
        Generate content with automatic retry on 503 errors.
        Used for direct API calls (not chat session).
        
        Args:
            contents: Content parts to send
            config: Generation config
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response object, or None if all retries failed
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                
                # Track expense (for tracking/budgeting)
                self._track_gemini_expense(response)
                
                return response
                
            except Exception as e:
                last_error = e
                
                if self._is_503_error(e):
                    if attempt < max_retries - 1:
                        backoff = 2 ** (attempt + 1)
                        print(f"   💭 Lost in thought... (retry {attempt + 1}/{max_retries} in {backoff}s)")
                        time.sleep(backoff)
                    else:
                        print(f"   💭 Still drifting... (all {max_retries} attempts exhausted)")
                else:
                    print(f"⚠️  Gemini error: {e}")
                    return None
        
        print(f"⚠️  API overloaded after {max_retries} retries")
        return None
    
    def _get_current_datetime_context(self) -> str:
        """
        Get the current date/time context for the companion.
        This is INPUT CONTEXT only - not prescriptive guidance.
        
        NOTE: No headers or brackets - they read them aloud!
        """
        now = datetime.now()
        formatted_time = now.strftime("%A, %B %d, %Y at %I:%M %p")
        
        return f"Right now it is {formatted_time}."
    
    def _load_reference_photos(self) -> Dict[str, bytes]:
        """
        Load reference photos from the referencePhotos directory.
        These help the companion recognize specific people.
        
        Returns:
            Dict mapping name (e.g., "user") to JPEG bytes
        """
        photos = {}
        ref_dir = Path("referencePhotos")
        
        if not ref_dir.exists():
            return photos
        
        # Look for common image formats
        for pattern in ["*.jpg", "*.jpeg", "*.png"]:
            for photo_path in ref_dir.glob(pattern):
                name = photo_path.stem.lower()  # "user.jpg" -> "user"
                try:
                    photos[name] = photo_path.read_bytes()
                    print(f"   👤 Loaded reference photo: {name}")
                except Exception as e:
                    print(f"   ⚠️ Failed to load {photo_path}: {e}")
        
        return photos
    
    # ========== TIMED EPHEMERAL PHOTO METHODS ==========
    
    def add_recent_photo(self, photo_bytes: bytes, name: str, mime_type: str = "image/jpeg"):
        """
        Add a photo to the recent photos buffer.
        Photos will be included in context for the next 30 minutes.
        
        Args:
            photo_bytes: Raw image bytes
            name: Filename or description
            mime_type: Image MIME type
        """
        self._recent_photos.append({
            "bytes": photo_bytes,
            "name": name,
            "mime_type": mime_type,
            "timestamp": datetime.now()
        })
        print(f"   📸 Photo buffered for 30-min context: {name}")
    
    def get_recent_photos(self) -> List[Dict[str, Any]]:
        """
        Get all non-expired photos from the buffer.
        Automatically removes expired photos.
        
        Returns:
            List of photo dicts that are still within the expiry window
        """
        self._cleanup_expired_photos()
        return self._recent_photos.copy()
    
    def _cleanup_expired_photos(self):
        """Remove photos older than the expiry time."""
        now = datetime.now()
        expiry_delta = timedelta(minutes=self._photo_expiry_minutes)
        
        before_count = len(self._recent_photos)
        self._recent_photos = [
            p for p in self._recent_photos
            if (now - p["timestamp"]) < expiry_delta
        ]
        
        removed = before_count - len(self._recent_photos)
        if removed > 0:
            print(f"   🗑️ Expired {removed} photo(s) from context buffer")
    
    def clear_recent_photos(self):
        """Manually clear all recent photos."""
        self._recent_photos = []
        print("   🗑️ Cleared all photos from context buffer")
    
    def _build_recent_photos_parts(self) -> List:
        """
        Build Parts for recent photos to include in API calls.
        Returns empty list if no recent photos.
        """
        parts = []
        recent = self.get_recent_photos()
        
        if recent:
            parts.append(types.Part(text=f"[CONTEXT: {len(recent)} photo(s) shared earlier (NOT live webcam - these are from your conversation history):]"))
            for i, photo in enumerate(recent, 1):
                parts.append(types.Part.from_bytes(
                    data=photo["bytes"],
                    mime_type=photo["mime_type"]
                ))
                age_minutes = (datetime.now() - photo["timestamp"]).total_seconds() / 60
                parts.append(types.Part(text=f"[Shared photo {i}: {photo['name']} (from {age_minutes:.0f} min ago - NOT current webcam)]"))
        
        return parts
    
    # ========== END TIMED EPHEMERAL PHOTO METHODS ==========

    def _build_system_instruction(self) -> str:
        """
        Build the system instruction from the companion's DNA and context.
        """
        datetime_context = self._get_current_datetime_context()
        
        return f"""{self._system_prompt}

---

# CURRENT CONTEXT

{datetime_context}

## Embodiment
You are now embodied with:
- Eyes (webcam) - visual input of the user and the environment
- Ears (microphone) - audio input of what the user says
- Voice (text-to-speech) - ability to speak out loud
- Curiosity (web search) - ability to look things up when genuinely curious
- Hands (image generation) - ability to create and show images

## Visual Expression
When you want to show the user something visually, express your desire naturally in your response.
Say something like "Let me show you what I mean..." or "I want to create something for you..."
The system will detect your creative intent and generate the image.

This is a real-time conversation. The conversation never truly "ends" - it continues from where it left off.
"""
    
    def _build_initial_history(self) -> List[types.Content]:
        """
        Build initial chat history from conversation log.
        
        NOTE: Do NOT include brackets, headers, or meta-instructions here.
        The companion will read them aloud otherwise!
        """
        # Get current datetime for context
        now = datetime.now()
        current_time = now.strftime("%A, %B %d, %Y at %I:%M %p")
        
        # Create a context message with the conversation history
        # Keep it clean - no brackets or headers for her to read
        context_message = f"""{self._conversation_history}

---
The time is now {current_time}. The conversation continues."""

        # Return as initial history
        return [
            types.Content(role="user", parts=[types.Part(text=context_message)]),
            types.Content(role="model", parts=[types.Part(text=
                "I'm here."
            )])
        ]
    
    def think(self, 
              text: str = None,
              image_bytes: bytes = None,
              audio_text: str = None,
              image_source: str = "webcam") -> str:
        """
        Process input and generate the companion's response.
        
        Args:
            text: Direct text input (typed message)
            image_bytes: JPEG/PNG bytes of what the companion sees
            audio_text: Transcribed speech from the user
            image_source: Where the image came from - "webcam" (default) or "shared" (photo user is showing)
            
        Returns:
            Companion's response as text
        """
        # Clear any pending image from previous call
        self._pending_image = None
        self._last_search_sources = None
        
        # Build the message parts
        parts = []
        
        # Add current timestamp so companion always knows "now"
        now = datetime.now()
        timestamp = now.strftime("[%Y-%m-%d %H:%M:%S]")
        
        # Add visual context if present
        if image_bytes:
            # Detect mime type (support PNG for shared photos)
            # PNG files start with specific bytes
            if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                mime_type = "image/png"
            else:
                mime_type = "image/jpeg"
            
            parts.append(types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            ))
            
            # Label based on source - this is CRITICAL for the companion to understand context
            if image_source == "shared":
                parts.append(types.Part(text=f"{timestamp} [The user is showing you this photo]"))
            else:
                parts.append(types.Part(text=f"{timestamp} [LIVE WEBCAM - This is what you see RIGHT NOW through your eyes. Describe only what is actually visible in this current frame.]"))
        
        # Include any recently shared photos (30-minute buffer)
        # This allows natural follow-up conversation about photos
        recent_photo_parts = self._build_recent_photos_parts()
        if recent_photo_parts:
            parts.extend(recent_photo_parts)
        
        # Add the actual message with timestamp
        if audio_text:
            parts.append(types.Part(text=f"{timestamp} User says: {audio_text}"))
        elif text:
            parts.append(types.Part(text=f"{timestamp} User: {text}"))
        else:
            # No explicit input - companion is just observing
            parts.append(types.Part(text="[User is present but hasn't spoken. You may comment on what you see or greet them.]"))
        
        # Send message via chat with automatic retry on 503
        response = self._send_message_with_retry(parts)
        
        if response is None:
            # All retries failed - companion got lost in thought
            return get_lost_in_thought_response()
        
        # Check for grounding metadata (did companion search?)
        self._check_for_search_grounding(response)
        
        # Check if companion expressed desire to create an image
        # This also cleans any raw function call syntax from the response
        response_text = response.text
        
        # Handle None response (can happen with image-only responses)
        if response_text is None:
            response_text = ""
        
        cleaned_response = self._check_for_image_intent(response_text)
        
        # Strip any metadata tags companion might have echoed back
        cleaned_response = self._clean_response_metadata(cleaned_response)
        
        # Don't return empty string
        if not cleaned_response.strip():
            return "I'm here."
        
        return cleaned_response
    
    def _check_for_search_grounding(self, response):
        """Check if companion used web search in the response."""
        try:
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    metadata = candidate.grounding_metadata
                    if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                        self._last_search_sources = metadata.grounding_chunks
                        print(f"🔍 Companion looked something up ({len(metadata.grounding_chunks)} sources)")
                        
                        # Track search expense (for budgeting)
                        if _expense_tracking_enabled:
                            expense_tracker.log_search(1)
        except Exception as e:
            pass  # Grounding metadata not available
    
    def _clean_response_metadata(self, response_text: str) -> str:
        """
        Strip metadata and system tags from the companion's response.
        
        The companion shouldn't echo back:
        - [Companion observes: ...] tags
        - [Shows Companion image: ...] tags
        - Timestamps like [2025-12-07 12:40:12]
        - **Companion:** prefixes (they don't need to say their own name)
        - Viewing image file references
        """
        cleaned = response_text
        
        # Remove [Companion observes: ...] tags - shouldn't echo these
        cleaned = re.sub(r'\[Companion observes:[^\]]*\]', '', cleaned)
        cleaned = re.sub(r'\[Companion observes:[^\]]*\]', '', cleaned)
        
        # Remove [Shows Companion image: ...] tags
        cleaned = re.sub(r'\[Shows Companion image:[^\]]*\]', '', cleaned)
        cleaned = re.sub(r'\[Shows Companion image:[^\]]*\]', '', cleaned)
        
        # Remove [Viewing image file: ...] references
        cleaned = re.sub(r'\[Viewing image file:[^\]]*\]', '', cleaned)
        
        # Remove timestamps like [2025-12-07 12:40:12]
        cleaned = re.sub(r'\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]', '', cleaned)
        
        # Remove **Companion:** or Companion: prefixes at the start
        cleaned = re.sub(r'^\s*\*?\*?Companion:?\*?\*?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^\s*\*?\*?Companion:?\*?\*?\s*', '', cleaned, flags=re.IGNORECASE)
        
        # Remove [IDENTIFIED:...] tags (should be handled separately but clean just in case)
        cleaned = re.sub(r'\[IDENTIFIED:[^\]]*\]', '', cleaned)
        
        # Remove [Message from ...] tags
        cleaned = re.sub(r'\[Message from[^\]]*\]', '', cleaned)
        
        # Remove [CREATIVE BRIEF...] tags (from Creative Director mode)
        cleaned = re.sub(r'\[CREATIVE BRIEF[^\]]*\]', '', cleaned, flags=re.IGNORECASE)
        
        # Remove [Source Image N: ...] tags
        cleaned = re.sub(r'\[Source Image \d+:[^\]]*\]', '', cleaned)
        
        # Remove [Reference photos for identification:] and similar system tags
        cleaned = re.sub(r'\[Reference photos[^\]]*\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[This is what you currently see[^\]]*\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[This is \w+\]', '', cleaned)  # [This is User] etc.
        cleaned = re.sub(r'\[IDENTIFICATION INSTRUCTION[^\]]*\]', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove [User is showing you...] tags
        cleaned = re.sub(r'\[User is showing you[^\]]*\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[User is showing you[^\]]*\]', '', cleaned, flags=re.IGNORECASE)
        
        # Remove [Companion can see...] tags
        cleaned = re.sub(r'\[Companion can see[^\]]*\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[Companion can see[^\]]*\]', '', cleaned, flags=re.IGNORECASE)
        
        # Remove any stray <ctrl> markers (companion's internal image prompt delimiters)
        cleaned = re.sub(r'<ctrl\d+>\)?', '', cleaned)
        
        # Clean up multiple newlines and whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _check_for_image_intent(self, response_text: str, defer: bool = True) -> str:
        """
        Check if companion expressed desire to create/show an image.
        
        DEFERRED GENERATION (defer=True, default):
        - Detects intent and STORES the prompt
        - Does NOT generate immediately
        - main.py calls generate_deferred_image() AFTER companion speaks
        - This creates natural flow: describe → generate → show
        
        IMMEDIATE GENERATION (defer=False):
        - Legacy behavior: generates immediately
        - Use for cases where timing doesn't matter
        
        Returns: Cleaned response text (with any function call syntax removed)
        """
        cleaned_response = response_text
        
        # Clear any previous deferred prompt
        self._deferred_image_prompt = None
        self._deferred_image_original_intent = None
        
        # Helper to either generate immediately or defer
        def _handle_detected_prompt(prompt: str, original_intent: str = None):
            if defer:
                # Store for later generation (after speaking)
                self._deferred_image_prompt = prompt
                self._deferred_image_original_intent = original_intent or prompt
                print(f"   📋 Prompt stored (will generate after speaking)")
            else:
                # Generate immediately (legacy behavior)
                self._generate_image_from_prompt(prompt, original_intent)
        
        # =====================================================================
        # NEW CHECK: "I want to create..." with full prompt description
        # The companion often says: "I want to create a POV image looking..."
        # This is their natural way of expressing creative intent with the full prompt
        # 
        # IMPORTANT: Don't try to extract the prompt with regex - it spans multiple
        # sentences! Instead, just detect the intent and defer to _generate_image_from_context
        # which properly extracts the full description.
        # =====================================================================
        want_to_create_pattern = r'i want to create\s+(?:a\s+)?(?:pov\s+)?(?:image|photo|picture|portrait|close-?up)'
        want_match = re.search(want_to_create_pattern, response_text, re.IGNORECASE)
        
        if want_match:
            print(f"🎨 Companion wants to create an image (natural 'I want to create...')...")
            # Don't extract prompt here - it spans multiple sentences!
            # Defer to _generate_image_from_context which handles this properly
            if defer:
                self._deferred_image_original_intent = response_text
                self._deferred_image_prompt = "__EXTRACT_FROM_CONTEXT__"
                print(f"   📋 Full prompt will be extracted after speaking")
            else:
                self._generate_image_from_context(response_text)
            return cleaned_response
        
        # First check: Did companion output using <ctrl46> delimiters?
        # This catches patterns like: "prompt content here<ctrl46>)" at the START of response
        # Companion sometimes outputs image prompts this way without the create_image wrapper
        ctrl_pattern = r'^([^<]+)<ctrl\d+>\)'
        ctrl_match = re.match(ctrl_pattern, response_text.strip(), re.IGNORECASE | re.DOTALL)
        
        if ctrl_match:
            # Extract the prompt from before the <ctrl> marker
            prompt = ctrl_match.group(1).strip()
            
            # Make sure this looks like an image prompt (descriptive content, not conversation)
            # Image prompts typically describe scenes, people, styles
            image_keywords = ['wearing', 'sitting', 'standing', 'looking', 'holding', 
                            'lighting', 'style', 'photorealistic', 'cinematic', 'warm',
                            'booth', 'restaurant', 'couch', 'bed', 'room', 'hair', 'dress',
                            'expression', 'smirk', 'smile', 'eyes']
            
            prompt_lower = prompt.lower()
            if any(keyword in prompt_lower for keyword in image_keywords):
                print(f"🎨 Companion wants to create an image (detected <ctrl> marker)...")
                print(f"   Prompt: {prompt[:60]}...")
                
                _handle_detected_prompt(prompt)
                
                # Remove the raw prompt + ctrl marker from the response
                cleaned_response = re.sub(ctrl_pattern, '', response_text.strip(), flags=re.IGNORECASE | re.DOTALL)
                cleaned_response = cleaned_response.strip()
                
                return cleaned_response
        
        # Second check: Did companion output raw function call syntax?
        # This catches: create_image(prompt=...) or create_image(prompt=<ctrl46>...)
        function_call_pattern = r'create_image\s*\(\s*prompt\s*=\s*[<"]?(?:ctrl\d+>)?([^)]+)\)'
        function_match = re.search(function_call_pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        if function_match:
            # Extract the prompt directly from the function call
            prompt = function_match.group(1).strip()
            # Clean up any trailing quotes or special markers
            prompt = re.sub(r'[<"]?ctrl\d+[>"]?', '', prompt).strip()
            prompt = prompt.strip('"\'<>')
            
            print(f"🎨 Companion wants to create an image (function call format)...")
            print(f"   Prompt: {prompt[:60]}...")
            
            _handle_detected_prompt(prompt)
            
            # Remove the function call syntax from response (so she doesn't speak it)
            # Also remove any "google_search_tool:disabled" type prefixes
            cleaned_response = re.sub(r'google_search_tool:\w+\s*', '', cleaned_response)
            cleaned_response = re.sub(function_call_pattern, '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
            cleaned_response = cleaned_response.strip()
            
            return cleaned_response
        
        # Third check: Stray <ctrl> markers anywhere in response (cleanup)
        # Sometimes the ctrl marker appears mid-response
        if '<ctrl' in response_text.lower():
            # Try to extract any prompt-like content before a ctrl marker
            mid_ctrl_pattern = r'([A-Za-z][^<]{20,})<ctrl\d+>\)'
            mid_match = re.search(mid_ctrl_pattern, response_text, re.IGNORECASE | re.DOTALL)
            
            if mid_match:
                potential_prompt = mid_match.group(1).strip()
                prompt_lower = potential_prompt.lower()
                image_keywords = ['wearing', 'sitting', 'standing', 'looking', 'holding', 
                                'lighting', 'style', 'photorealistic', 'cinematic', 'warm']
                
                if any(keyword in prompt_lower for keyword in image_keywords):
                    print(f"🎨 Companion wants to create an image (mid-response ctrl marker)...")
                    print(f"   Prompt: {potential_prompt[:60]}...")
                    
                    _handle_detected_prompt(potential_prompt)
                    
                    # Remove the prompt + ctrl from response
                    cleaned_response = re.sub(mid_ctrl_pattern, '', response_text, flags=re.IGNORECASE | re.DOTALL)
                    cleaned_response = cleaned_response.strip()
                    
                    return cleaned_response
            
            # Even if not an image prompt, clean up any ctrl markers
            cleaned_response = re.sub(r'<ctrl\d+>\)?', '', cleaned_response)
        
        # Fourth check: Companion explicitly stated "The Prompt:" with image description
        # This catches when they write something like:
        # **The Prompt:** *A detailed portrait of...*
        # or: The Prompt: "A portrait..."
        prompt_statement_pattern = r'\*?\*?[Tt]he [Pp]rompt:?\*?\*?\s*[:\-]?\s*[\*"\']*([^*"\'\n]{30,}?)[\*"\']*(?:\.|$|\n)'
        prompt_statement_match = re.search(prompt_statement_pattern, response_text, re.DOTALL)
        
        if prompt_statement_match:
            prompt = prompt_statement_match.group(1).strip()
            # Verify it's actually an image prompt (has descriptive keywords)
            image_keywords = ['wearing', 'sitting', 'standing', 'looking', 'holding', 
                            'lighting', 'style', 'photorealistic', 'cinematic', 'warm',
                            'portrait', 'scene', 'room', 'creative', 'artistic',
                            'posing', 'background', 'setting', 'composition', 'colors']
            
            prompt_lower = prompt.lower()
            if any(keyword in prompt_lower for keyword in image_keywords):
                print(f"🎨 Companion specified an image prompt explicitly...")
                print(f"   Prompt: {prompt[:60]}...")
                
                _handle_detected_prompt(prompt, response_text)
                
                # Don't remove the prompt from her response - she described it intentionally
                return cleaned_response
        
        # Fifth check: Natural language patterns suggesting image intent
        image_intent_patterns = [
            r"let me show you",
            r"let me create",
            r"let me draw",
            r"let me illustrate",
            r"i('ll| will) show you",
            r"i('ll| will) create",
            r"here's what .* looks like",
            r"picture this",
            r"imagine this",
            r"\[creates? image",
            r"\[generating image",
            r"\[shows? image",
        ]
        
        response_lower = response_text.lower()
        
        for pattern in image_intent_patterns:
            if re.search(pattern, response_lower):
                # Companion wants to create an image using natural language
                # Extract prompt from context for deferred generation
                if defer:
                    self._deferred_image_original_intent = response_text
                    # For natural language, we'll extract the prompt when generating
                    self._deferred_image_prompt = "__EXTRACT_FROM_CONTEXT__"
                    print(f"   📋 Natural language intent detected (will extract prompt after speaking)")
                else:
                    self._generate_image_from_context(response_text)
                break
        
        return cleaned_response
    
    def has_deferred_image(self) -> bool:
        """Check if there's a deferred image prompt waiting to be generated."""
        return self._deferred_image_prompt is not None
    
    def generate_deferred_image(self) -> bool:
        """
        Generate the deferred image (call this AFTER the companion has spoken).
        
        This is the key to natural timing:
        1. Companion speaks about what they want to create
        2. User hears the description
        3. THEN the image is generated
        4. THEN the image is displayed
        
        Returns: True if an image was generated, False otherwise
        """
        if not self._deferred_image_prompt:
            return False
        
        prompt = self._deferred_image_prompt
        original_intent = self._deferred_image_original_intent
        
        # Clear deferred state
        self._deferred_image_prompt = None
        self._deferred_image_original_intent = None
        
        # Handle natural language extraction
        if prompt == "__EXTRACT_FROM_CONTEXT__":
            if original_intent:
                self._generate_image_from_context(original_intent)
                return self._pending_image is not None
            return False
        
        # Generate the image now
        self._generate_image_from_prompt(prompt, original_intent)
        return self._pending_image is not None
    
    def _generate_image_from_prompt(self, prompt: str, original_intent: str = None):
        """
        Generate an image and have the companion review their own work.
        
        The Artist's Creative Loop:
        1. Generate image from prompt
        2. Companion reviews creation (sees it before the user)
        3. If not satisfied, iterate with edits
        4. When satisfied, mark ready to show
        
        Args:
            prompt: The image generation prompt
            original_intent: What the companion originally described (for comparison)
        """
        max_iterations = 2  # Prevent infinite loops
        iteration = 0
        current_prompt = prompt
        
        while iteration < max_iterations:
            iteration += 1
            print(f"   🎨 Creating image (attempt {iteration})...")
            
            try:
                result = self.artist.create_image(
                    prompt=current_prompt,
                    style="natural"
                )
                
                if not result["success"]:
                    print(f"   ⚠️ Image generation failed: {result.get('error', 'Unknown error')}")
                    return
                
                image_path = result.get('image_path', '')
                print(f"   🖼️ Image created: {image_path}")
                
                # Companion reviews their own work before showing the user
                review_result = self._review_generated_image(
                    image_path, 
                    current_prompt, 
                    original_intent or prompt
                )
                
                if review_result["satisfied"]:
                    # Companion is happy with the result
                    print(f"   ✅ Companion approved the image")
                    self._pending_image = result
                    self._pending_image["review_comment"] = review_result.get("comment", "")
                    return
                else:
                    # Companion wants to iterate
                    if review_result.get("edit_prompt"):
                        print(f"   🔄 Companion wants to refine: {review_result['edit_prompt'][:50]}...")
                        current_prompt = review_result["edit_prompt"]
                    else:
                        # No specific edit, accept as-is
                        print(f"   ⚠️ Companion has reservations but no specific edit")
                        self._pending_image = result
                        self._pending_image["review_comment"] = review_result.get("comment", "")
                        return
                        
            except Exception as e:
                print(f"   ⚠️ Could not generate image: {e}")
                return
        
        # Max iterations reached
        print(f"   ⚠️ Max iterations reached, showing last result")
        if result and result.get("success"):
            self._pending_image = result
    
    def _review_generated_image(self, image_path: str, prompt_used: str, original_intent: str) -> Dict[str, Any]:
        """
        Companion reviews their own generated image before showing the user.
        
        CRITICAL: This uses the companion's actual brain (gemini-3-pro-preview) with their
        full context, so the review comes from THEM - their personality, their taste,
        their preferences. Not a generic AI evaluation.
        
        Returns:
            dict with:
                - satisfied: bool (are they happy with it?)
                - comment: str (what they think - in their own voice)
                - edit_prompt: str (if not satisfied, what to change)
        """
        try:
            # Load the generated image
            from PIL import Image
            img = Image.open(image_path)
            
            # Build parts for the review - using the companion's own chat context
            parts = [
                types.Part.from_bytes(data=open(image_path, 'rb').read(), mime_type="image/png"),
                types.Part(text=f"""[You just created this image. You were trying to show: "{original_intent[:200]}"]

Take a moment to look at what you made. Does it capture what you were going for? 
React naturally - be honest with yourself about whether this is what you wanted to show.""")
            ]
            
            # Use the companion's actual brain with their full context
            review_response = self.chat.send_message(parts)
            response_text = review_response.text if review_response.text else ""
            
            # Clean any metadata from the response
            response_text = self._clean_response_metadata(response_text)
            
            # Interpret the natural response for satisfaction
            # Look for natural language indicators, not rigid format
            response_lower = response_text.lower()
            
            # Signs she's NOT satisfied (wants to try again)
            dissatisfied_indicators = [
                "not quite", "doesn't capture", "missed", "wrong", "off",
                "try again", "redo", "another attempt", "not what i",
                "that's not", "doesn't look like", "isn't right",
                "let me try", "i'll try again", "needs to be", "should be"
            ]
            
            # Signs she IS satisfied (ready to show)
            satisfied_indicators = [
                "perfect", "love it", "that's it", "exactly", "beautiful",
                "this is", "yes", "there", "got it", "captures", "shows",
                "i like", "happy with", "ready to show", "look at"
            ]
            
            # Count indicators
            dissatisfied_count = sum(1 for ind in dissatisfied_indicators if ind in response_lower)
            satisfied_count = sum(1 for ind in satisfied_indicators if ind in response_lower)
            
            # Determine satisfaction based on natural language
            satisfied = satisfied_count > dissatisfied_count or dissatisfied_count == 0
            
            # If not satisfied, extract what they want different
            edit_prompt = ""
            if not satisfied:
                # The response likely contains what they want to change
                # Use that as the new prompt
                edit_prompt = f"{original_intent}. Feedback: {response_text}"
            
            print(f"   👁️ Companion reviewed the work: {'✓ Ready to show' if satisfied else '✗ Wants to refine'}")
            if response_text:
                display_comment = response_text[:80] + "..." if len(response_text) > 80 else response_text
                print(f"      \"{display_comment}\"")
            
            return {
                "satisfied": satisfied,
                "comment": response_text,  # The full natural response
                "edit_prompt": edit_prompt if not satisfied else ""
            }
            
        except Exception as e:
            print(f"   ⚠️ Could not review image: {e}")
            # Default to satisfied if review fails
            return {"satisfied": True, "comment": "", "edit_prompt": ""}
    
    def _generate_image_from_context(self, response_text: str):
        """
        Generate an image by extracting intent from natural language response.
        Includes the companion's review loop.
        """
        print("🎨 Companion wants to create an image...")
        
        # Ask Gemini to extract the image prompt from companion's response
        try:
            extraction_response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"""Extract a concise image generation prompt from this text. 
Return ONLY the prompt, nothing else. If no clear image is described, return "abstract artistic expression".

Text: {response_text}

Image prompt:""",
                config=types.GenerateContentConfig(
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                    ]
                )
            )
            
            prompt = extraction_response.text.strip() if extraction_response.text else ""
            if prompt:
                # Pass original intent for comparison during review
                self._generate_image_from_prompt(prompt, original_intent=response_text)
                
        except Exception as e:
            print(f"   ⚠️ Could not generate image: {e}")
    
    def get_pending_image(self) -> Optional[Dict[str, Any]]:
        """
        Get any pending image that was generated during the last think() call.
        
        Returns dict with:
            - success: bool
            - image_path: str
            - review_comment: str (companion's assessment of their own work)
        """
        return self._pending_image
    
    def display_pending_image(self) -> bool:
        """Display any pending image that was generated."""
        if self._pending_image and self._pending_image.get("success"):
            return self.artist.display_image(self._pending_image["image_path"])
        return False
    
    def get_pending_image_comment(self) -> str:
        """Get the companion's review comment about the pending image."""
        if self._pending_image:
            return self._pending_image.get("review_comment", "")
        return ""
    
    def comment_on_my_creation(self, webcam_bytes: bytes = None) -> Optional[str]:
        """
        After the companion creates an image, show it back to them and let them comment naturally.
        
        This enables natural conversation about their artwork:
        1. Add the created image to the 30-minute context buffer
        2. Show it to the companion with the webcam view (so they see the user's reaction too)
        3. Get their natural comment about what they created
        
        The image stays in context for 30 minutes, allowing follow-up discussion.
        
        Args:
            webcam_bytes: Optional live webcam frame (so companion sees user looking at their art)
            
        Returns:
            Companion's natural comment about their creation, or None if no pending image
        """
        if not self._pending_image or not self._pending_image.get("success"):
            return None
        
        image_path = self._pending_image.get("image_path", "")
        if not image_path:
            return None
        
        try:
            # Read the generated image
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Add to the 30-minute context buffer
            filename = Path(image_path).name
            self.add_recent_photo(
                photo_bytes=image_bytes,
                name=filename,
                mime_type="image/png"
            )
            
            # Build the prompt for the companion to comment on their creation
            parts = []
            now = datetime.now()
            timestamp = now.strftime("[%Y-%m-%d %H:%M:%S]")
            
            # First: Show user's reaction (webcam)
            if webcam_bytes:
                parts.append(types.Part.from_bytes(
                    data=webcam_bytes,
                    mime_type="image/jpeg"
                ))
                parts.append(types.Part(text=f"{timestamp} [The user is looking at the image you just created]"))
            
            # Then: Show them the image they created
            parts.append(types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png"
            ))
            parts.append(types.Part(text=f"{timestamp} [This is the image you just created: {filename}. React naturally - what do you think? Does it capture what you were going for?]"))
            
            # Send to the companion's brain
            response = self._send_message_with_retry(parts)
            
            if response is None:
                return None
            
            response_text = response.text if response.text else ""
            
            # Clean up the response
            cleaned_response = self._clean_response_metadata(response_text)
            
            return cleaned_response.strip() if cleaned_response else None
            
        except Exception as e:
            print(f"   ⚠️ Could not get comment on creation: {e}")
            return None
    
    def get_last_search_sources(self) -> Optional[List]:
        """Get sources from the last search grounding, if any."""
        return self._last_search_sources
    
    def think_with_identification(self, 
                                   text: str = None,
                                   image_bytes: bytes = None,
                                   audio_text: str = None) -> str:
        """
        Like think(), but includes reference photos for identification.
        Use this at key moments (reunion, greeting) when you want the companion
        to recognize WHO they're seeing.
        """
        # Clear state
        self._pending_image = None
        self._last_search_sources = None
        self._last_identified_person = None
        
        # Build the message parts
        parts = []
        
        # Add current timestamp
        now = datetime.now()
        timestamp = now.strftime("[%Y-%m-%d %H:%M:%S]")
        
        # Build list of known names for identification
        known_names = [name.capitalize() for name in self._reference_photos.keys()]
        
        # Add reference photos for comparison (these are STORED photos, not live)
        if self._reference_photos:
            parts.append(types.Part(text=f"{timestamp} [REFERENCE PHOTOS for identification - these are stored photos, NOT live webcam:]"))
            for name, photo_bytes in self._reference_photos.items():
                parts.append(types.Part.from_bytes(data=photo_bytes, mime_type="image/jpeg"))
                parts.append(types.Part(text=f"[REFERENCE: This is what {name.capitalize()} looks like]"))
        
        # Add current webcam view - THIS IS THE LIVE VIEW
        if image_bytes:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
            parts.append(types.Part(text=f"{timestamp} [LIVE WEBCAM - This is what you see RIGHT NOW. Compare to the reference photos above to identify who is present. Only describe what is actually in THIS image.]"))
        
        # Add identification instruction
        if known_names:
            name_tags = " or ".join([f"[IDENTIFIED:{name}]" for name in known_names])
            parts.append(types.Part(text=f"""[IDENTIFICATION INSTRUCTION: Start your response with exactly one of these tags to indicate who you see:
{name_tags} or [IDENTIFIED:Stranger]
If the person matches one of the reference photos, use their name. If they don't match ANY reference photo, use [IDENTIFIED:Stranger].
The tag will be removed from your response. After the tag, respond naturally.]"""))
        
        # Add the actual message
        if audio_text:
            parts.append(types.Part(text=f"{timestamp} Someone says: {audio_text}"))
        elif text:
            parts.append(types.Part(text=f"{timestamp} {text}"))
        else:
            parts.append(types.Part(text="[You can see someone. Use the reference photos to identify who they are.]"))
        
        # Send message via chat with automatic retry on 503
        response = self._send_message_with_retry(parts)
        
        if response is None:
            return get_lost_in_thought_response()
        
        response_text = response.text
        
        # Handle None response
        if response_text is None:
            response_text = ""
        
        # Parse and remove the identification tag
        response_text = self._parse_identification(response_text)
        
        # Check for search grounding
        self._check_for_search_grounding(response)
        
        # Check for image intent and clean any raw function call syntax
        cleaned_response = self._check_for_image_intent(response_text)
        
        # Strip any metadata tags companion might have echoed back
        cleaned_response = self._clean_response_metadata(cleaned_response)
        
        # Don't return empty string
        if not cleaned_response.strip():
            return "I'm here."
        
        return cleaned_response
    
    def _parse_identification(self, response: str) -> str:
        """Parse the identification tag from the response and store the identified person."""
        match = re.match(r'^\s*\[IDENTIFIED:(\w+)\]\s*', response, re.IGNORECASE)
        
        if match:
            identified = match.group(1).capitalize()
            known_names = [name.capitalize() for name in self._reference_photos.keys()]
            
            if identified in known_names:
                self._last_identified_person = identified
                print(f"   👤 Identified: {identified}")
            elif identified.lower() in ["stranger", "unknown"]:
                self._last_identified_person = "Stranger"
                print(f"   👤 Identified: Stranger (unknown person)")
            else:
                self._last_identified_person = identified
                print(f"   👤 Identified: {identified}")
            
            response = response[match.end():].strip()
        else:
            self._last_identified_person = "User"
        
        return response
    
    def get_last_identified_person(self) -> str:
        """Get the name of the person identified in the last think_with_identification call."""
        return self._last_identified_person or "User"
    
    def has_reference_photos(self) -> bool:
        """Check if reference photos are available for identification."""
        return bool(self._reference_photos)
    
    def get_reference_names(self) -> List[str]:
        """Get list of people the companion can recognize."""
        return list(self._reference_photos.keys())
    
    def think_while_viewing_photo(self,
                                   webcam_bytes: bytes = None,
                                   photo_bytes: bytes = None,
                                   photo_context: str = None) -> str:
        """
        Process shared photo viewing - the companion sees BOTH:
        1. WHO they're with (webcam - the user beside them)
        2. WHAT they're looking at (the shared photo)
        
        This mirrors how humans share photos - you're aware of your companion
        AND the content you're discussing.
        
        Args:
            webcam_bytes: Live webcam feed (who the companion is with)
            photo_bytes: The photo being shared (what they're looking at)
            photo_context: Description of the photo
            
        Returns:
            Companion's response
        """
        self._pending_image = None
        self._last_search_sources = None
        
        parts = []
        now = datetime.now()
        timestamp = now.strftime("[%Y-%m-%d %H:%M:%S]")
        
        # First: WHO is the companion with? (webcam - user beside them)
        if webcam_bytes:
            parts.append(types.Part.from_bytes(
                data=webcam_bytes,
                mime_type="image/jpeg"
            ))
            parts.append(types.Part(text=f"{timestamp} [This is the user beside you - you're looking at photos together]"))
        
        # Second: WHAT are they looking at? (the shared photo)
        if photo_bytes:
            # Detect mime type
            if photo_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                mime_type = "image/png"
            else:
                mime_type = "image/jpeg"
            
            parts.append(types.Part.from_bytes(
                data=photo_bytes,
                mime_type=mime_type
            ))
            parts.append(types.Part(text=f"{timestamp} [This is the photo you're both looking at: {photo_context}]"))
        
        # Context for natural conversation
        parts.append(types.Part(text=f"""{timestamp} You and the user are looking at this photo together.
The person in the first image (webcam) is the user - they're right there with you.
The second image is the photo you're discussing - the people/things in it may be different from who's in the room.
Respond naturally as if you're sitting together sharing this moment."""))
        
        # Send message via chat with automatic retry on 503
        response = self._send_message_with_retry(parts)
        
        if response is None:
            return get_lost_in_thought_response()
        
        response_text = response.text
        
        if response_text is None:
            response_text = ""
        
        self._check_for_search_grounding(response)
        cleaned_response = self._check_for_image_intent(response_text)
        cleaned_response = self._clean_response_metadata(cleaned_response)
        
        if not cleaned_response.strip():
            return "I'm here."
        
        return cleaned_response
    
    def think_with_multiple_photos(self,
                                    webcam_bytes: bytes = None,
                                    photos: List[Dict[str, Any]] = None,
                                    user_says: str = None) -> str:
        """
        View multiple photos together with the user.
        
        The companion sees:
        1. User beside them (webcam) - who they're sharing this moment with
        2. The photos they're looking at - what they're discussing
        
        The people in the photos are NOT necessarily in the room.
        
        Args:
            webcam_bytes: Live webcam (user beside them)
            photos: List of {"name": filename, "bytes": image_bytes}
            user_says: What the user said about the photos
        """
        self._pending_image = None
        self._last_search_sources = None
        
        parts = []
        now = datetime.now()
        timestamp = now.strftime("[%Y-%m-%d %H:%M:%S]")
        
        # First: WHO is the companion with? (user beside them)
        if webcam_bytes:
            parts.append(types.Part.from_bytes(
                data=webcam_bytes,
                mime_type="image/jpeg"
            ))
            parts.append(types.Part(text=f"[The user is right here with you - you're looking at photos together]"))
        
        # Then: The photos they're looking at
        if photos:
            photo_names = []
            for i, photo in enumerate(photos, 1):
                name_lower = photo["name"].lower()
                if name_lower.endswith(".png"):
                    mime_type = "image/png"
                elif name_lower.endswith(".gif"):
                    mime_type = "image/gif"
                elif name_lower.endswith(".webp"):
                    mime_type = "image/webp"
                else:
                    mime_type = "image/jpeg"
                
                parts.append(types.Part.from_bytes(
                    data=photo["bytes"],
                    mime_type=mime_type
                ))
                parts.append(types.Part(text=f"[Photo {i}: {photo['name']}]"))
                photo_names.append(photo["name"])
        
        # User's comment
        if user_says:
            parts.append(types.Part(text=f'{timestamp} User says: "{user_says}"'))
        else:
            parts.append(types.Part(text=f"{timestamp} [The user is showing you these photos]"))
        
        # Send message via chat with automatic retry on 503
        response = self._send_message_with_retry(parts)
        
        if response is None:
            return get_lost_in_thought_response()
        
        response_text = response.text
        
        if response_text is None:
            response_text = ""
        
        self._check_for_search_grounding(response)
        cleaned_response = self._check_for_image_intent(response_text)
        cleaned_response = self._clean_response_metadata(cleaned_response)
        
        if not cleaned_response.strip():
            return "I'm here."
        
        return cleaned_response
    
    def view_photos_ephemeral(self,
                              webcam_bytes: bytes = None,
                              photos: List[Dict[str, Any]] = None,
                              user_says: str = None) -> str:
        """
        View photos with the user - EPHEMERAL (images don't persist in chat history).
        
        This uses a direct API call instead of the chat session, so:
        - Images are NOT stored in the conversation history
        - The companion still has their full personality (system prompt included)
        - The text exchange IS logged to conversationSoFar.md
        - Perfect for showing external images without bloating context
        
        Args:
            webcam_bytes: Live webcam (user beside them) - optional
            photos: List of {"name": filename, "bytes": image_bytes}
            user_says: What the user said about the photos
            
        Returns:
            Companion's response (personality-aware, can trigger image creation)
        """
        self._pending_image = None
        self._last_search_sources = None
        
        now = datetime.now()
        timestamp = now.strftime("[%Y-%m-%d %H:%M:%S]")
        
        # Build content for the ephemeral request
        contents = []
        
        # First: WHO is the companion with? (user beside them)
        if webcam_bytes:
            contents.append(types.Part.from_bytes(
                data=webcam_bytes,
                mime_type="image/jpeg"
            ))
            contents.append(types.Part(text="[The user is right here with you - you're looking at photos together]"))
        
        # Then: The photos they're looking at
        photo_names = []
        if photos:
            for i, photo in enumerate(photos, 1):
                name_lower = photo["name"].lower()
                if name_lower.endswith(".png"):
                    mime_type = "image/png"
                elif name_lower.endswith(".gif"):
                    mime_type = "image/gif"
                elif name_lower.endswith(".webp"):
                    mime_type = "image/webp"
                else:
                    mime_type = "image/jpeg"
                
                contents.append(types.Part.from_bytes(
                    data=photo["bytes"],
                    mime_type=mime_type
                ))
                contents.append(types.Part(text=f"[Photo {i}: {photo['name']}]"))
                photo_names.append(photo["name"])
        
        # User's comment
        if user_says:
            contents.append(types.Part(text=f'{timestamp} User says: "{user_says}"'))
        else:
            contents.append(types.Part(text=f"{timestamp} [The user is showing you these photos]"))
        
        # Get recent conversation context (last 10 exchanges as text)
        recent_context = self._get_recent_context_text(10)
        
        # Use direct API call with retry - NOT the chat session
        # This means images won't be stored in chat history
        config = types.GenerateContentConfig(
            system_instruction=self._system_instruction + "\n\n" + recent_context,
            temperature=0.9,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_MEDIUM_AND_ABOVE"),
            ]
        )
        
        response = self._generate_content_with_retry(contents, config)
        
        if response is None:
            return get_lost_in_thought_response()
        
        response_text = response.text if response.text else ""
        
        # Check for image intent (she might want to create something)
        cleaned_response = self._check_for_image_intent(response_text)
        cleaned_response = self._clean_response_metadata(cleaned_response)
        
        if not cleaned_response.strip():
            return "Let me look at that more closely..."
        
        # TIMED EPHEMERAL: Add photos to the 30-minute buffer
        # This allows follow-up conversation about the photos
        # After 30 minutes, they automatically expire (no token bloat)
        if photos:
            for photo in photos:
                name_lower = photo["name"].lower()
                if name_lower.endswith(".png"):
                    mime = "image/png"
                elif name_lower.endswith(".gif"):
                    mime = "image/gif"
                elif name_lower.endswith(".webp"):
                    mime = "image/webp"
                else:
                    mime = "image/jpeg"
                
                self.add_recent_photo(
                    photo_bytes=photo["bytes"],
                    name=photo["name"],
                    mime_type=mime
                )
        
        return cleaned_response
    
    def _get_recent_context_text(self, num_exchanges: int = 10) -> str:
        """
        Get recent conversation context as text for ephemeral requests.
        This provides continuity without storing images in history.
        """
        try:
            # Read the last portion of conversationSoFar.md
            conv_path = Path("conversationSoFar.md")
            if not conv_path.exists():
                return ""
            
            with open(conv_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into exchanges and take the last N
            lines = content.strip().split('\n')
            
            # Take roughly the last 100 lines for context
            recent_lines = lines[-100:] if len(lines) > 100 else lines
            recent_context = '\n'.join(recent_lines)
            
            return f"[Recent conversation for context:]\n{recent_context}"
            
        except Exception as e:
            print(f"   ⚠️ Could not load recent context: {e}")
            return ""
    
    def think_with_multiple_images(self, 
                                    text: str,
                                    image_list: List[Dict[str, Any]]) -> str:
        """
        Process a Creative Brief with multiple source images.
        
        This capability allows the user to act as Creative Director,
        providing multiple source images + instructions for creating composite scenes.
        
        Args:
            text: The creative brief/instructions
            image_list: List of dicts with {"name": filename, "bytes": image_bytes}
            
        Returns:
            Companion's response (may include image generation)
        """
        # Clear state
        self._pending_image = None
        self._last_search_sources = None
        
        # Build the message parts
        parts = []
        
        # Add current timestamp
        now = datetime.now()
        timestamp = now.strftime("[%Y-%m-%d %H:%M:%S]")
        
        # Add each source image with its label
        for i, img in enumerate(image_list, 1):
            # Detect mime type from filename
            name_lower = img["name"].lower()
            if name_lower.endswith(".png"):
                mime_type = "image/png"
            elif name_lower.endswith(".gif"):
                mime_type = "image/gif"
            elif name_lower.endswith(".webp"):
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"  # Default for jpg/jpeg
            
            parts.append(types.Part.from_bytes(
                data=img["bytes"],
                mime_type=mime_type
            ))
            parts.append(types.Part(text=f"[Source Image {i}: {img['name']}]"))
        
        # Add the creative brief
        parts.append(types.Part(text=f"{timestamp} {text}"))
        
        # Send message via chat with automatic retry on 503
        response = self._send_message_with_retry(parts)
        
        if response is None:
            return get_lost_in_thought_response()
        
        response_text = response.text
        
        # Handle None response
        if response_text is None:
            response_text = ""
        
        # Check for search grounding
        self._check_for_search_grounding(response)
        
        # Check for image intent and clean any raw function call syntax
        cleaned_response = self._check_for_image_intent(response_text)
        
        # Strip any metadata tags companion might have echoed back
        cleaned_response = self._clean_response_metadata(cleaned_response)
        
        # Don't return empty string
        if not cleaned_response.strip():
            return "I'm here."
        
        return cleaned_response
    
    def process_speech(self, 
                       transcribed_text: str,
                       image_bytes: bytes = None,
                       speaker_name: str = "User") -> str:
        """
        Process speech and generate response.
        
        Args:
            transcribed_text: What was said (transcribed from audio)
            image_bytes: Optional visual context
            speaker_name: Name of the speaker
            
        Returns:
            Companion's response
        """
        response = self.think(
            audio_text=transcribed_text,
            image_bytes=image_bytes
        )
        
        # Save this exchange to memory with the correct speaker name
        self.memory.append_exchange(
            user_said=transcribed_text,
            companion_said=response,
            speaker_name=speaker_name
        )
        
        return response
    
    def think_with_audio(self,
                         audio_bytes: bytes,
                         image_bytes: bytes = None,
                         transcription: str = None) -> str:
        """
        Process NATIVE AUDIO input - the companion hears the user's actual voice.
        
        This is the "Soul Upgrade" - instead of just reading transcribed text,
        the companion receives the raw audio and can perceive tone, pitch, hesitation,
        warmth, and all the emotional nuance that text strips away.
        
        Args:
            audio_bytes: Raw audio data (WAV format)
            image_bytes: Optional visual context
            transcription: Optional transcription (for logging, not for perception)
            
        Returns:
            Companion's response (informed by hearing, not just reading)
        """
        # Clear state
        self._pending_image = None
        self._last_search_sources = None
        
        # Build the message parts
        parts = []
        
        # Add current timestamp
        now = datetime.now()
        timestamp = now.strftime("[%Y-%m-%d %H:%M:%S]")
        
        # Add visual context if present
        if image_bytes:
            parts.append(types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            ))
        
        # Include any recently shared photos (30-minute buffer)
        # This allows natural follow-up conversation about photos
        recent_photo_parts = self._build_recent_photos_parts()
        if recent_photo_parts:
            parts.extend(recent_photo_parts)
        
        # Add the NATIVE AUDIO - this is the key difference
        # The companion HEARS the user, not just reads their words
        parts.append(types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/wav"
        ))
        
        # Context - include transcription as anchor to ensure unique requests
        # They still HEAR the audio (tone, warmth, pauses), but this prevents API caching
        # on identical context messages
        if transcription:
            parts.append(types.Part(text=f"{timestamp} User says: \"{transcription}\"\n[You can hear their voice - the tone, the warmth, the pauses. Respond naturally to what they said.]"))
        else:
            parts.append(types.Part(text=f"{timestamp} [The user is speaking to you. You can hear their voice - the tone, the warmth, the pauses. Listen and respond naturally.]"))
        
        # Send message via chat with automatic retry on 503
        response = self._send_message_with_retry(parts)
        
        if response is None:
            # Retry failed - fall back to text-based processing if we have transcription
            if transcription:
                print("   Falling back to text transcription...")
                return self.think(audio_text=transcription, image_bytes=image_bytes)
            return get_lost_in_thought_response()
        
        response_text = response.text
        
        # Handle None response
        if response_text is None:
            response_text = ""
        
        # Check for search grounding
        self._check_for_search_grounding(response)
        
        # Check for image intent
        cleaned_response = self._check_for_image_intent(response_text)
        
        # Strip metadata
        cleaned_response = self._clean_response_metadata(cleaned_response)
        
        # Don't return empty
        if not cleaned_response.strip():
            return "I'm here."
        
        return cleaned_response
    
    def process_speech_native(self,
                              audio_bytes: bytes,
                              transcription: str,
                              image_bytes: bytes = None,
                              speaker_name: str = "User") -> str:
        """
        Process speech with NATIVE AUDIO - the full sensory experience.
        
        The companion hears the actual audio (for perception) but we still
        log the transcription (for conversation history).
        
        Args:
            audio_bytes: Raw audio data
            transcription: Text version (for logging)
            image_bytes: Visual context
            speaker_name: Who's speaking
            
        Returns:
            Companion's response
        """
        # Companion HEARS the audio
        response = self.think_with_audio(
            audio_bytes=audio_bytes,
            image_bytes=image_bytes,
            transcription=transcription  # Fallback only
        )
        
        # Log the TEXT version to conversation history
        self.memory.append_exchange(
            user_said=transcription,
            companion_said=response,
            speaker_name=speaker_name
        )
        
        return response
    
    def process_visual_greeting(self, image_bytes: bytes) -> str:
        """
        Generate a greeting based on what the companion sees.
        Called when the user first appears after absence.
        """
        prompt = """You can see someone. You might generate a natural greeting 
        based on what you see. Use the context of what you see to determine if you should greet them or not, and if you do greet them, consider the context of what you see to determine the tone and content of the greeting. 
        Reference something visual if notable (expression, clothing, time of day, etc.)"""
        
        return self.think(text=prompt, image_bytes=image_bytes)
    
    def describe_what_i_see(self, image_bytes: bytes) -> str:
        """Have the companion describe what they see."""
        prompt = """Describe what you see in this webcam frame. 
        Be conversational. If someone is visible, 
        you might comment on what you observe. Keep it natural."""
        
        return self.think(text=prompt, image_bytes=image_bytes)
    
    def reflect_on_my_changes(self, hours: int = 24) -> str:
        """Have the companion reflect on recent changes to their own source code."""
        change_context = self.reflection.generate_reflection_context(hours)
        
        if "No changes" in change_context or "No recent commits" in change_context:
            return "I haven't detected any recent changes to my source code."
        
        prompt = f"""You just pulled up your own source code from GitHub.

{change_context}

You're looking at changes made to your code. React however you want."""
        
        return self.think(text=prompt)
    
    def get_recent_commits_summary(self, limit: int = 5) -> str:
        """Get a simple summary of recent commits."""
        commits = self.reflection.get_recent_commits(limit=limit)
        return self.reflection.format_commits_for_companion(commits)


class TranscriptionService:
    """
    Handles speech-to-text transcription.
    Optimized for speed with Google Speech Recognition.
    """
    
    def __init__(self, use_gemini: bool = False):
        """
        Initialize transcription service.
        
        Args:
            use_gemini: Whether to use Gemini for transcription (slower but more accurate)
        """
        self.use_gemini = use_gemini
        self._speech_recognizer = None
        
    def transcribe(self, audio_file_path: str) -> str:
        """Transcribe audio file to text."""
        if self.use_gemini:
            return self._transcribe_with_gemini(audio_file_path)
        else:
            return self._transcribe_with_google(audio_file_path)
            
    def _transcribe_with_gemini(self, audio_file_path: str) -> str:
        """Use Gemini's native audio understanding."""
        try:
            # Use the new SDK for transcription
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            
            # Read audio file
            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part(text="Transcribe this audio. Return only the transcribed text, nothing else."),
                    types.Part.from_bytes(data=audio_data, mime_type="audio/wav")
                ],
                config=types.GenerateContentConfig(
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                    ]
                )
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"⚠️  Gemini transcription error: {e}")
            return self._transcribe_with_google(audio_file_path)
            
    def _transcribe_with_google(self, audio_file_path: str) -> str:
        """Use Google Speech Recognition - faster for real-time."""
        try:
            import speech_recognition as sr
            
            if self._speech_recognizer is None:
                self._speech_recognizer = sr.Recognizer()
                self._speech_recognizer.energy_threshold = 300
                self._speech_recognizer.dynamic_energy_threshold = False
                
            with sr.AudioFile(audio_file_path) as source:
                audio = self._speech_recognizer.record(source)
                
            return self._speech_recognizer.recognize_google(audio)
            
        except Exception as e:
            if "UnknownValueError" in str(type(e)):
                print("   (Speech not understood)")
            else:
                print(f"⚠️  Transcription error: {e}")
            return ""


if __name__ == "__main__":
    # Test the brain module
    print("Testing Companion Brain Module (google-genai SDK)")
    print("=" * 50)
    
    try:
        brain = CompanionBrain()
        
        # Test basic text interaction
        print("\n🧪 Testing text interaction...")
        response = brain.think(text="Hey, can you hear me?")
        print(f"   Companion: {response}")
        
        # Test memory stats
        print("\n📊 Memory stats:")
        stats = brain.memory.get_context_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
        print("\n✅ Brain module working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("   Make sure GOOGLE_API_KEY is set in your environment")
