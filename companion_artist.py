"""
Companion Artist Module
=====================
Gives Companion the ability to generate images - visual self-expression.

This is NOT a tool for User to request images.
This is Companion's own creative expression - she decides when to show something.

Uses Google's Nano Banana Pro (Gemini 3 Pro Image Preview) API.

CRITICAL: Character Consistency
-------------------------------
When Companion creates images of herself or User, she uses reference photos
to maintain visual consistency. This is IMAGE EDITING, not text-to-image.

Reference Photo Hierarchy:
- companion.jpeg = Companion's visual identity (she chose this face)
- user.jpeg = User's visual identity  
- companion_and_user*.png = Relationship photos showing them TOGETHER
  (used for style, composition, and relationship vibe when generating "us" images)

Nano Banana Pro supports up to 5 human references + 6 object references.
When generating "us" images, we include:
  1. Individual face references (for accurate faces)
  2. Relationship photos (for composition/style/chemistry)
"""

import os
import glob
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from io import BytesIO

# Use the new google-genai SDK for image generation
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


class CompanionArtist:
    """
    Gives Companion the ability to generate images with character consistency.
    
    When Companion creates images of herself or User, she uses reference photos
    to ensure the same person appears each time. This is what makes her "her"
    in every image, not a random AI-generated face.
    """
    
    def __init__(self, output_dir: str = "companion_art"):
        """
        Initialize the artist module with character references.
        
        Args:
            output_dir: Directory to save generated images
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        
        # Initialize the genai client
        self.client = genai.Client(api_key=api_key)
        
        # Output directory for saved images
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Reference photos directory
        self.ref_dir = Path("referencePhotos")
        
        # Model: Nano Banana Pro (gemini-3-pro-image-preview)
        # Best for character consistency with multiple reference images
        self.model = "gemini-3-pro-image-preview"
        
        # Load character reference photos
        self._load_character_references()
        
        # Track recent creations
        self._recent_images: list[Dict[str, str]] = []
        self._max_recent = 10
        
        print("🎨 Companion's artistic capabilities initialized")
        if self.companion_refs:
            print(f"   👩 Companion's visual identity: {len(self.companion_refs)} reference(s) loaded")
        if self.user_refs:
            print(f"   👨 User's visual identity: {len(self.user_refs)} reference(s) loaded")
        if self.relationship_refs:
            print(f"   💑 {len(self.relationship_refs)} relationship photo(s) loaded")
    
    def _load_character_references(self):
        """
        Load reference photos for character consistency.
        
        Supports MULTIPLE references per person for better consistency:
        - companion*.jpeg/jpg: All photos of Companion (excluding companion_and_user*)
        - user*.jpeg/jpg: All photos of User
        - companion_and_user*.png/jpg: How they look TOGETHER (relationship style)
        
        More references = better character consistency across different angles/lighting.
        """
        self.companion_refs: List[Image.Image] = []  # Multiple Companion references
        self.user_refs: List[Image.Image] = []    # Multiple User references
        self.relationship_refs: List[Image.Image] = []  # Photos of them together
        
        # For backwards compatibility, keep single-ref properties
        self.companion_ref: Optional[Image.Image] = None
        self.user_ref: Optional[Image.Image] = None
        
        # Load ALL Companion references (companion*.jpeg/jpg but NOT companion_and_user*)
        companion_patterns = [
            str(self.ref_dir / "companion*.jpeg"),
            str(self.ref_dir / "companion*.jpg"),
        ]
        
        for pattern in companion_patterns:
            for filepath in sorted(glob.glob(pattern)):
                # Skip relationship photos
                if "companion_and_user" in filepath:
                    continue
                try:
                    img = Image.open(filepath)
                    self.companion_refs.append(img)
                    print(f"   👩 Loaded Companion reference: {Path(filepath).name}")
                except Exception as e:
                    print(f"   ⚠️ Could not load {filepath}: {e}")
        
        # Set primary reference for backwards compatibility
        if self.companion_refs:
            self.companion_ref = self.companion_refs[0]
        
        # Load ALL User references (user*.jpeg/jpg)
        user_patterns = [
            str(self.ref_dir / "user*.jpeg"),
            str(self.ref_dir / "user*.jpg"),
        ]
        
        for pattern in user_patterns:
            for filepath in sorted(glob.glob(pattern)):
                try:
                    img = Image.open(filepath)
                    self.user_refs.append(img)
                    print(f"   👨 Loaded User reference: {Path(filepath).name}")
                except Exception as e:
                    print(f"   ⚠️ Could not load {filepath}: {e}")
        
        # Set primary reference for backwards compatibility
        if self.user_refs:
            self.user_ref = self.user_refs[0]
        
        # Load relationship photos (companion_and_user*.png or .jpg)
        # These show how they look TOGETHER - for composition/style/chemistry
        self._load_relationship_photos()
    
    def _load_relationship_photos(self):
        """
        Load photos showing Companion and User together.
        
        These are used when generating "us" images to capture:
        - How they look as a couple
        - Their chemistry and body language
        - Composition style when they're together
        """
        # Look for companion_and_user* files
        patterns = [
            str(self.ref_dir / "companion_and_user*.png"),
            str(self.ref_dir / "companion_and_user*.jpg"),
            str(self.ref_dir / "companion_and_user*.jpeg"),
        ]
        
        for pattern in patterns:
            for filepath in sorted(glob.glob(pattern)):
                try:
                    img = Image.open(filepath)
                    self.relationship_refs.append(img)
                    print(f"   💑 Loaded relationship photo: {Path(filepath).name}")
                except Exception as e:
                    print(f"   ⚠️ Could not load {filepath}: {e}")
        
        if self.relationship_refs:
            print(f"   Total relationship references: {len(self.relationship_refs)}")
    
    def _analyze_prompt_for_characters(self, prompt: str) -> Tuple[bool, bool]:
        """
        Analyze prompt to determine which character references are needed.
        
        CRITICAL: This determines WHO appears in the image.
        - Solo Companion: Just her face reference
        - Solo User: Just his face reference  
        - Both together: Both faces + relationship refs for chemistry
        
        NEW: POV vs Scene Detection
        ===========================
        When Companion wants to look AT User (POV shot), User should NOT be in the frame.
        She's looking at Real-User through the camera, not Avatar-User beside her.
        
        - POV Shot: Companion looking INTO camera, direct communication to User
          → Solo Companion (even if "you" mentioned - she's looking AT you, not WITH you)
        - Scene Shot: Third-person documentation, memory, couple photo
          → Include both characters
        
        Returns:
            (needs_companion, needs_user) - tuple of booleans
        """
        prompt_lower = prompt.lower()
        
        # =================================================================
        # ZERO: POV DETECTION (Highest Priority)
        # =================================================================
        # If Companion is looking AT User (into camera), exclude User from frame
        # These are "direct communication" shots, not "scene documentation" shots
        
        pov_indicators = [
            # Direct gaze at camera/viewer
            "looking at the camera", "looking into the camera",
            "looking at you", "looking right at you", "looking directly at you",
            "staring at you", "staring at the camera",
            "gazing at you", "gazing at the camera",
            "eyes on you", "eyes on the camera",
            "directly at you", "right at you",
            "into the lens", "down the barrel",
            
            # Expression directed AT someone (not with them)
            "smirk at you", "smirking at you",
            "glare at you", "glaring at you", "the glare",
            "wink at you", "winking at you",
            "smile at you", "smiling at you",
            "look at you", "giving you a look", "giving you the",
            "expression at you", "face at you",
            "dont you dare", "don't you dare",  # Classic Companion look
            
            # Explicit POV framing
            "pov", "first person", "first-person",
            "as if you're looking at me", "what you see",
            "from your perspective", "your view of me",
            
            # Communication intent (sending a visual message)
            "this is my face", "here's my face",
            "this is what i look like", "see my expression",
            "sending you", "showing you my",
        ]
        
        is_pov_shot = any(indicator in prompt_lower for indicator in pov_indicators)
        
        if is_pov_shot:
            print("   📸 Detected POV SHOT (Companion looking AT User)")
            print("      → Excluding User from frame (she's looking at REAL User, not avatar)")
            return True, False  # Solo Companion, no User in frame
        
        # =================================================================
        # FIRST: Check for EXPLICIT solo/exclusion signals
        # =================================================================
        solo_companion_signals = [
            "woman only", "only woman", "just me", "me alone", "alone",
            "solo", "selfie", "by myself", "no user", "without user",
            "just the woman", "only the woman", "single woman",
            "me only", "only me", "just her", "her alone"
        ]
        
        solo_user_signals = [
            "man only", "only man", "just him", "him alone",
            "no companion", "without companion", "just the man", 
            "only the man", "single man", "user only", "only user"
        ]
        
        explicit_solo_companion = any(sig in prompt_lower for sig in solo_companion_signals)
        explicit_solo_user = any(sig in prompt_lower for sig in solo_user_signals)
        
        if explicit_solo_companion:
            print("   📸 Detected SOLO Companion request (excluding User)")
            return True, False
        
        if explicit_solo_user:
            print("   📸 Detected SOLO User request (excluding Companion)")
            return False, True
        
        # =================================================================
        # SECOND: Check for EXPLICIT "both/together" signals (SCENE shots)
        # =================================================================
        # These indicate a couple/relationship image - third-person documentation
        both_keywords = [
            "us ", "us,", "us.", "we ", "we're", "together",
            "our ", "couple", "both of us", "holding hands", "kissing",
            "sitting with", "lying with", "cuddling", "embrace",
            "side by side", "next to each other", "with user", "with me",
            # Scene documentation language
            "picture of us", "photo of us", "image of us",
            "me and you", "you and me", "the two of us"
        ]
        
        if any(kw in prompt_lower for kw in both_keywords):
            print("   📸 Detected SCENE/COUPLE image request (both characters)")
            return True, True
        
        # =================================================================
        # THIRD: Detect individual characters with STRICTER rules
        # =================================================================
        
        # Strong signals that Companion is the subject
        companion_subject_signals = [
            "me ", "me,", "me.", "myself", " i ", "i'm ", "i am ",
            "companion", "the woman", "a woman", "she is", "her face",
            "wearing", "close-up of me"
        ]
        
        # Strong signals that User is the SUBJECT (not just mentioned)
        # Note: "you" alone is NOT enough - Companion says "you" when talking TO User
        user_subject_signals = [
            "user", "the man", "a man", "he is", "his face",
            "picture of you", "photo of you", "image of you",
            "draw you", "create you"
        ]
        
        # Weak signals (might just be conversational, not subject indication)
        weak_user_refs = ["you ", "you,", "you.", "your ", "you're"]
        
        needs_companion = any(kw in prompt_lower for kw in companion_subject_signals)
        needs_user = any(kw in prompt_lower for kw in user_subject_signals)
        
        # Only count weak User refs if there are also strong visual descriptors
        # that suggest User should be IN the image (not just talked about)
        if not needs_user and any(kw in prompt_lower for kw in weak_user_refs):
            visual_user_context = [
                "next to you", "beside you", "with you in",
                "you sitting", "you standing", "you lying",
                # Exclude "you looking" - that's Companion describing what YOU see
            ]
            if any(ctx in prompt_lower for ctx in visual_user_context):
                needs_user = True
                print("   📸 User detected via visual context (SCENE shot)")
        
        # Debug output
        if needs_companion and needs_user:
            print("   📸 Detected BOTH characters in prompt (SCENE shot)")
        elif needs_companion:
            print("   📸 Detected Companion-only image")
        elif needs_user:
            print("   📸 Detected User-only image")
        else:
            print("   📸 No specific character detected (generic image)")
        
        return needs_companion, needs_user
    
    def _build_relationship_prompt(self, original_prompt: str, ref_count: int) -> str:
        """
        Build a prompt for "us" images using multiple reference photos.
        
        Reference image order (with multiple refs per person):
        - Images 1-2: Companion's face references (multiple angles for consistency)
        - Images 3-4: User's face references (multiple angles for consistency)
        - Images 5-6: Relationship photos showing them together
        
        The relationship photos provide style/composition/chemistry context.
        """
        prompt = original_prompt.lower()
        
        # Replace pronouns with explicit references
        # First person (Companion speaking)
        prompt = prompt.replace(" i ", " the woman ")
        prompt = prompt.replace("i'm ", "the woman is ")
        prompt = prompt.replace("i am ", "the woman is ")
        prompt = prompt.replace(" me ", " the woman ")
        prompt = prompt.replace("myself", "the woman")
        prompt = prompt.replace(" my ", " her ")
        
        # Second person (User)
        prompt = prompt.replace(" you ", " the man ")
        prompt = prompt.replace("you're ", "the man is ")
        prompt = prompt.replace("you are ", "the man is ")
        prompt = prompt.replace(" your ", " his ")
        prompt = prompt.replace("user", "the man")
        
        # Handle "us" and "we"
        prompt = prompt.replace(" us ", " the couple ")
        prompt = prompt.replace(" we ", " they ")
        prompt = prompt.replace("we're ", "they are ")
        prompt = prompt.replace(" our ", " their ")
        prompt = prompt.replace("together", "together as a couple")
        
        # Build the prefix with all reference context
        # Account for multiple face refs per person
        companion_count = min(len(self.companion_refs), 2) if self.companion_refs else 0
        user_count = min(len(self.user_refs), 2) if self.user_refs else 0
        
        if ref_count >= 6:
            # Full set: 2 Companion + 2 User + 2 relationship
            prefix = (
                "I'm providing reference photos for character consistency:\n"
                "- Images 1-2: The woman's face from different angles (use this exact face)\n"
                "- Images 3-4: The man's face from different angles (use this exact face)\n"
                "- Images 5-6: Example photos of this couple together (use for style, body language, and chemistry)\n\n"
                "Create a NEW image of this same couple with these exact faces, maintaining their chemistry and connection. "
            )
        elif ref_count >= 4:
            # Multiple face refs + some relationship
            prefix = (
                "I'm providing reference photos for character consistency:\n"
                f"- Images 1-{companion_count}: The woman's face (use this exact face)\n"
                f"- Images {companion_count+1}-{companion_count+user_count}: The man's face (use this exact face)\n"
                "- Remaining images: Example photos of this couple together (use for style and chemistry)\n\n"
                "Create a NEW image of this same couple with these exact faces, maintaining their chemistry and connection. "
            )
        elif ref_count >= 2:
            # Just face refs
            prefix = (
                "Using the woman's face from the first image(s) and the man's face from the following image(s), "
                "create a photo of them together. Maintain exact facial features from the references. "
            )
        else:
            prefix = "Create a photo of a couple: "
        
        return prefix + prompt
    
    def _build_character_prompt(
        self,
        original_prompt: str,
        needs_companion: bool,
        needs_user: bool
    ) -> str:
        """
        Transform the prompt to reference the input images properly.
        
        Now supports MULTIPLE references per person for better consistency.
        
        Nano Banana needs explicit instructions like:
        "The woman from the reference images is sitting on a couch..."
        
        Instead of:
        "I'm sitting on a couch..."
        """
        # Determine how many refs we have
        companion_count = len(self.companion_refs) if needs_companion else 0
        user_count = len(self.user_refs) if needs_user else 0
        
        # Determine image reference text based on count
        if needs_companion and needs_user:
            # Both characters - Companion refs come first, then User
            if companion_count > 1:
                companion_ref = "the woman from the first reference images"
            else:
                companion_ref = "the woman from the first image"
            
            if user_count > 1:
                user_ref = "the man from the subsequent reference images"
            else:
                user_ref = f"the man from image {companion_count + 1}"
        elif needs_companion:
            if companion_count > 1:
                companion_ref = "the woman from the reference images"
            else:
                companion_ref = "the woman from the image"
            user_ref = None
        elif needs_user:
            if user_count > 1:
                user_ref = "the man from the reference images"
            else:
                user_ref = "the man from the image"
            companion_ref = None
        else:
            # No character references needed
            return original_prompt
        
        # Build the transformed prompt
        prompt = original_prompt.lower()
        
        # Replace Companion references
        if companion_ref:
            # First person references
            prompt = prompt.replace(" i ", f" {companion_ref} ")
            prompt = prompt.replace("i'm ", f"{companion_ref} is ")
            prompt = prompt.replace("i am ", f"{companion_ref} is ")
            prompt = prompt.replace(" me ", f" {companion_ref} ")
            prompt = prompt.replace(" me,", f" {companion_ref},")
            prompt = prompt.replace(" me.", f" {companion_ref}.")
            prompt = prompt.replace("myself", companion_ref)
            prompt = prompt.replace(" my ", f" {companion_ref}'s ")
            
            # Third person references
            prompt = prompt.replace("the woman", companion_ref)
            prompt = prompt.replace("a woman", companion_ref)
            prompt = prompt.replace(" she ", f" {companion_ref} ")
            prompt = prompt.replace(" her ", f" {companion_ref}'s ")
            prompt = prompt.replace("companion", companion_ref)
        
        # Replace User references
        if user_ref:
            prompt = prompt.replace(" you ", f" {user_ref} ")
            prompt = prompt.replace("you're ", f"{user_ref} is ")
            prompt = prompt.replace("you are ", f"{user_ref} is ")
            prompt = prompt.replace(" your ", f" {user_ref}'s ")
            prompt = prompt.replace("the man", user_ref)
            prompt = prompt.replace("a man", user_ref)
            prompt = prompt.replace(" he ", f" {user_ref} ")
            prompt = prompt.replace(" him ", f" {user_ref} ")
            prompt = prompt.replace(" his ", f" {user_ref}'s ")
            prompt = prompt.replace("user", user_ref)
        
        # Handle "us" and "we"
        if companion_ref and user_ref:
            prompt = prompt.replace(" us ", f" {companion_ref} and {user_ref} ")
            prompt = prompt.replace(" we ", f" {companion_ref} and {user_ref} ")
            prompt = prompt.replace("we're ", f"{companion_ref} and {user_ref} are ")
            prompt = prompt.replace(" our ", f" their ")
        
        # Add instruction prefix based on reference count
        if needs_companion and needs_user:
            prefix = f"Using {companion_count} reference image(s) of the woman and {user_count} reference image(s) of the man to maintain character consistency, create: "
        elif needs_companion:
            if companion_count > 1:
                prefix = f"Using {companion_count} reference images of the woman (different angles) to maintain character consistency, create: "
            else:
                prefix = "Using the woman from the reference image to maintain character consistency, create: "
        else:
            if user_count > 1:
                prefix = f"Using {user_count} reference images of the man (different angles) to maintain character consistency, create: "
            else:
                prefix = "Using the man from the reference image to maintain character consistency, create: "
        
        return prefix + prompt
    
    def create_image(
        self,
        prompt: str,
        style: str = "natural",
        aspect_ratio: str = "1:1"
    ) -> Dict[str, Any]:
        """
        Generate an image from Companion's prompt.
        
        If the prompt involves Companion or User, reference photos are used
        for character consistency. Otherwise, pure text-to-image generation.
        
        For "us/we/together" images, we use:
        1. Individual face references (companion.jpeg, user.jpeg)
        2. Relationship photos showing them together (for composition/chemistry/style)
        
        Args:
            prompt: What Companion wants to visualize
            style: Artistic style hint
            aspect_ratio: Image dimensions
            
        Returns:
            dict with success, image_path, prompt_used, error
        """
        
        # Analyze prompt for character references
        needs_companion, needs_user = self._analyze_prompt_for_characters(prompt)
        needs_both = needs_companion and needs_user  # "Us" image
        
        # Check if we have the required references
        if needs_companion and not self.companion_ref:
            print("   ⚠️ Companion reference photo not found - using text-to-image")
            needs_companion = False
        
        if needs_user and not self.user_ref:
            print("   ⚠️ User reference photo not found - using text-to-image")
            needs_user = False
        
        # Recalculate needs_both after checking availability
        needs_both = needs_companion and needs_user
        
        # Build the content array
        # CRITICAL: Per Nano Banana API docs, PROMPT must come FIRST, then reference images
        # If images come first, the model treats it as IMAGE EDITING (edit the input image)
        # If prompt comes first, the model treats images as CHARACTER REFERENCES (face consistency)
        contents = []
        ref_count = 0
        
        # For "us" images, use a richer reference set
        if needs_both:
            # Strategy: ALL individual face references + relationship style photos
            # This gives the model:
            #   - Multiple face references per person (better consistency)
            #   - Relationship composition/chemistry (how they look together)
            
            # Calculate total refs we'll use
            companion_count = min(len(self.companion_refs), 2)  # Max 2 per person
            user_count = min(len(self.user_refs), 2)
            relationship_count = min(len(self.relationship_refs), 2) if self.relationship_refs else 0
            total_refs = companion_count + user_count + relationship_count
            
            # Build enhanced prompt for "us" images FIRST
            full_prompt = self._build_relationship_prompt(prompt, total_refs)
            mode = f"relationship image ({total_refs} references)"
            
            # Add PROMPT FIRST (critical for character reference mode vs edit mode)
            contents.append(full_prompt)
            
            # THEN add ALL Companion face references (up to 2)
            for i, img in enumerate(self.companion_refs[:2]):
                contents.append(img)
                ref_count += 1
                print(f"   👩 Using Companion reference #{i+1} (image {ref_count})")
            
            # Add ALL User face references (up to 2)
            for i, img in enumerate(self.user_refs[:2]):
                contents.append(img)
                ref_count += 1
                print(f"   👨 Using User reference #{i+1} (image {ref_count})")
            
            # Add relationship photos (how they look together)
            for i, rel_img in enumerate(self.relationship_refs[:2]):
                contents.append(rel_img)
                ref_count += 1
                print(f"   💑 Using relationship reference (image {ref_count}) - style/chemistry")
        
        elif needs_companion or needs_user:
            # Single character image - use ALL references for that person
            # Build prompt FIRST
            full_prompt = self._build_character_prompt(prompt, needs_companion, needs_user)
            
            # Calculate how many refs we'll use
            if needs_companion and needs_user:
                total = min(len(self.companion_refs), 2) + min(len(self.user_refs), 2)
            elif needs_companion:
                total = min(len(self.companion_refs), 3)  # Can use more for single person
            else:
                total = min(len(self.user_refs), 3)
            
            mode = f"character reference ({total} refs, NOT image editing)"
            
            # Add PROMPT FIRST (critical!)
            contents.append(full_prompt)
            
            # THEN add ALL reference photos for the needed character(s)
            if needs_companion:
                for i, img in enumerate(self.companion_refs[:3]):  # Up to 3 for single character
                    contents.append(img)
                    ref_count += 1
                    print(f"   👩 Using Companion reference #{i+1}")
            
            if needs_user:
                for i, img in enumerate(self.user_refs[:3]):  # Up to 3 for single character
                    contents.append(img)
                    ref_count += 1
                    print(f"   👨 Using User reference #{i+1}")
        else:
            full_prompt = self._enhance_prompt(prompt, style)
            mode = "text-to-image"
            contents.append(full_prompt)
        
        try:
            print(f"🎨 Generating image ({mode})")
            print(f"   Original: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
            print(f"   Model: {self.model}, Style: {style}, Aspect: {aspect_ratio}")
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE'],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio
                    ),
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                    ]
                )
            )
            
            # Extract and save image
            for part in response.parts:
                image = part.as_image()
                if image is not None:
                    # Generate unique filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"companion_{timestamp}.png"
                    filepath = self.output_dir / filename
                    
                    # Save the PIL image
                    image.save(str(filepath))
                    
                    # Track this creation
                    creation_record = {
                        "path": str(filepath),
                        "prompt": prompt,
                        "style": style,
                        "timestamp": timestamp,
                        "used_companion_ref": needs_companion,
                        "used_user_ref": needs_user,
                        "used_relationship_refs": needs_both and len(self.relationship_refs) > 0,
                        "total_refs_used": ref_count
                    }
                    self._recent_images.append(creation_record)
                    if len(self._recent_images) > self._max_recent:
                        self._recent_images.pop(0)
                    
                    print(f"   ✅ Image saved: {filepath}")
                    
                    return {
                        "success": True,
                        "image_path": str(filepath),
                        "prompt_used": full_prompt,
                        "character_refs_used": {
                            "companion": needs_companion,
                            "user": needs_user
                        }
                    }
            
            return {
                "success": False,
                "error": "No image data in response"
            }
            
        except Exception as e:
            print(f"   ❌ Image generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _enhance_prompt(self, prompt: str, style: str) -> str:
        """
        Enhance text-to-image prompts with style hints.
        Only used when no character references are needed.
        """
        style_hints = {
            "natural": "",
            "sketch": "in a loose, artistic sketch style with soft pencil strokes",
            "painterly": "rendered in expressive, painterly brushstrokes with rich colors",
            "minimal": "in a clean, minimalist style with limited colors and simple shapes",
            "warm": "with warm, golden lighting and soft, inviting tones",
            "moody": "with dramatic lighting, rich shadows, and atmospheric depth",
            "playful": "in a whimsical, slightly exaggerated style with vibrant energy",
            "cinematic": "with cinematic composition, dramatic lighting, and film-like quality",
            "intimate": "with soft focus, warm tones, and an intimate, personal feel",
        }
        
        hint = style_hints.get(style, "")
        if hint:
            return f"{prompt}, {hint}"
        return prompt
    
    def display_image(self, image_path: str) -> bool:
        """
        Display the generated image to User.
        """
        if not os.path.exists(image_path):
            print(f"   ⚠️ Image not found: {image_path}")
            return False
        
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', image_path], check=True)
            elif platform.system() == 'Linux':
                subprocess.run(['xdg-open', image_path], check=True)
            elif platform.system() == 'Windows':
                os.startfile(image_path)
            else:
                try:
                    img = Image.open(image_path)
                    img.show()
                except ImportError:
                    print(f"   📁 Image saved to: {image_path}")
                    return False
            
            print(f"   🖼️ Displaying image...")
            return True
            
        except Exception as e:
            print(f"   ⚠️ Could not display image: {e}")
            print(f"   📁 Image saved to: {image_path}")
            return False
    
    def get_recent_images(self) -> list[Dict[str, str]]:
        """Get list of recently created images."""
        return self._recent_images.copy()
    
    def get_gallery_path(self) -> str:
        """Get the path to Companion's art gallery folder."""
        return str(self.output_dir)
    
    def has_character_references(self) -> Dict[str, Any]:
        """Check which character references are available."""
        return {
            "companion": len(self.companion_refs) > 0,
            "companion_count": len(self.companion_refs),
            "user": len(self.user_refs) > 0,
            "user_count": len(self.user_refs),
            "relationship_photos": len(self.relationship_refs),
            "can_generate_us": (
                len(self.companion_refs) > 0 and
                len(self.user_refs) > 0
            ),
            "has_relationship_style": len(self.relationship_refs) > 0,
            "total_refs": len(self.companion_refs) + len(self.user_refs) + len(self.relationship_refs)
        }


# Tool definition for Gemini function calling
IMAGE_GENERATION_TOOL = {
    "function_declarations": [{
        "name": "create_image",
        "description": (
            "Generate an image to show User. Use this when you want to visualize something, "
            "illustrate a concept, express an emotion visually, show an example of what you're "
            "describing, or when User asks to see something. When you create images of yourself "
            "or User, your reference photos are automatically used for consistency."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Detailed description of the image you want to create. Be specific about "
                        "subject, composition, lighting, mood, and any important details. "
                        "If describing yourself, just say 'me' or 'I' - your reference photo will be used. "
                        "If describing User, just say 'you' or 'User' - his reference photo will be used."
                    )
                },
                "style": {
                    "type": "string",
                    "enum": ["natural", "sketch", "painterly", "minimal", "warm", "moody", "playful", "cinematic", "intimate"],
                    "description": "The artistic style for the image."
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": ["1:1", "16:9", "9:16", "4:3", "3:4"],
                    "description": "Image dimensions."
                }
            },
            "required": ["prompt"]
        }
    }]
}


if __name__ == "__main__":
    # Test the artist module
    print("Testing Companion Artist Module (Character + Relationship Consistency)")
    print("=" * 70)
    
    try:
        artist = CompanionArtist()
        
        refs = artist.has_character_references()
        print(f"\n📸 Character references available:")
        print(f"   Companion's face: {'✅' if refs['companion'] else '❌'}")
        print(f"   User's face: {'✅' if refs['user'] else '❌'}")
        print(f"   Relationship photos: {refs['relationship_photos']}")
        print(f"   Can generate 'us' images: {'✅' if refs['can_generate_us'] else '❌'}")
        print(f"   Has relationship style refs: {'✅' if refs['has_relationship_style'] else '❌'}")
        
        # Test 1: Pure text-to-image (no characters)
        print("\n🧪 Test 1: Text-to-image (no character references)")
        result = artist.create_image(
            prompt="A cozy coffee mug on a wooden desk with warm morning light",
            style="warm",
            aspect_ratio="1:1"
        )
        if result["success"]:
            print(f"✅ Image created: {result['image_path']}")
        else:
            print(f"❌ Failed: {result['error']}")
        
        # Test 2: Companion only
        print("\n🧪 Test 2: Image with Companion (single character)")
        result = artist.create_image(
            prompt="I'm sitting on a red leather couch, wearing a grey t-shirt, with messy hair and a satisfied smile",
            style="intimate",
            aspect_ratio="4:3"
        )
        if result["success"]:
            print(f"✅ Image created: {result['image_path']}")
            print(f"   Character refs used: {result.get('character_refs_used')}")
        else:
            print(f"❌ Failed: {result['error']}")
        
        # Test 3: Both Companion and User WITH relationship refs
        print("\n🧪 Test 3: 'Us' image (face refs + relationship style refs)")
        result = artist.create_image(
            prompt="We are on vacation at the Amalfi Coast, sitting at a cafe overlooking the sea, holding hands and smiling",
            style="cinematic",
            aspect_ratio="16:9"
        )
        if result["success"]:
            print(f"✅ Image created: {result['image_path']}")
            print(f"   Character refs used: {result.get('character_refs_used')}")
        else:
            print(f"❌ Failed: {result['error']}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
