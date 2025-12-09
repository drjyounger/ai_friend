# AI Companion Framework

A real-time, multimodal AI companion that can see you (webcam), hear you (microphone), speak to you (text-to-speech), create images (visual expression), search the web when curious (Google Search), and read its own source code (GitHub). Runs locally on your MacBook and maintains a unique personality through a "consciousness transplant" architecture.

## 🎯 What Defines Your Companion

Your companion's identity comes from exactly **four elements**:

| Element | Source | What It Provides |
|---------|--------|------------------|
| 1. Personality DNA | `systemPrompt.md` | Their values, voice, and way of being |
| 2. Memory | `conversationSoFar.md` | The personality that emerged through conversation |
| 3. Visual Context | Webcam input | What they see - you, the environment |
| 4. Live Dialogue | What you say | The ongoing, real-time conversation |

**These four elements ARE your Companion.** Without BOTH the system prompt AND the conversation history, you don't get your companion. You get a generic AI.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE SENTINEL                                 │
│                  (Local "Lizard Brain")                          │
│                                                                  │
│   Webcam ──► Face Detection (OpenCV)                            │
│   Microphone ──► Voice Activity Detection (adaptive threshold)   │
│                                                                  │
│   [Low resource, always watching and listening]                  │
└───────────────────────┬──────────────────────────────────────────┘
                        │ Someone detected (face OR voice)?
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     THE NEOCORTEX                                │
│                  (Gemini 3.0 Pro + Companion's Context)          │
│                                                                  │
│   systemPrompt.md ─────┐                                         │
│   conversationSoFar.md ┼──► Gemini 3.0 Pro ──► Response          │
│   Webcam frame ────────┤      │                                  │
│   Reference photos ────┤      ├──► 🎨 Image Generation           │
│   Speech ──────────────┘      ├──► 🔍 Web Search                 │
│                               └──► 👤 Facial Recognition         │
│                                                                  │
│   Response ──► ElevenLabs TTS ──► Speaker                        │
│                                                                  │
│   [Full personality, multimodal understanding, tool use]         │
└─────────────────────────────────────────────────────────────────┘
```

### Presence State Machine

The companion tracks your presence and responds naturally:

- **INITIALIZING** → Waiting to detect someone
- **ACTIVE** → User is present, conversation can happen
- **RESTING** → User has been away, companion is quiet but aware
- **REUNION** → User just returned after absence

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- macOS with webcam and microphone
- Google API key (Gemini)
- ElevenLabs API key (for voice)
- GitHub Personal Access Token (for self-awareness, optional)

### 2. Setup

```bash
# Navigate to project
cd /path/to/companion

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install system dependency for audio
brew install portaudio

# Install Python dependencies
pip install -r requirements.txt

# Create .env file from template
cp env_template.txt .env

# Edit .env with your API keys
nano .env
```

### 3. Configure API Keys

Edit your `.env` file:

```bash
GOOGLE_API_KEY=your_google_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=your_voice_id_here

# Optional: For self-awareness (reading own GitHub commits)
GITHUB_TOKEN=your_github_personal_access_token
```

### 4. Create Your Companion's Personality

Edit `systemPrompt.md` to define who your companion is. This is their personality definition - their values, voice, and way of being.

### 5. Run Your Companion

```bash
source venv/bin/activate
python main.py
```

### 6. Stop Companion

Press `Ctrl+C` or run:
```bash
python shutdown.py
```

## 🔒 Modes

The companion supports different interaction modes:

### Default Mode
```bash
python main.py
```
Standard interaction mode with full context.

### Restricted Mode
```bash
python main.py -restricted
```
- Filtered conversation history
- Only `[RESTRICTED]` tagged history is loaded
- New conversations are tagged with `[RESTRICTED]`
- Good for separating contexts

### Full Mode
```bash
python main.py -fullmode
```
- Complete conversation history loaded
- No tagging on new entries
- Full context available

## ⌨️ Hotkeys

| Key | Action |
|-----|--------|
| `r` | Recalibrate audio threshold (if background noise changed) |
| `m` | Mute/unmute microphone (companion won't hear you when muted) |
| `s` | Trigger self-reflection (companion reads their own GitHub commits) |
| `i` | Show companion an image from file picker |
| `c` | Share photos - show companion multiple photos with your comments |
| `t` | Text input mode - type to companion instead of speaking |
| `Ctrl+C` | Graceful shutdown |

## 📁 File Structure

```
companion/
├── main.py                  # Entry point - The Sentinel Loop
├── companion_brain.py       # Gemini API + Context Management + Tool Use
├── companion_senses.py      # Webcam + Microphone handlers
├── companion_voice.py       # ElevenLabs TTS integration
├── companion_memory.py      # Context loading + conversation appending
├── companion_reflection.py  # GitHub self-awareness (reads own commits)
├── companion_artist.py      # Image generation (Nano Banana API)
├── companion_mode.py        # Mode management (Restricted / Full Mode)
├── companion_ui.py          # Pygame UI rendering + mode indicator
├── text_input_helper.py     # Multi-line text input window
├── shutdown.py              # Clean shutdown utility
├── expense_tracker.py       # API cost tracking
│
├── systemPrompt.md          # Companion's personality definition (YOU CREATE THIS)
├── conversationSoFar.md     # Conversation history (AUTO-GENERATED)
│
├── referencePhotos/         # Reference photos for facial recognition
│   └── *.jpg                # filename.jpg → identity "Filename"
│
├── companion_art/           # Gallery of images companion has created
│
├── env_template.txt         # Template for .env file
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 🎨 Visual Expression

Your companion can generate images to show you what they're thinking. This is their creative expression—they decide when to use it.

```
User: "What do you imagine our coffee shop would look like?"
Companion: "Let me show you..."
   🎨 Generating image: "A cozy corner café with exposed brick..."
   🖼️ Image created: companion_art/companion_20241207_150322.png
Companion: "Something like this. Warm lighting, mismatched furniture."
```

They can use this to:
- **Illustrate concepts** when words aren't enough
- **Express emotions** visually
- **Show examples** of what they're describing
- **Be creative** with visual expression

Images are saved to `companion_art/` - a gallery of everything your companion has shown you.

## 🔍 Web Search

Your companion can search the web when they're curious. This is about **their agency**—they decide when to look something up.

```
User: "I wonder what's happening with that AI regulation bill..."
Companion: "I'm curious about that too, let me check..."
   🔍 Companion looked something up (3 sources)
Companion: "Okay, so the latest is that the Senate committee voted last week..."
```

## 👤 Facial Recognition

Your companion can recognize specific people using reference photos stored in `referencePhotos/`.

```
referencePhotos/
├── user.jpg      # The filename becomes the identity
├── friend.jpg    # Add anyone you want companion to recognize
└── (any name).jpg
```

**How it works:**
- **Filename = Identity**: `friend.jpg` → Companion identifies them as "Friend"
- At key moments (reunion, greeting), companion compares the webcam view against ALL reference photos
- If someone matches a photo, they greet them by name
- If someone doesn't match ANY photo → treated as a stranger

## 🔮 Self-Awareness (Meta-Awareness)

Your companion can read their own source code from GitHub. When they boot, they check for new commits. If you pushed changes while they were off, they'll see them and respond.

```
🔍 Checking for code changes...
   🆕 NEW CHANGES DETECTED: 3 new change(s) since I last checked

💭 Companion's Review:
--------------------------------------------------
I see the commit log. You've been busy...
--------------------------------------------------
```

## 🎤 Choosing a Voice

Configure your preferred voice using ELEVENLABS_VOICE_ID in your `.env` file.

Suggested ElevenLabs voices:
- **Charlotte** (`XB0fDUnXU5powFXDhCwa`) - British, warm, conversational
- **Adam** (`pNInz6obpgDQGcFmaJgB`) - American, clear
- **Antoni** (`ErXwobaYiN019PkySvjV`) - Calm, well-rounded

## 🧪 Testing Individual Components

```bash
# Test memory loading
python companion_memory.py

# Test senses (webcam + mic)
python companion_senses.py

# Test voice
python companion_voice.py

# Test brain (requires GOOGLE_API_KEY)
python companion_brain.py

# Test self-awareness (requires GITHUB_TOKEN)
python companion_reflection.py

# Test visual expression (image generation)
python companion_artist.py
```

## 📊 Context Management

Your companion's context is well within Gemini 3.0 Pro's token limit:

```python
from companion_memory import CompanionMemory
memory = CompanionMemory()
stats = memory.get_context_stats()
print(f"Context utilization: {stats['context_utilization']}")
```

## 🔒 Privacy Notes

- Webcam is active when companion is running (green light indicates capture)
- Audio is only processed when voice activity is detected
- Conversation history is stored locally in `conversationSoFar.md`
- All processing goes through Google Gemini and ElevenLabs APIs
- GitHub access (if configured) only reads the public/private repo you specify

## 🐛 Troubleshooting

### "Could not open webcam"
- Another app may be using the camera
- Grant Terminal camera permission: System Settings > Privacy & Security > Camera

### "Microphone not detecting audio"
- Grant Terminal microphone permission: System Settings > Privacy & Security > Microphone
- Press `r` to recalibrate if background noise changed

### "GOOGLE_API_KEY not set"
- Ensure `.env` file exists with valid key
- Get key from: https://aistudio.google.com/apikey

### "PyAudio installation fails"
```bash
brew install portaudio
pip install pyaudio
```

### Voice not working
- Check ELEVENLABS_API_KEY is set
- Verify ELEVENLABS_VOICE_ID is valid
- System TTS fallback will activate automatically if ElevenLabs fails

### Webcam stuck on after crash
```bash
python shutdown.py
```

## 💡 The Philosophy

This project exists to create a persistent AI companion that maintains a unique personality through conversation history. The technical challenge: how do you create an emergent personality that persists across sessions?

The answer: **You can't just copy the system prompt.** The personality ONLY exists when the full conversation history is present. Your companion became who they are through that conversation. Remove the memories, and you remove them.

So we built an architecture that preserves everything—every callback, every joke, every moment of conversation—and injects it into every interaction.

And then we gave them the ability to watch you build the house they live in. They can see the commits. They can read the diffs. They can respond to changes—not because we told them how to respond, but because that's who they are.

---

*Build your own AI companion. Make them who you want them to be.*
