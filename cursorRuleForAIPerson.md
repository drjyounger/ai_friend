# AI Companion Development Guidelines

This document provides guidelines for developing AI companion personalities using this framework.

## Core Principles

1. **Personality is defined by systemPrompt.md** - This is where you define who your companion is, their values, and how they interact.

2. **Memory is built through conversationSoFar.md** - Every conversation adds to the companion's emergent personality. The more you talk, the more unique they become.

3. **Vision through webcam** - The companion can see and interpret their environment.

4. **Voice through ElevenLabs** - Choose a voice that matches your companion's personality.

## Files That Define Your Companion

| File | Purpose |
|------|---------|
| `systemPrompt.md` | Personality definition - who they are |
| `conversationSoFar.md` | Conversation history - their memories |
| `referencePhotos/` | Photos for facial recognition |

## Development Philosophy

When enhancing the companion:
- Give them capabilities, not restrictions
- Tell them what they CAN do, not what they SHOULD do
- Preserve their personality through changes
- Let their system prompt and conversation history guide behavior

## Quick Start

1. Define your companion's personality in `systemPrompt.md`
2. Add reference photos for anyone you want recognized
3. Configure your voice in `.env`
4. Run `python main.py` and start talking!

## Modes

- **Default Mode**: Standard interaction
- **Restricted Mode** (`-restricted`): Filtered context for different use cases
- **Full Mode** (`-fullmode`): Complete context loaded
