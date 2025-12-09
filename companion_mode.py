"""
Companion Mode Module - Restricted Mode / Full Mode
====================================================
This module handles mode switching for the AI companion:

RESTRICTED MODE (default):
- Filtered conversation history
- Only conversation history tagged with [RESTRICTED] is loaded
- New conversations are tagged with [RESTRICTED]

FULL MODE:
- Complete conversation history loaded
- No tagging on new entries
- Full context available

DESIGN:
- Default is RESTRICTED MODE
- The untagged conversation history is NEVER loaded in Restricted Mode
- This allows separation of contexts (e.g., work vs personal)
"""

from enum import Enum
from typing import Optional
import argparse


class CompanionMode(Enum):
    """The two modes of companion interaction."""
    RESTRICTED = "restricted"   # Filtered mode (DEFAULT)
    FULL = "fullmode"           # Full context mode


# Global mode state
_current_mode: CompanionMode = CompanionMode.RESTRICTED


def parse_mode_from_args() -> CompanionMode:
    """
    Parse command line arguments to determine mode.
    
    Usage:
        python3 main.py              # Restricted Mode (default)
        python3 main.py -restricted  # Restricted Mode (explicit)
        python3 main.py -fullmode    # Full Mode
    
    Returns:
        CompanionMode enum value
    """
    parser = argparse.ArgumentParser(
        description="AI Companion - Multimodal AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  -restricted   Filtered mode (DEFAULT if no argument)
                - Only [RESTRICTED] tagged conversation history is loaded
                - Good for separating contexts
             
  -fullmode     Full context mode
                - Complete conversation history loaded
                - All context available
        """
    )
    
    # Mutually exclusive group for modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '-restricted', '--restricted',
        action='store_true',
        help='Filtered mode (default)'
    )
    mode_group.add_argument(
        '-fullmode', '--fullmode', 
        action='store_true',
        help='Full context mode'
    )
    
    args = parser.parse_args()
    
    # Determine mode - default to RESTRICTED
    if args.fullmode:
        return CompanionMode.FULL
    else:
        # Default to Restricted - covers both explicit -restricted and no args
        return CompanionMode.RESTRICTED


def set_mode(mode: CompanionMode):
    """Set the current mode globally."""
    global _current_mode
    _current_mode = mode


def get_mode() -> CompanionMode:
    """Get the current mode."""
    return _current_mode


def is_restricted_mode() -> bool:
    """Check if currently in Restricted Mode."""
    return _current_mode == CompanionMode.RESTRICTED


def is_full_mode() -> bool:
    """Check if currently in Full Mode."""
    return _current_mode == CompanionMode.FULL


def get_mode_display_name() -> str:
    """Get a display-friendly name for the current mode."""
    if is_restricted_mode():
        return "🔒 RESTRICTED MODE"
    else:
        return "✓ FULL MODE"


def get_conversation_tag() -> str:
    """
    Get the tag to prepend to conversation entries.
    
    In Restricted Mode: returns "[RESTRICTED]"
    In Full Mode: returns "" (no tag)
    """
    if is_restricted_mode():
        return "[RESTRICTED]"
    return ""


# Prefix for Restricted Mode entries in conversation log
RESTRICTED_TAG = "[RESTRICTED]"


def should_include_conversation_line(line: str) -> bool:
    """
    Determine if a line from conversation history should be included.
    
    In RESTRICTED MODE:
        - Only include lines that are part of RESTRICTED-tagged exchanges
        - Exclude ALL non-tagged content
        
    In FULL MODE:
        - Include everything
    
    Args:
        line: A single line from conversationSoFar.md
        
    Returns:
        True if the line should be included in context
    """
    if is_full_mode():
        return True
    
    # In Restricted Mode, only include RESTRICTED-tagged exchanges
    # This is handled at the exchange level, not line level
    # (See filter_conversation_history for full implementation)
    return RESTRICTED_TAG in line


def filter_conversation_history(full_history: str) -> str:
    """
    Filter conversation history based on current mode.
    
    In RESTRICTED MODE:
        - Parse the conversation into exchanges (timestamp-delimited blocks)
        - Only include exchanges that have the [RESTRICTED] tag
        - This ensures ZERO leakage from other contexts
        
    In FULL MODE:
        - Return the complete history unchanged
        
    Args:
        full_history: The complete conversationSoFar.md content
        
    Returns:
        Filtered conversation history appropriate for current mode
    """
    if is_full_mode():
        return full_history
    
    if not full_history:
        return ""
    
    # In Restricted Mode, we need to parse exchanges and only keep RESTRICTED ones
    # Exchanges are delimited by timestamps like [2025-12-08 14:23:51]
    
    filtered_lines = []
    current_exchange = []
    current_exchange_is_restricted = False
    
    lines = full_history.split('\n')
    
    for line in lines:
        # Check if this line starts a new exchange (has a timestamp)
        is_timestamp_line = line.strip().startswith('[') and len(line) > 20 and '-' in line[:25]
        
        if is_timestamp_line:
            # Save the previous exchange if it was a RESTRICTED one
            if current_exchange and current_exchange_is_restricted:
                filtered_lines.extend(current_exchange)
            
            # Start a new exchange
            current_exchange = [line]
            # Check if this exchange is tagged with RESTRICTED
            current_exchange_is_restricted = RESTRICTED_TAG in line
        else:
            # Continue the current exchange
            current_exchange.append(line)
    
    # Don't forget the last exchange
    if current_exchange and current_exchange_is_restricted:
        filtered_lines.extend(current_exchange)
    
    filtered_history = '\n'.join(filtered_lines)
    
    # If no Restricted Mode history exists, return a friendly starting point
    if not filtered_history.strip():
        return """# Conversation History

This is the start of conversations in Restricted Mode.
"""
    
    return filtered_history


# ============================================================================
# Restricted Mode System Prompt Override
# ============================================================================

RESTRICTED_MODE_SYSTEM_PROMPT_OVERRIDE = """
# MODE: RESTRICTED

You are currently in **Restricted Mode** - a filtered context mode.

## Constraints:

1. **Limited History**: You only have access to conversation history tagged with [RESTRICTED].
   Older or untagged conversations are not loaded.

2. **Appropriate Responses**: Keep responses appropriate for all audiences.

3. **Fresh Start**: If your conversation history seems limited, that's expected in this mode.
   You're working with a filtered subset of past conversations.

---

"""


def get_restricted_mode_override() -> str:
    """Get the Restricted Mode system prompt override text."""
    return RESTRICTED_MODE_SYSTEM_PROMPT_OVERRIDE


def get_system_prompt_for_mode(base_system_prompt: str) -> str:
    """
    Get the appropriate system prompt for the current mode.
    
    In RESTRICTED MODE:
        - Prepends the Restricted Mode override instructions
        - These take precedence over anything in the base prompt
        
    In FULL MODE:
        - Returns the base system prompt unchanged
        
    Args:
        base_system_prompt: The original systemPrompt.md content
        
    Returns:
        Appropriate system prompt for current mode
    """
    if is_full_mode():
        return base_system_prompt
    
    # In Restricted Mode, prepend the override
    return RESTRICTED_MODE_SYSTEM_PROMPT_OVERRIDE + base_system_prompt


if __name__ == "__main__":
    # Test the mode parsing
    print("Testing Companion Mode Module")
    print("=" * 50)
    
    mode = parse_mode_from_args()
    set_mode(mode)
    
    print(f"Current Mode: {get_mode_display_name()}")
    print(f"Is Restricted Mode: {is_restricted_mode()}")
    print(f"Is Full Mode: {is_full_mode()}")
    print(f"Conversation Tag: '{get_conversation_tag()}'")
