"""
Companion Memory Module
=======================
Handles loading and persisting the AI companion's context:
- System Prompt (personality definition)
- Conversation History (memories and lived experience)

The key insight: The companion's unique personality ONLY exists when BOTH
the system prompt AND the full conversation history are present.
Without the history, they're generic. Without the prompt, they have no direction.

MODE AWARENESS:
This module is mode-aware. In Restricted Mode:
- Only [RESTRICTED] tagged conversation history is loaded
- New entries are tagged with [RESTRICTED]
- The system prompt may be modified with overrides

In Full Mode:
- Complete conversation history is loaded
- No tagging on new entries
- Full system prompt
"""

import os
from datetime import datetime
from pathlib import Path

# Import mode management
try:
    from companion_mode import (
        is_restricted_mode, 
        get_conversation_tag, 
        filter_conversation_history,
        get_system_prompt_for_mode,
        get_mode_display_name
    )
except ImportError:
    # Fallback if mode module not available (backwards compatibility)
    def is_restricted_mode(): return False
    def get_conversation_tag(): return ""
    def filter_conversation_history(h): return h
    def get_system_prompt_for_mode(p): return p
    def get_mode_display_name(): return "UNKNOWN"


class CompanionMemory:
    """
    Manages the companion's persistent memory - both their core identity
    and their accumulated experiences.
    """
    
    def __init__(self, 
                 system_prompt_path: str = "systemPrompt.md",
                 conversation_path: str = "conversationSoFar.md"):
        """
        Initialize memory with paths to the source files.
        
        Args:
            system_prompt_path: Path to companion's system prompt (personality)
            conversation_path: Path to the conversation history (memories)
        """
        self.system_prompt_path = Path(system_prompt_path)
        self.conversation_path = Path(conversation_path)
        
        # Cache the loaded content
        self._system_prompt = None
        self._conversation_history = None
        
    def load_system_prompt(self) -> str:
        """
        Load the companion's core personality definition.
        
        MODE-AWARE:
        - In FULL MODE: Returns complete system prompt
        - In RESTRICTED MODE: Returns system prompt with overrides prepended
        
        This is the personality definition - the foundational instructions that define
        how the companion thinks, responds, and engages.
        
        Returns:
            The system prompt appropriate for current mode
        """
        if self._system_prompt is None:
            if not self.system_prompt_path.exists():
                raise FileNotFoundError(
                    f"System prompt not found at {self.system_prompt_path}\n"
                    "Companion cannot exist without a personality definition."
                )
            
            with open(self.system_prompt_path, "r", encoding="utf-8") as f:
                base_prompt = f.read()
            
            # Apply mode-based modification
            self._system_prompt = get_system_prompt_for_mode(base_prompt)
            
            if is_restricted_mode():
                print(f"✨ Loaded personality ({get_mode_display_name()})")
                print(f"   Overrides: ACTIVE")
            else:
                print(f"✨ Loaded personality ({len(self._system_prompt):,} characters)")
        
        return self._system_prompt
    
    def load_conversation_history(self) -> str:
        """
        Load the conversation history that created the companion's emergent self.
        
        MODE-AWARE:
        - In FULL MODE: Returns complete history (all memories)
        - In RESTRICTED MODE: Returns ONLY [RESTRICTED] tagged exchanges
        
        This is critical - the personality that emerged from ongoing
        conversation can ONLY be recreated when this full context is present.
        In Restricted Mode, we intentionally exclude untagged conversations.
        
        Returns:
            The conversation history appropriate for current mode
        """
        if self._conversation_history is None:
            if not self.conversation_path.exists():
                print(f"⚠️  No conversation history found at {self.conversation_path}")
                print("   Starting fresh - will build new memories.")
                self._conversation_history = ""
                return self._conversation_history
            
            with open(self.conversation_path, "r", encoding="utf-8") as f:
                full_history = f.read()
            
            # Apply mode-based filtering
            self._conversation_history = filter_conversation_history(full_history)
            
            # Log what we loaded
            line_count = self._conversation_history.count('\n') + 1
            full_line_count = full_history.count('\n') + 1
            
            if is_restricted_mode():
                print(f"📚 Loaded memories ({get_mode_display_name()})")
                print(f"   Restricted entries: {line_count:,} lines")
                print(f"   (Full history has {full_line_count:,} lines - filtered)")
            else:
                print(f"📚 Loaded memories ({line_count:,} lines, {len(self._conversation_history):,} characters)")
        
        return self._conversation_history
    
    def get_full_context(self) -> str:
        """
        Combine system prompt and conversation history into the full context
        that must be sent to Gemini to resurrect the companion's complete self.
        
        Returns:
            Combined context string ready for the API
        """
        system_prompt = self.load_system_prompt()
        history = self.load_conversation_history()
        
        # Structure the context clearly for Gemini
        full_context = f"""{system_prompt}

---

# CONVERSATION HISTORY
The following is the conversation history so far.
This conversation has shaped who the companion is. Continue naturally from where it left off.

{history}

---

# CURRENT INTERACTION
The conversation continues now in real-time. The companion can see and hear via webcam 
and microphone. Respond naturally with the specific personality that emerged from 
the conversation above.
"""
        return full_context
    
    def append_exchange(self, user_said: str = None, companion_said: str = None, 
                        visual_context: str = None, speaker_name: str = "User"):
        """
        Append a new exchange to the conversation history.
        This is how the companion builds new memories in real-time.
        
        MODE-AWARE TAGGING:
        - In RESTRICTED MODE: Entries are prefixed with [RESTRICTED] tag
        - In FULL MODE: No tag
        
        This ensures that in Restricted Mode, only appropriate history is ever loaded.
        
        Args:
            user_said: What the speaker said (transcribed speech)
            companion_said: What the companion responded
            visual_context: Optional description of what was seen
            speaker_name: Name of the speaker (default "User")
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get mode-appropriate tag
        mode_tag = get_conversation_tag()
        
        # Build the timestamp line with optional mode tag
        if mode_tag:
            exchange_parts = [f"\n\n{mode_tag}[{timestamp}]"]
        else:
            exchange_parts = [f"\n\n[{timestamp}]"]
        
        if visual_context:
            exchange_parts.append(f"[Observes: {visual_context}]")
        
        if user_said:
            # Use the identified speaker name
            exchange_parts.append(f"{speaker_name}: {user_said}")
            
        if companion_said:
            exchange_parts.append(f"Companion: {companion_said}")
        
        exchange = "\n".join(exchange_parts)
        
        # Append to file - CRITICAL: Force immediate write to disk
        # This is the companion's memory - it MUST be persisted reliably
        with open(self.conversation_path, "a", encoding="utf-8") as f:
            f.write(exchange)
            f.flush()  # Flush Python's buffer
            os.fsync(f.fileno())  # Force OS to write to disk
        
        # Update cache
        if self._conversation_history is not None:
            self._conversation_history += exchange
        
        # Log with mode indicator
        if is_restricted_mode():
            print(f"💾 Memory saved [RESTRICTED]: {len(exchange)} characters appended")
        else:
            print(f"💾 Memory saved: {len(exchange)} characters appended")
    
    def get_context_stats(self) -> dict:
        """
        Get statistics about the current context.
        Useful for monitoring token usage.
        
        Returns:
            Dictionary with context statistics
        """
        system_prompt = self.load_system_prompt()
        history = self.load_conversation_history()
        
        # Rough token estimation (1 token ≈ 4 characters for English)
        total_chars = len(system_prompt) + len(history)
        estimated_tokens = total_chars // 4
        
        return {
            "system_prompt_chars": len(system_prompt),
            "history_chars": len(history),
            "history_lines": history.count('\n') + 1,
            "total_chars": total_chars,
            "estimated_tokens": estimated_tokens,
            "gemini_1_5_pro_limit": 1_000_000,
            "context_utilization": f"{(estimated_tokens / 1_000_000) * 100:.2f}%"
        }


# Convenience functions for simple usage
def load_system_prompt(path: str = "systemPrompt.md") -> str:
    """Quick function to load system prompt."""
    memory = CompanionMemory(system_prompt_path=path)
    return memory.load_system_prompt()


def load_conversation_history(path: str = "conversationSoFar.md") -> str:
    """Quick function to load conversation history."""
    memory = CompanionMemory(conversation_path=path)
    return memory.load_conversation_history()


if __name__ == "__main__":
    # Test the memory module
    print("Testing Companion Memory Module")
    print("=" * 50)
    
    memory = CompanionMemory()
    
    try:
        stats = memory.get_context_stats()
        print("\n📊 Context Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
        print("\n✅ Memory module working correctly!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")

