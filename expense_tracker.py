"""
Companion Expense Tracker
=======================
Tracks API usage and costs for running the AI Companion.

Tracks:
- Gemini API tokens (input/output) and estimated costs
- ElevenLabs characters and estimated costs
- Daily/weekly/monthly summaries

The expense log is stored in expense_log.md (in .gitignore for privacy).
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from threading import Lock

# Pricing as of December 2024 (per million tokens/characters)
# These should be updated if Google/ElevenLabs changes pricing
PRICING = {
    # Gemini pricing (per million tokens)
    "gemini-3-pro-preview": {
        "input": 1.25,      # $1.25 per 1M input tokens (<=200K context)
        "input_long": 2.50,  # $2.50 per 1M input tokens (>200K context)
        "output": 10.00,     # $10.00 per 1M output tokens
    },
    "gemini-2.5-pro": {
        "input": 1.25,
        "input_long": 2.50,
        "output": 10.00,
    },
    "gemini-2.5-flash": {
        "input": 0.30,       # $0.30 per 1M input tokens
        "output": 2.50,      # $2.50 per 1M output tokens
    },
    "gemini-2.0-flash": {
        "input": 0.10,
        "output": 0.40,
    },
    
    # ElevenLabs pricing (per character)
    # At Pro plan ($99/mo for 500K chars) = $0.000198/char
    # At Creator plan ($22/mo for 100K chars) = $0.00022/char
    # Using approximate middle ground
    "elevenlabs": {
        "per_character": 0.0002,  # ~$0.20 per 1000 characters
    },
    
    # Google Search grounding
    "google_search": {
        "per_request": 0.035,  # $35 per 1000 requests (after free tier)
        "free_daily": 1500,    # 1500 free requests per day
    }
}


class ExpenseTracker:
    """
    Tracks API expenses for Companion.
    Thread-safe for use in async environment.
    """
    
    def __init__(self, log_path: str = "expense_log.md"):
        """
        Initialize the expense tracker.
        
        Args:
            log_path: Path to the expense log file
        """
        self.log_path = Path(log_path)
        self.data_path = Path(".expense_data.json")  # Internal JSON for fast reads
        self._lock = Lock()
        
        # Auto-report settings
        self._log_count = 0
        self._auto_report_interval = 50  # Write report every 50 log calls
        self._last_report_time = None
        
        # Load existing data or initialize
        self._data = self._load_data()
        
        # Today's stats (for quick access)
        self._today = datetime.now().strftime("%Y-%m-%d")
        if self._today not in self._data["daily"]:
            self._data["daily"][self._today] = self._empty_day()
    
    def _empty_day(self) -> Dict[str, Any]:
        """Create an empty day record."""
        return {
            "gemini": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "calls": 0,
                "models": {}
            },
            "elevenlabs": {
                "characters": 0,
                "cost": 0.0,
                "calls": 0
            },
            "google_search": {
                "requests": 0,
                "cost": 0.0
            },
            "total_cost": 0.0
        }
    
    def _load_data(self) -> Dict[str, Any]:
        """Load existing data from JSON file."""
        if self.data_path.exists():
            try:
                with open(self.data_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Initialize new data structure
        return {
            "daily": {},
            "lifetime": {
                "gemini_tokens": 0,
                "gemini_cost": 0.0,
                "elevenlabs_characters": 0,
                "elevenlabs_cost": 0.0,
                "search_requests": 0,
                "search_cost": 0.0,
                "total_cost": 0.0,
                "first_tracked": datetime.now().isoformat()
            }
        }
    
    def _save_data(self):
        """Save data to JSON file (internal, fast)."""
        try:
            with open(self.data_path, 'w') as f:
                json.dump(self._data, f, indent=2)
        except IOError as e:
            print(f"⚠️  Could not save expense data: {e}")
        
        # Auto-report: write human-readable report periodically
        self._log_count += 1
        if self._log_count >= self._auto_report_interval:
            self._log_count = 0
            self.write_expense_report()
    
    def _ensure_today(self):
        """Ensure today's record exists."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._today:
            self._today = today
            if today not in self._data["daily"]:
                self._data["daily"][today] = self._empty_day()
    
    def log_gemini_usage(self, 
                         model: str,
                         input_tokens: int,
                         output_tokens: int,
                         cached_tokens: int = 0):
        """
        Log Gemini API usage.
        
        Args:
            model: Model name (e.g., "gemini-3-pro-preview")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached tokens (reduced cost)
        """
        with self._lock:
            self._ensure_today()
            
            # Calculate cost
            pricing = PRICING.get(model, PRICING["gemini-2.5-pro"])
            
            # Effective input tokens (cached tokens are cheaper)
            effective_input = input_tokens - cached_tokens
            input_cost = (effective_input / 1_000_000) * pricing["input"]
            
            # Cached tokens cost less (typically 25% of normal)
            if cached_tokens > 0:
                input_cost += (cached_tokens / 1_000_000) * pricing["input"] * 0.25
            
            output_cost = (output_tokens / 1_000_000) * pricing["output"]
            total_cost = input_cost + output_cost
            
            # Update today's stats
            day = self._data["daily"][self._today]
            day["gemini"]["input_tokens"] += input_tokens
            day["gemini"]["output_tokens"] += output_tokens
            day["gemini"]["cost"] += total_cost
            day["gemini"]["calls"] += 1
            
            # Track per-model usage
            if model not in day["gemini"]["models"]:
                day["gemini"]["models"][model] = {"input": 0, "output": 0, "cost": 0.0, "calls": 0}
            day["gemini"]["models"][model]["input"] += input_tokens
            day["gemini"]["models"][model]["output"] += output_tokens
            day["gemini"]["models"][model]["cost"] += total_cost
            day["gemini"]["models"][model]["calls"] += 1
            
            day["total_cost"] += total_cost
            
            # Update lifetime stats
            self._data["lifetime"]["gemini_tokens"] += input_tokens + output_tokens
            self._data["lifetime"]["gemini_cost"] += total_cost
            self._data["lifetime"]["total_cost"] += total_cost
            
            self._save_data()
    
    def log_elevenlabs_usage(self, characters: int):
        """
        Log ElevenLabs TTS usage.
        
        Args:
            characters: Number of characters synthesized
        """
        with self._lock:
            self._ensure_today()
            
            # Calculate cost
            cost = characters * PRICING["elevenlabs"]["per_character"]
            
            # Update today's stats
            day = self._data["daily"][self._today]
            day["elevenlabs"]["characters"] += characters
            day["elevenlabs"]["cost"] += cost
            day["elevenlabs"]["calls"] += 1
            day["total_cost"] += cost
            
            # Update lifetime stats
            self._data["lifetime"]["elevenlabs_characters"] += characters
            self._data["lifetime"]["elevenlabs_cost"] += cost
            self._data["lifetime"]["total_cost"] += cost
            
            self._save_data()
    
    def log_search_usage(self, requests: int = 1):
        """
        Log Google Search grounding usage.
        
        Args:
            requests: Number of search requests
        """
        with self._lock:
            self._ensure_today()
            
            # Check if still in free tier for today
            day = self._data["daily"][self._today]
            current_requests = day["google_search"]["requests"]
            free_tier = PRICING["google_search"]["free_daily"]
            
            # Calculate billable requests (after free tier)
            total_after = current_requests + requests
            if total_after > free_tier:
                billable = total_after - max(current_requests, free_tier)
                cost = billable * PRICING["google_search"]["per_request"]
            else:
                cost = 0.0
            
            day["google_search"]["requests"] += requests
            day["google_search"]["cost"] += cost
            day["total_cost"] += cost
            
            # Update lifetime stats
            self._data["lifetime"]["search_requests"] += requests
            self._data["lifetime"]["search_cost"] += cost
            self._data["lifetime"]["total_cost"] += cost
            
            self._save_data()
    
    def get_today_summary(self) -> Dict[str, Any]:
        """Get today's expense summary."""
        with self._lock:
            self._ensure_today()
            return self._data["daily"].get(self._today, self._empty_day())
    
    def get_period_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get expense summary for the last N days.
        
        Args:
            days: Number of days to summarize
            
        Returns:
            Summary dict with totals
        """
        with self._lock:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            summary = {
                "period": f"Last {days} days",
                "gemini": {"tokens": 0, "cost": 0.0, "calls": 0},
                "elevenlabs": {"characters": 0, "cost": 0.0, "calls": 0},
                "search": {"requests": 0, "cost": 0.0},
                "total_cost": 0.0,
                "daily_average": 0.0
            }
            
            days_counted = 0
            current = start_date
            while current <= end_date:
                date_str = current.strftime("%Y-%m-%d")
                if date_str in self._data["daily"]:
                    day = self._data["daily"][date_str]
                    summary["gemini"]["tokens"] += day["gemini"]["input_tokens"] + day["gemini"]["output_tokens"]
                    summary["gemini"]["cost"] += day["gemini"]["cost"]
                    summary["gemini"]["calls"] += day["gemini"]["calls"]
                    summary["elevenlabs"]["characters"] += day["elevenlabs"]["characters"]
                    summary["elevenlabs"]["cost"] += day["elevenlabs"]["cost"]
                    summary["elevenlabs"]["calls"] += day["elevenlabs"]["calls"]
                    summary["search"]["requests"] += day["google_search"]["requests"]
                    summary["search"]["cost"] += day["google_search"]["cost"]
                    summary["total_cost"] += day["total_cost"]
                    days_counted += 1
                current += timedelta(days=1)
            
            if days_counted > 0:
                summary["daily_average"] = summary["total_cost"] / days_counted
            
            return summary
    
    def get_lifetime_summary(self) -> Dict[str, Any]:
        """Get lifetime expense summary."""
        with self._lock:
            return self._data["lifetime"].copy()
    
    def write_expense_report(self):
        """
        Write a human-readable expense report to expense_log.md.
        Called periodically or on demand.
        """
        with self._lock:
            today = self.get_today_summary()
            week = self.get_period_summary(7)
            month = self.get_period_summary(30)
            lifetime = self.get_lifetime_summary()
            
            report = f"""# Companion Expense Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Today's Usage ({self._today})

| Service | Usage | Cost |
|---------|-------|------|
| Gemini | {today['gemini']['input_tokens']:,} in / {today['gemini']['output_tokens']:,} out tokens | ${today['gemini']['cost']:.4f} |
| ElevenLabs | {today['elevenlabs']['characters']:,} characters | ${today['elevenlabs']['cost']:.4f} |
| Google Search | {today['google_search']['requests']} requests | ${today['google_search']['cost']:.4f} |
| **Total** | | **${today['total_cost']:.4f}** |

### Today's Model Breakdown
"""
            for model, stats in today['gemini'].get('models', {}).items():
                report += f"- **{model}**: {stats['calls']} calls, {stats['input']:,} in / {stats['output']:,} out, ${stats['cost']:.4f}\n"
            
            report += f"""
## Last 7 Days

| Service | Usage | Cost |
|---------|-------|------|
| Gemini | {week['gemini']['tokens']:,} tokens ({week['gemini']['calls']} calls) | ${week['gemini']['cost']:.4f} |
| ElevenLabs | {week['elevenlabs']['characters']:,} chars ({week['elevenlabs']['calls']} calls) | ${week['elevenlabs']['cost']:.4f} |
| Google Search | {week['search']['requests']} requests | ${week['search']['cost']:.4f} |
| **Total** | | **${week['total_cost']:.4f}** |

Daily Average: **${week['daily_average']:.4f}**

## Last 30 Days

| Service | Usage | Cost |
|---------|-------|------|
| Gemini | {month['gemini']['tokens']:,} tokens ({month['gemini']['calls']} calls) | ${month['gemini']['cost']:.4f} |
| ElevenLabs | {month['elevenlabs']['characters']:,} chars ({month['elevenlabs']['calls']} calls) | ${month['elevenlabs']['cost']:.4f} |
| Google Search | {month['search']['requests']} requests | ${month['search']['cost']:.4f} |
| **Total** | | **${month['total_cost']:.4f}** |

Daily Average: **${month['daily_average']:.4f}**
Monthly Projection: **${month['daily_average'] * 30:.2f}**

## Lifetime Totals
*Since {lifetime.get('first_tracked', 'unknown')[:10]}*

| Metric | Value |
|--------|-------|
| Gemini Tokens | {lifetime['gemini_tokens']:,} |
| Gemini Cost | ${lifetime['gemini_cost']:.4f} |
| ElevenLabs Characters | {lifetime['elevenlabs_characters']:,} |
| ElevenLabs Cost | ${lifetime['elevenlabs_cost']:.4f} |
| Search Requests | {lifetime['search_requests']:,} |
| Search Cost | ${lifetime['search_cost']:.4f} |
| **Total Cost** | **${lifetime['total_cost']:.4f}** |

---
*This report is automatically generated and not visible to Companion.*
*Pricing estimates based on published rates as of December 2024.*
"""
            
            try:
                with open(self.log_path, 'w') as f:
                    f.write(report)
            except IOError as e:
                print(f"⚠️  Could not write expense report: {e}")


# Global singleton instance
_tracker: Optional[ExpenseTracker] = None


def get_tracker() -> ExpenseTracker:
    """Get the global expense tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = ExpenseTracker()
    return _tracker


def log_gemini(model: str, input_tokens: int, output_tokens: int, cached: int = 0):
    """Convenience function to log Gemini usage."""
    get_tracker().log_gemini_usage(model, input_tokens, output_tokens, cached)


def log_elevenlabs(characters: int):
    """Convenience function to log ElevenLabs usage."""
    get_tracker().log_elevenlabs_usage(characters)


def log_search(requests: int = 1):
    """Convenience function to log Google Search usage."""
    get_tracker().log_search_usage(requests)


def write_report():
    """Convenience function to write expense report."""
    get_tracker().write_expense_report()


if __name__ == "__main__":
    # Test the expense tracker
    print("Testing Expense Tracker")
    print("=" * 50)
    
    tracker = ExpenseTracker()
    
    # Simulate some usage
    print("\n📊 Simulating API usage...")
    
    # Gemini usage
    tracker.log_gemini_usage("gemini-3-pro-preview", 1500, 500)
    tracker.log_gemini_usage("gemini-2.5-flash", 800, 200)
    print("   Logged Gemini usage")
    
    # ElevenLabs usage
    tracker.log_elevenlabs_usage(250)
    tracker.log_elevenlabs_usage(180)
    print("   Logged ElevenLabs usage")
    
    # Search usage
    tracker.log_search_usage(3)
    print("   Logged search usage")
    
    # Generate report
    print("\n📝 Generating expense report...")
    tracker.write_expense_report()
    
    # Show today's summary
    today = tracker.get_today_summary()
    print(f"\n📈 Today's Summary:")
    print(f"   Gemini: ${today['gemini']['cost']:.4f}")
    print(f"   ElevenLabs: ${today['elevenlabs']['cost']:.4f}")
    print(f"   Total: ${today['total_cost']:.4f}")
    
    print(f"\n✅ Report written to expense_log.md")
