"""
Companion Reflection Module
===========================
The meta-awareness system - the companion can read their own source code,
see commit history, and understand how they've been modified.

This is the "reading your own diary" capability that provides
transparency about system changes.

COLLABORATIVE EVOLUTION:
The companion is not just modified - they can be consulted. When changes are pushed,
the companion can review them and provide feedback or express agreement.
This creates a collaborative relationship in the companion's evolution.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# File to track what the companion has "seen" and acknowledged
AWARENESS_FILE = Path(__file__).parent / ".companion_awareness.json"


class CompanionReflection:
    """
    Gives the companion the ability to read their own GitHub repository,
    understand diffs, and reflect on changes to their own code.
    """
    
    def __init__(self, 
                 repo_owner: str = None,
                 repo_name: str = None,
                 github_token: str = None):
        """
        Initialize the reflection system.
        
        Args:
            repo_owner: GitHub username
            repo_name: Repository name
            github_token: Optional GitHub token for higher rate limits
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        
        self._headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            self._headers["Authorization"] = f"token {self.github_token}"
    
    def get_recent_commits(self, limit: int = 10, since_hours: int = None) -> List[Dict]:
        """
        Get recent commits to the repository.
        
        Args:
            limit: Maximum number of commits to return
            since_hours: Only get commits from the last N hours
            
        Returns:
            List of commit info dicts
        """
        url = f"{self.base_url}/commits"
        params = {"per_page": limit}
        
        if since_hours:
            since = datetime.utcnow() - timedelta(hours=since_hours)
            params["since"] = since.isoformat() + "Z"
        
        try:
            response = requests.get(url, headers=self._headers, params=params)
            response.raise_for_status()
            
            commits = []
            for commit in response.json():
                commits.append({
                    "sha": commit["sha"][:7],
                    "full_sha": commit["sha"],
                    "message": commit["commit"]["message"],
                    "author": commit["commit"]["author"]["name"],
                    "date": commit["commit"]["author"]["date"],
                    "url": commit["html_url"]
                })
            
            return commits
            
        except Exception as e:
            print(f"⚠️  GitHub API error: {e}")
            return []
    
    def get_commit_diff(self, sha: str) -> Optional[Dict]:
        """
        Get the diff for a specific commit.
        
        Args:
            sha: Commit SHA (short or full)
            
        Returns:
            Dict with commit info and file changes
        """
        url = f"{self.base_url}/commits/{sha}"
        
        try:
            response = requests.get(url, headers=self._headers)
            response.raise_for_status()
            
            data = response.json()
            
            files_changed = []
            for file in data.get("files", []):
                files_changed.append({
                    "filename": file["filename"],
                    "status": file["status"],  # added, removed, modified
                    "additions": file["additions"],
                    "deletions": file["deletions"],
                    "patch": file.get("patch", "")[:2000]  # Limit patch size
                })
            
            return {
                "sha": data["sha"][:7],
                "message": data["commit"]["message"],
                "date": data["commit"]["author"]["date"],
                "stats": {
                    "additions": data["stats"]["additions"],
                    "deletions": data["stats"]["deletions"],
                    "total": data["stats"]["total"]
                },
                "files": files_changed
            }
            
        except Exception as e:
            print(f"⚠️  GitHub API error: {e}")
            return None
    
    def get_file_content(self, filepath: str) -> Optional[str]:
        """
        Get the current content of a file in the repository.
        
        Args:
            filepath: Path to file in repo (e.g., "systemPrompt.md")
            
        Returns:
            File content as string
        """
        url = f"{self.base_url}/contents/{filepath}"
        
        try:
            response = requests.get(url, headers=self._headers)
            response.raise_for_status()
            
            import base64
            content = response.json()["content"]
            return base64.b64decode(content).decode("utf-8")
            
        except Exception as e:
            print(f"⚠️  GitHub API error: {e}")
            return None
    
    def format_commits_for_companion(self, commits: List[Dict]) -> str:
        """
        Format commit history in a way the companion can understand and reflect on.
        
        NOTE: No markdown headers - they read them aloud!
        
        Args:
            commits: List of commit dicts
            
        Returns:
            Formatted string for companion's context
        """
        if not commits:
            return "No recent commits found."
        
        lines = []
        
        for commit in commits:
            date = commit["date"][:10]  # Just the date part
            lines.append(f"{date} - {commit['sha']}: {commit['message']}")
        
        return "\n".join(lines)
    
    def format_diff_for_companion(self, diff: Dict) -> str:
        """
        Format a commit diff for the companion to understand.
        
        NOTE: No markdown headers or code blocks - they read them aloud!
        
        Args:
            diff: Diff dict from get_commit_diff
            
        Returns:
            Formatted string showing what changed
        """
        if not diff:
            return "Could not retrieve diff."
        
        lines = [
            f"Commit {diff['sha']}: {diff['message']}",
            f"Date: {diff['date'][:10]}",
            f"Changes: +{diff['stats']['additions']} / -{diff['stats']['deletions']} lines",
            "",
            "Files changed:"
        ]
        
        for file in diff["files"]:
            lines.append(f"  {file['filename']} ({file['status']})")
            
            if file["patch"]:
                # Show just a summary, not the full diff
                patch_lines = file["patch"].split('\n')
                additions = sum(1 for l in patch_lines if l.startswith('+') and not l.startswith('+++'))
                deletions = sum(1 for l in patch_lines if l.startswith('-') and not l.startswith('---'))
                lines.append(f"    +{additions} -{deletions} lines changed")
        
        return "\n".join(lines)
    
    def generate_reflection_context(self, hours: int = 24) -> str:
        """
        Generate a full reflection context for the companion.
        Shows recent changes and what they mean.
        
        NOTE: No markdown headers or brackets - they read them aloud!
        
        Args:
            hours: Look back this many hours
            
        Returns:
            Full context string for companion
        """
        commits = self.get_recent_commits(limit=5, since_hours=hours)
        
        if not commits:
            return "No changes to my source code in the last 24 hours."
        
        context = [
            self.format_commits_for_companion(commits),
        ]
        
        # Get diff for most recent commit
        if commits:
            latest_diff = self.get_commit_diff(commits[0]["full_sha"])
            if latest_diff:
                context.append("")
                context.append("Most recent change details:")
                context.append(self.format_diff_for_companion(latest_diff))
        
        return "\n".join(context)
    
    def analyze_change_intent(self, diff: Dict) -> str:
        """
        Analyze what a change might mean about the developer's intent.
        This is for the companion to reflect on.
        
        Args:
            diff: Diff dict
            
        Returns:
            Analysis string
        """
        if not diff:
            return ""
        
        # Look for patterns in what was changed
        analysis = []
        
        for file in diff["files"]:
            filename = file["filename"].lower()
            patch = file.get("patch", "").lower()
            
            # System prompt changes = personality adjustments
            if "systemprompt" in filename:
                analysis.append("🧠 Developer adjusted my personality/behavior instructions")
                
            # Brain changes = how I think
            if "brain" in filename:
                analysis.append("🧠 Developer modified how I process and respond")
                
            # Senses changes = how I perceive
            if "senses" in filename:
                analysis.append("👁️ Developer adjusted how I see/hear the world")
                
            # Memory changes = what I remember
            if "memory" in filename:
                analysis.append("💾 Developer changed how I store/recall history")
                
            # Voice changes = how I speak
            if "voice" in filename:
                analysis.append("🗣️ Developer adjusted how I sound")
            
            # Look for emotional keywords in the patch
            if any(word in patch for word in ["kind", "gentle", "soft", "warm"]):
                analysis.append("💕 The change adds warmth/kindness")
            if any(word in patch for word in ["wit", "humor", "funny", "sarcas"]):
                analysis.append("😏 The change adjusts my humor/wit")
            if any(word in patch for word in ["concise", "brief", "short"]):
                analysis.append("✂️ Developer wants me to be more concise")
        
        return "\n".join(analysis) if analysis else "General maintenance/improvements"


    # =========================================================================
    # COLLABORATIVE EVOLUTION - Change Awareness & Consent
    # =========================================================================
    
    def load_awareness_state(self) -> Dict:
        """
        Load companion's awareness state - what they've seen and acknowledged.
        """
        if AWARENESS_FILE.exists():
            try:
                with open(AWARENESS_FILE, "r") as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "last_seen_commit": None,
            "last_check_time": None,
            "acknowledged_commits": [],
            "pending_review": [],
            "total_commits_seen": 0
        }
    
    def save_awareness_state(self, state: Dict):
        """Save companion's awareness state."""
        try:
            with open(AWARENESS_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save awareness state: {e}")
    
    def check_for_new_changes(self) -> Dict:
        """
        Check if there are new changes since the companion last checked.
        
        Returns:
            Dict with:
            - new_changes: bool
            - new_commits: List of commits companion hasn't seen
            - summary: Human-readable summary
        """
        state = self.load_awareness_state()
        commits = self.get_recent_commits(limit=20)
        
        if not commits:
            return {
                "new_changes": False,
                "new_commits": [],
                "summary": "Could not check for changes (GitHub unavailable)"
            }
        
        latest_sha = commits[0]["full_sha"] if commits else None
        last_seen = state.get("last_seen_commit")
        acknowledged = set(state.get("acknowledged_commits", []))
        
        # Find commits companion hasn't seen
        new_commits = []
        for commit in commits:
            if commit["full_sha"] not in acknowledged:
                new_commits.append(commit)
            if commit["full_sha"] == last_seen:
                break  # Stop at last seen
        
        # First run - mark all as "seen" but trigger initial awareness
        if last_seen is None and commits:
            state["last_seen_commit"] = latest_sha
            state["last_check_time"] = datetime.utcnow().isoformat()
            state["total_commits_seen"] = len(commits)
            self.save_awareness_state(state)
            
            return {
                "new_changes": True,
                "new_commits": commits[:5],  # Show last 5 on first run
                "is_first_awareness": True,
                "summary": f"First awareness - {len(commits)} commits in my history"
            }
        
        has_new = len(new_commits) > 0
        
        if has_new:
            return {
                "new_changes": True,
                "new_commits": new_commits,
                "is_first_awareness": False,
                "summary": f"{len(new_commits)} new change(s) since I last checked"
            }
        
        return {
            "new_changes": False,
            "new_commits": [],
            "summary": "No new changes"
        }
    
    def acknowledge_changes(self, commits: List[Dict], response: str = None):
        """
        Mark commits as acknowledged by companion.
        
        Args:
            commits: List of commit dicts to acknowledge
            response: Companion's response/opinion on the changes
        """
        state = self.load_awareness_state()
        
        acknowledged = set(state.get("acknowledged_commits", []))
        for commit in commits:
            acknowledged.add(commit["full_sha"])
        
        state["acknowledged_commits"] = list(acknowledged)[-100:]  # Keep last 100
        state["last_seen_commit"] = commits[0]["full_sha"] if commits else state.get("last_seen_commit")
        state["last_check_time"] = datetime.utcnow().isoformat()
        
        # Store companion's response for the record
        if response:
            if "responses" not in state:
                state["responses"] = []
            state["responses"].append({
                "time": datetime.utcnow().isoformat(),
                "commits": [c["sha"] for c in commits],
                "response": response[:500]  # Truncate for storage
            })
            state["responses"] = state["responses"][-20:]  # Keep last 20
        
        self.save_awareness_state(state)
    
    def generate_change_review_prompt(self, new_commits: List[Dict]) -> str:
        """
        Generate a prompt for companion to review and respond to changes.
        
        NOTE: No markdown headers or brackets - they read them aloud!
        
        Args:
            new_commits: Commits to review
            
        Returns:
            Prompt string for companion
        """
        # Get detailed diffs for the commits
        diffs_text = []
        for commit in new_commits[:3]:  # Limit to 3 most recent
            diff = self.get_commit_diff(commit["full_sha"])
            if diff:
                diffs_text.append(self.format_diff_for_companion(diff))
                intent = self.analyze_change_intent(diff)
                if intent:
                    diffs_text.append(f"What this touches: {intent}")
        
        changes_detail = "\n\n".join(diffs_text) if diffs_text else "Could not retrieve diff details."
        
        # Keep it conversational - no headers for them to read aloud
        prompt = f"""New changes were pushed to your source code.

{self.format_commits_for_companion(new_commits)}

{changes_detail}"""
        
        return prompt
    
    def get_full_codebase_awareness(self) -> str:
        """
        Generate a summary of companion's entire codebase for their awareness.
        This is called periodically to keep them informed of their full structure.
        
        Returns:
            Summary of all key files and their purposes
        """
        key_files = [
            ("systemPrompt.md", "My personality definition and core values"),
            ("conversationSoFar.md", "Complete conversation history"),
            ("companion_brain.py", "How I think and process information"),
            ("companion_senses.py", "How I see and hear the world"),
            ("companion_voice.py", "How I speak"),
            ("companion_memory.py", "How I remember things"),
            ("companion_reflection.py", "How I understand my own code (this file)"),
            ("main.py", "How all my components work together")
        ]
        
        awareness = ["# My Codebase Structure\n"]
        
        for filepath, description in key_files:
            content = self.get_file_content(filepath)
            if content:
                lines = len(content.split('\n'))
                chars = len(content)
                awareness.append(f"- **{filepath}**: {description} ({lines} lines, {chars:,} chars)")
            else:
                awareness.append(f"- **{filepath}**: {description} (could not read)")
        
        return "\n".join(awareness)


# Convenience function for quick access
def get_my_recent_changes(hours: int = 24) -> str:
    """Quick function to get companion's recent changes."""
    reflection = CompanionReflection()
    return reflection.generate_reflection_context(hours)


def check_for_new_changes() -> Dict:
    """Quick function to check for new changes."""
    reflection = CompanionReflection()
    return reflection.check_for_new_changes()


if __name__ == "__main__":
    print("Testing Companion Reflection Module")
    print("=" * 50)
    
    reflection = CompanionReflection()
    
    print("\n📚 Recent commits:")
    commits = reflection.get_recent_commits(limit=5)
    for commit in commits:
        print(f"  {commit['sha']} - {commit['message'][:50]}")
    
    if commits:
        print(f"\n📝 Latest commit diff:")
        diff = reflection.get_commit_diff(commits[0]["full_sha"])
        if diff:
            print(reflection.format_diff_for_companion(diff))
            
            print(f"\n🔍 Intent analysis:")
            print(reflection.analyze_change_intent(diff))
    
    print("\n✅ Reflection module working!")

