"""
Little Fig × Ember's Diaries — Embedded Memory Training

Trains an LLM to have cognitive memory capabilities built into its weights.
Not RAG (external retrieval) — the model itself learns to:
  1. STORE: Persist information across conversations with episodic segmentation
  2. RECALL: Retrieve relevant memories using learned attention patterns
  3. CONSOLIDATE: Merge related memories (sensory → short-term → long-term)
  4. FORGET: Apply Ebbinghaus-style decay to outdated information
  5. RESOLVE: Detect and handle conflicting memories

Training pipeline:
  1. Generate synthetic memory-operation training data from Ember's cognitive modules
  2. Format as instruction-following examples with special <memory> tokens
  3. Fine-tune using Fig Engine (LoRA/LISA) to teach the model memory operations
  4. The trained model can then operate its own Ember's Diaries instance

This makes the LLM a cognitive agent — not just a text predictor.
"""

import json
import random
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional


# ── Memory Operation Tags (injected into model vocabulary) ────────────────────

MEMORY_TOKENS = {
    "store":       "<|mem_store|>",
    "recall":      "<|mem_recall|>",
    "consolidate": "<|mem_consolidate|>",
    "forget":      "<|mem_forget|>",
    "conflict":    "<|mem_conflict|>",
    "episode":     "<|mem_episode|>",
    "reflect":     "<|mem_reflect|>",
    "mem_start":   "<|memory_start|>",
    "mem_end":     "<|memory_end|>",
}


# ── Training Data Generator ──────────────────────────────────────────────────

class EmberTrainingDataGenerator:
    """
    Generates training examples that teach an LLM to perform memory operations.

    Each example is a conversation where the model must:
    - Decide WHEN to store a memory (important info detection)
    - Decide WHAT to recall (relevance matching)
    - Decide HOW to consolidate (merge related memories)
    - Decide WHEN to forget (confidence decay)
    - Detect contradictions (conflict resolution)

    Output format: JSONL with instruction/input/output fields,
    compatible with Fig Engine's trainer.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._templates = self._build_templates()

    def generate_dataset(self, n_examples: int = 1000) -> list[dict]:
        """Generate a full training dataset for memory-aware fine-tuning."""
        examples = []
        generators = [
            (self._gen_store_example, 0.25),
            (self._gen_recall_example, 0.25),
            (self._gen_consolidate_example, 0.15),
            (self._gen_forget_example, 0.10),
            (self._gen_conflict_example, 0.10),
            (self._gen_episode_boundary_example, 0.10),
            (self._gen_reflect_example, 0.05),
        ]

        for _ in range(n_examples):
            # Weighted random selection
            r = self.rng.random()
            cumulative = 0
            for gen_fn, weight in generators:
                cumulative += weight
                if r < cumulative:
                    examples.append(gen_fn())
                    break

        return examples

    def generate_jsonl(self, n_examples: int = 1000, path: str = "ember_memory_train.jsonl"):
        """Generate and save training data as JSONL."""
        examples = self.generate_dataset(n_examples)
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"🔥 Generated {len(examples)} Ember memory training examples → {path}")
        return path

    # ── Store: teach the model to identify and persist important info ──────

    def _gen_store_example(self) -> dict:
        facts = self._templates["facts"]
        fact = self.rng.choice(facts)
        return {
            "instruction": "You have memory capabilities. When the user shares important information, store it using <|mem_store|>.",
            "input": f"User: {fact['statement']}",
            "output": (
                f"{MEMORY_TOKENS['mem_start']}\n"
                f"{MEMORY_TOKENS['store']} namespace=\"{fact['namespace']}\" "
                f"tags=[{', '.join(fact['tags'])}] "
                f"confidence={fact['confidence']}\n"
                f"content: {fact['statement']}\n"
                f"{MEMORY_TOKENS['mem_end']}\n\n"
                f"I've noted that. {fact['acknowledgment']}"
            ),
        }

    # ── Recall: teach the model to search memories when relevant ──────────

    def _gen_recall_example(self) -> dict:
        scenarios = self._templates["recall_scenarios"]
        s = self.rng.choice(scenarios)
        return {
            "instruction": "You have access to stored memories. When a question relates to past information, recall it using <|mem_recall|>.",
            "input": f"Previous context: {s['context']}\n\nUser: {s['question']}",
            "output": (
                f"{MEMORY_TOKENS['mem_start']}\n"
                f"{MEMORY_TOKENS['recall']} query=\"{s['query']}\" namespace=\"{s['namespace']}\"\n"
                f"Found: {s['memory_content']}\n"
                f"{MEMORY_TOKENS['mem_end']}\n\n"
                f"{s['response']}"
            ),
        }

    # ── Consolidate: teach the model to merge related memories ────────────

    def _gen_consolidate_example(self) -> dict:
        c = self.rng.choice(self._templates["consolidations"])
        return {
            "instruction": "When you notice related memories that should be merged, consolidate them using <|mem_consolidate|>.",
            "input": f"Memory 1: {c['mem1']}\nMemory 2: {c['mem2']}",
            "output": (
                f"{MEMORY_TOKENS['mem_start']}\n"
                f"{MEMORY_TOKENS['consolidate']} sources=[mem_1, mem_2] "
                f"stage=\"{c['stage']}\"\n"
                f"consolidated: {c['merged']}\n"
                f"confidence_boost: +0.1\n"
                f"decay_rate: reduced by 50%\n"
                f"{MEMORY_TOKENS['mem_end']}\n\n"
                f"{c['response']}"
            ),
        }

    # ── Forget: teach the model about confidence decay ────────────────────

    def _gen_forget_example(self) -> dict:
        f = self.rng.choice(self._templates["decay_scenarios"])
        return {
            "instruction": "Memories have confidence that decays over time. Flag low-confidence memories using <|mem_forget|>.",
            "input": f"Recalling memory: {f['memory']}\nLast accessed: {f['last_accessed']}\nConfidence: {f['confidence']}",
            "output": (
                f"{MEMORY_TOKENS['mem_start']}\n"
                f"{MEMORY_TOKENS['forget']} record_id=\"{f['record_id']}\" "
                f"effective_confidence={f['effective_conf']}\n"
                f"action: {f['action']}\n"
                f"{MEMORY_TOKENS['mem_end']}\n\n"
                f"{f['response']}"
            ),
        }

    # ── Conflict: teach the model to detect contradictions ────────────────

    def _gen_conflict_example(self) -> dict:
        c = self.rng.choice(self._templates["conflicts"])
        return {
            "instruction": "Detect contradictions between memories. Flag conflicts using <|mem_conflict|>.",
            "input": f"Memory A: {c['mem_a']}\nMemory B: {c['mem_b']}",
            "output": (
                f"{MEMORY_TOKENS['mem_start']}\n"
                f"{MEMORY_TOKENS['conflict']} type=\"{c['conflict_type']}\" severity={c['severity']}\n"
                f"description: {c['description']}\n"
                f"resolution: {c['resolution']}\n"
                f"{MEMORY_TOKENS['mem_end']}\n\n"
                f"{c['response']}"
            ),
        }

    # ── Episode boundary: teach the model to segment conversations ────────

    def _gen_episode_boundary_example(self) -> dict:
        e = self.rng.choice(self._templates["episode_boundaries"])
        return {
            "instruction": "Detect when a conversation shifts to a new topic. Mark episode boundaries using <|mem_episode|>.",
            "input": f"Previous topic: {e['prev_topic']}\nNew message: {e['new_message']}",
            "output": (
                f"{MEMORY_TOKENS['mem_start']}\n"
                f"{MEMORY_TOKENS['episode']} boundary=true "
                f"reason=\"{e['reason']}\"\n"
                f"prev_episode_summary: {e['prev_summary']}\n"
                f"new_episode_tags: [{', '.join(e['new_tags'])}]\n"
                f"{MEMORY_TOKENS['mem_end']}\n\n"
                f"{e['response']}"
            ),
        }

    # ── Reflect: teach the model to self-reflect on memories ──────────────

    def _gen_reflect_example(self) -> dict:
        r = self.rng.choice(self._templates["reflections"])
        return {
            "instruction": "Periodically reflect on stored memories to generate insights using <|mem_reflect|>.",
            "input": f"Memory store contains {r['count']} records. Key themes: {', '.join(r['themes'])}",
            "output": (
                f"{MEMORY_TOKENS['mem_start']}\n"
                f"{MEMORY_TOKENS['reflect']}\n"
                f"insight: {r['insight']}\n"
                f"action: {r['action']}\n"
                f"new_connections: {r['connections']}\n"
                f"{MEMORY_TOKENS['mem_end']}\n\n"
                f"{r['response']}"
            ),
        }

    # ── Templates ─────────────────────────────────────────────────────────

    def _build_templates(self) -> dict:
        return {
            "facts": [
                {"statement": "My name is Alex and I'm a software engineer at Google.", "namespace": "personal", "tags": ["name", "occupation", "company"], "confidence": 0.95, "acknowledgment": "I'll remember your name and that you work at Google as a software engineer."},
                {"statement": "I have a meeting with the design team every Tuesday at 2pm.", "namespace": "schedule", "tags": ["meeting", "recurring", "design"], "confidence": 0.9, "acknowledgment": "Got it — Tuesday design meetings at 2pm, noted."},
                {"statement": "I'm allergic to peanuts and shellfish.", "namespace": "health", "tags": ["allergy", "food", "important"], "confidence": 0.99, "acknowledgment": "That's important — I've stored your peanut and shellfish allergies with high priority."},
                {"statement": "My project deadline is March 15th for the API migration.", "namespace": "work", "tags": ["deadline", "project", "api"], "confidence": 0.85, "acknowledgment": "March 15th for the API migration — I'll keep track of that."},
                {"statement": "I prefer Python over JavaScript for backend work.", "namespace": "preferences", "tags": ["programming", "preference", "backend"], "confidence": 0.8, "acknowledgment": "Noted — Python is your go-to for backend development."},
                {"statement": "My daughter's birthday is June 12th.", "namespace": "personal", "tags": ["family", "birthday", "important"], "confidence": 0.95, "acknowledgment": "June 12th — your daughter's birthday. I'll remember that."},
                {"statement": "I've been learning Rust for the past 3 months.", "namespace": "learning", "tags": ["programming", "rust", "ongoing"], "confidence": 0.85, "acknowledgment": "Three months into Rust — I'll keep that in mind for future conversations."},
                {"statement": "The production database runs on PostgreSQL 15.", "namespace": "infrastructure", "tags": ["database", "production", "postgres"], "confidence": 0.9, "acknowledgment": "PostgreSQL 15 in production — stored for reference."},
            ],
            "recall_scenarios": [
                {"context": "User mentioned being a software engineer at Google.", "question": "What do I do for a living?", "query": "occupation", "namespace": "personal", "memory_content": "Software engineer at Google (confidence: 0.95)", "response": "You're a software engineer at Google — you mentioned that earlier."},
                {"context": "User said they have a meeting every Tuesday at 2pm.", "question": "When is my next meeting?", "query": "meeting schedule", "namespace": "schedule", "memory_content": "Design team meeting, Tuesday 2pm, recurring (confidence: 0.9)", "response": "Your design team meeting is every Tuesday at 2pm."},
                {"context": "User mentioned a project deadline.", "question": "When is the API migration due?", "query": "api migration deadline", "namespace": "work", "memory_content": "March 15th, API migration deadline (confidence: 0.85)", "response": "The API migration is due March 15th."},
                {"context": "User discussed their tech preferences.", "question": "Should I use Python or JS for this backend service?", "query": "backend language preference", "namespace": "preferences", "memory_content": "Prefers Python over JavaScript for backend (confidence: 0.8)", "response": "Based on your preference, I'd suggest Python — you mentioned you prefer it for backend work."},
            ],
            "consolidations": [
                {"mem1": "User works at Google", "mem2": "User is a senior engineer leading the search team", "stage": "short_term→long_term", "merged": "Senior software engineer at Google, leads search team", "response": "I've consolidated your work information — you're a senior engineer leading Google's search team."},
                {"mem1": "Project deadline is March 15", "mem2": "API migration involves 3 microservices", "stage": "sensory→short_term", "merged": "API migration: 3 microservices, deadline March 15", "response": "I've linked your deadline with the project scope — 3 microservices by March 15th."},
            ],
            "decay_scenarios": [
                {"memory": "User's favorite restaurant is Nobu", "last_accessed": "6 months ago", "confidence": 0.8, "effective_conf": 0.35, "record_id": "rec_abc123", "action": "flag_for_reinforcement", "response": "I have a memory about your restaurant preference, but it's been a while — is Nobu still your favorite?"},
                {"memory": "User's project uses React 17", "last_accessed": "1 year ago", "confidence": 0.7, "effective_conf": 0.15, "record_id": "rec_def456", "action": "deprecate", "response": "I had a note about React 17, but that's quite old — you may have upgraded since then."},
            ],
            "conflicts": [
                {"mem_a": "User works at Google", "mem_b": "User just started at Meta", "conflict_type": "value_mismatch", "severity": 0.9, "description": "Employer field differs: Google vs Meta", "resolution": "Update to Meta (more recent)", "response": "I notice a conflict — I had you at Google, but you just said Meta. I'll update to your current employer."},
                {"mem_a": "Meeting is Tuesday at 2pm", "mem_b": "Meeting moved to Wednesday at 3pm", "conflict_type": "value_mismatch", "severity": 0.7, "description": "Meeting time changed", "resolution": "Update to Wednesday 3pm, keep old as history", "response": "Got it — your design meeting moved from Tuesday 2pm to Wednesday 3pm. I've updated that."},
            ],
            "episode_boundaries": [
                {"prev_topic": "Discussing Python code optimization", "new_message": "By the way, what should I get my wife for our anniversary?", "reason": "Topic shift: technical → personal", "prev_summary": "Code optimization discussion with profiling tips", "new_tags": ["personal", "anniversary", "gift"], "response": "Switching gears! For an anniversary gift, here are some ideas..."},
                {"prev_topic": "Database migration planning", "new_message": "I need to prepare for my job interview tomorrow", "reason": "Topic shift: work project → career", "prev_summary": "PostgreSQL migration plan with timeline", "new_tags": ["career", "interview", "preparation"], "response": "Let's focus on your interview prep. What role is it for?"},
            ],
            "reflections": [
                {"count": 47, "themes": ["work", "programming", "health"], "insight": "User is under deadline pressure (March 15) while learning Rust — may need focused help", "action": "prioritize deadline-related queries", "connections": "Rust learning may relate to the API migration project", "response": "Reflecting on our conversations, I notice you're balancing the March 15th deadline with learning Rust. Want me to help you prioritize?"},
                {"count": 23, "themes": ["family", "schedule", "preferences"], "insight": "User values family time (daughter's birthday noted) and has a structured schedule", "action": "be mindful of scheduling conflicts with family events", "connections": "Tuesday meetings might conflict with family activities", "response": "I've been thinking about your schedule — your daughter's birthday is coming up in June. Want me to remind you closer to the date?"},
            ],
        }


# ── Ember-Aware Chat Wrapper ─────────────────────────────────────────────────

class EmberChatManager:
    """
    Wraps Ember's Diaries for use with Little Fig's chat system.
    Stores conversations as episodic memories, enables recall across sessions.

    Usage:
        manager = EmberChatManager("./fig_memory")
        # On each message:
        manager.store_message(chat_id, role, content)
        # For context:
        relevant = manager.recall(query, limit=5)
    """

    def __init__(self, store_path: str = ".fig_memory"):
        self._path = store_path
        self._db = None
        self._init_db()

    def _init_db(self):
        try:
            from embers import EmberDB
            self._db = EmberDB.connect(self._path)
        except ImportError:
            print("⚠ embers-diaries not installed. Memory features disabled.")
            print("  Install: pip install embers-diaries")

    def store_message(self, chat_id: str, role: str, content: str,
                      tags: list = None):
        if not self._db:
            return None
        from embers import EmberRecord, RecordType
        record = EmberRecord(
            namespace=f"chat/{chat_id}",
            record_type=RecordType.DOCUMENT,
            data={"role": role, "content": content, "chat_id": chat_id},
            tags=(tags or []) + [role, "chat"],
            confidence=0.9 if role == "user" else 0.7,
        )
        return self._db.write(record)

    def recall(self, query: str, limit: int = 5, namespace: str = None):
        if not self._db:
            return []
        try:
            results = self._db.query().search(query, limit=limit)
            return [r.data for r in results]
        except Exception:
            return []

    def get_chat_history(self, chat_id: str, limit: int = 50):
        if not self._db:
            return []
        try:
            results = self._db.query().namespace(f"chat/{chat_id}").limit(limit).execute()
            return [r.data for r in results]
        except Exception:
            return []
