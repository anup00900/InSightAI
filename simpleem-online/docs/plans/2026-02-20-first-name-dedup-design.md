# First-Name-Only Participant Display + Deduplication

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate participant duplicates by using first names only in the real-time UI, while keeping full names in DB and PDF reports.

**Architecture:** Add a `_first_name()` helper to extract display names. Use first names for all WebSocket-facing data (signals, speaking distribution, emotion timeline). Keep full names in `_name_map` for DB persistence and PDF export.

**Tech Stack:** Python (backend pipeline), TypeScript/React (frontend hooks)

---

## Problem

OCR reads inconsistent full name variants from Teams recordings:
- "Sarang Zendehrooh", "Sarang Zenderrooh", "Sarang Zenderhooh"
- "Christoph Schmell", "C Schmell"
- "Giorgio Manenti", "M Giorgio", "Giorgio"

These create 20+ duplicate participant entries in the UI instead of ~12 unique people.

## Solution

### 1. Backend: `_first_name()` helper

Add static method to `RealtimeSession`:

```python
@staticmethod
def _first_name(full_name: str) -> str:
    words = full_name.split()
    if not words:
        return full_name
    if len(words) >= 2 and len(words[0]) <= 2:
        return words[1]  # "M Giorgio" -> "Giorgio", "Nr Rangwan" -> "Rangwan"
    return words[0]  # "Sarang Zendehrooh" -> "Sarang"
```

### 2. Backend: Dedup via first name comparison

Change `_names_are_same_person()` to compare first names:

```python
a_first = cls._first_name(name_a).lower()
b_first = cls._first_name(name_b).lower()
if a_first == b_first and len(a_first) >= 3:
    return True
```

### 3. Backend: WebSocket sends first names

In `_analyze_frame_live()`, convert participant labels to first names before sending:

```python
for p in final_participants:
    p["label"] = self._first_name(p["label"])
```

And in `speaking_distribution`:
```python
speaking_dist[self._first_name(name)] = pct
```

### 4. Backend: name_map sends first names

When sending `name_map` to frontend, convert values to first names:

```python
display_map = {k: self._first_name(v) for k, v in self._name_map.items()}
await self._send_json({"type": "name_map", "data": display_map})
```

### 5. Backend: DB persistence keeps full names

`_persist_to_db()` continues using `participant_names` (full names from `_name_map`).

### 6. Backend: Speaking duration tracking uses first names

`_speaking_durations` dict keys must use first names (since visual labels and transcript labels both need to match):

```python
speaker_name = self._first_name(p["label"])
self._speaking_durations[speaker_name] = ...
```

### 7. Frontend: No changes needed

Since backend now sends first-name-only labels, all frontend components automatically display first names. The existing fuzzy matching in `useRealtimeAnalysis.ts` becomes simpler (single word exact match).

### 8. Vision model prompt: Request first names

In `analyze_frame_all_signals()`, update `name_instruction`:
```
"Use only the FIRST NAME as the 'label' field (e.g., 'Sarang' not 'Sarang Zendehrooh')."
```

In `extract_names_from_frame()`: no change needed — we still want full names for DB storage.

## Collision Handling

If two participants share the same first name (e.g., two "David"s):
- In `_merge_names()`, detect collision: if first names match but full names don't match ANY existing entry → suffix with number
- "David", "David 2"

## Files Changed

- `backend/realtime_pipeline.py` — add `_first_name()`, update `_names_are_same_person()`, update signal sending, update name_map sending
- `backend/core42_client.py` — update vision model prompt to request first names
- No frontend changes needed
