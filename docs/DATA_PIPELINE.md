# Data Collection & Quality Pipeline

## Overview
This pipeline collects CTF writeups from CTFTime.org and validates data quality.

## Database Configuration

**Collection:** `raw_writeups` (configured in `config/scraper_config.yaml`)

All scripts use `get_collection(db)` which reads from config - no hardcoded names.

## Pipeline Stages

### Stage 1: Metadata Collection
```bash
python3 the scraping pipeline (not in this release) --pages 50
```
- Scrapes CTFTime task/writeup pages
- Collects: event name, task name, tags, URLs
- Output: MongoDB `raw_writeups` collection with metadata

### Stage 2: Content Fetching
```bash
python3 the scraping pipeline (not in this release)
```
- Follows URLs to external sites
- Extracts writeup content using trafilatura
- Handles: GitHub, blogs, CTFTime-hosted
- Output: `markdown_content`, `raw_content`, `raw_html` fields

### Stage 3: Quality Validation
```bash
python3 the scraping pipeline (not in this release)
```
- Checks completeness, size distribution, duplicates
- Generates `data/quality_report.json`
- Provides recommendations

### Stage 4: Data Cleaning
```bash
python3 the scraping pipeline (not in this release)        # Dry run (preview)
python3 the scraping pipeline (not in this release) --apply  # Apply changes
```
- Flags stubs and duplicates (non-destructive)
- Adds `quality_flag` field
- Reversible (unset flag to restore)

## Data Quality Flags

| Flag | Meaning | Action |
|------|---------|--------|
| `stub` | <200 chars | Exclude from training |
| `duplicate` | Duplicate content | Keep original only |
| (none) | Good quality | Use for training |

## Phase 2: Standardization & Artifacts (Challenge-First Architecture)

To support System 2 training (Challenge -> Solution), we are transitioning to a relational schema.

### Architecture
*   **Collection: `challenges`** (The Question)
    *   Unique Entity per CTF Challenge.
    *   Contains: Name, Event, Description, Category.
    *   **Artifacts:** List of local file paths (binaries, source code) scraped from repos.
*   **Collection: `raw_writeups`** (The Answer)
    *   Contains: Standardized JSON (Goal, Steps, Code).
    *   Linked to `challenges` via `challenge_id`.

### Artifact Scraper Strategy
1.  **Fingerprinting:** Challenges are identified by unique `(Event, Task)` tuples or shared GitHub Source URLs.
2.  **Repo Scanning:** A specialized agent clones GitHub repositories linked in writeups.
3.  **Heuristic Extraction:**
    *   `*.c`, `*.py` -> Source Code.
    *   `Dockerfile` -> Environment.
    *   `chall`, `vuln`, `*.zip` -> Binaries.
4.  **Storage:** Files are stored in `/data/artifacts/{event_slug}/{challenge_slug}/`.

### Migration Plan
1.  **Preview:** Group existing writeups to identify unique challenges.
2.  **Create:** Populate `challenges` collection.
3.  **Link:** Update `raw_writeups` with `challenge_id`.
4.  **Fetch:** Run `RepoScanner` on unique challenges to populate artifacts.

## Document Schema

```python
{
    "_id": ObjectId,
    "source": "ctftime",
    # ... (existing fields) ...
    
    # New Architecture Fields
    "challenge_id": ObjectId,             # Link to 'challenges' collection
    "standardized_data": {
        "goal": str,
        "steps": list,
        "extracted_code": list,
        "flag": str
    }
}
```

## Analysis Commands
# ... (existing commands) ...