#!/usr/bin/env python3
"""
PicoGym Scraper - ported from ctf-solver/scripts/pico/scrape_picogym.py.

Provides functions to:
1. Log into PicoCTF (manual CAPTCHA)
2. Scrape challenge cards from the Gym (paginated)
3. Expand challenge modals for full details (description, hints, files)
4. Download challenge artifacts to a local directory
"""
import os
import re
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# JavaScript extraction snippets (carried over from the original scraper)
# ---------------------------------------------------------------------------

JS_EXTRACT_CHALLENGES = """(found_selector) => {
    const cards = Array.from(document.querySelectorAll(found_selector));
    return cards.map(card => {
        const titleEl = card.querySelector('.card-body h4');
        const title = titleEl ? titleEl.innerText.trim() : "";

        const header = card.querySelector('.card-header')?.innerText || "";
        if (!title) {
             if (header.includes("Progress") || header.includes("Filters")) return null;
             return null;
        }

        const category = card.querySelector('.card-title, span[class*="category"]')?.innerText?.trim() || "Unknown";
        const difficulty = card.querySelector('.pill')?.innerText?.trim() || "";

        const footerCols = Array.from(card.querySelectorAll('.card-footer .body-md'));
        let solvesText = "0";
        for (const col of footerCols) {
            if (col.innerText.toLowerCase().includes('solves')) {
                solvesText = col.innerText;
                break;
            }
        }
        const solves = parseInt(solvesText.replace(/,/g, '').replace(/[^0-9]/g, '')) || 0;

        const scoreDiv = card.querySelector('.score');
        let points = 0;
        if (scoreDiv) {
            let text = scoreDiv.innerText || "";
            if (difficulty) text = text.replace(difficulty, "");
            const matched = text.match(/(\\d+)/);
            if (matched) points = parseInt(matched[1]);
        }

        // Check if already solved (green check, "solved" class, etc.)
        const isSolved = !!(
            card.querySelector('.solved, .text-success, [class*="solved"]') ||
            card.classList.contains('solved') ||
            card.querySelector('svg[class*="check"]')
        );

        return {
            name: title,
            category: category,
            difficulty: difficulty,
            solves: solves,
            points: points,
            already_solved: isSolved
        };
    }).filter(c => c !== null && c.name !== "Progress Tracker" && c.name !== "Filters");
}"""

JS_FIND_CARD_BY_TITLE = """(args) => {
    const safe_name = args.safe_name;
    const titles = Array.from(document.querySelectorAll('.card-body h4'));
    const target = titles.find(el => el.innerText.trim() === safe_name);
    return target ? target.closest('.card') : null;
}"""

JS_EXTRACT_HINTS = """(modal) => {
    const hintDivs = Array.from(modal.querySelectorAll('.hint-content'));
    return hintDivs.map(d => d.innerText.trim());
}"""

JS_EXTRACT_DETAILS = """(modal) => {
    let descContainer = modal.querySelector('div[class*="description"]');

    if (!descContainer) {
        const cols = Array.from(modal.querySelectorAll('.col-md-8, .col-lg-8, .col-sm-12'));
        descContainer = cols.find(c => !c.innerText.includes("Hints"));
    }

    if (!descContainer) {
        descContainer = modal.querySelector('.modal-body');
    }

    const clone = descContainer.cloneNode(true);

    // Remove hints section
    const hintCols = clone.querySelectorAll('.col-md-4');
    hintCols.forEach(col => {
        if (col.innerText.includes("Hints")) col.remove();
    });

    const hintContent = clone.querySelectorAll('.hint-content');
    hintContent.forEach(el => el.remove());

    const headers = clone.querySelectorAll('h4');
    headers.forEach(h => {
        if (h.innerText.trim() === "Hints") h.remove();
    });

    const title = clone.querySelector('h3');
    if (title) title.remove();

    const desc = clone.innerText.trim();
    const html = clone.innerHTML;

    const codeBlocks = Array.from(modal.querySelectorAll('code, pre, input[readonly]'))
        .map(el => el.innerText || el.value)
        .filter(t => t && t.length > 3);

    const links = Array.from(modal.querySelectorAll('a[href]'));
    let files = links
        .filter(a => {
            const href = a.href.toLowerCase();
            if (href.includes('twitter') || href.includes('facebook') || href.includes('linkedin')) return false;
            if (href.includes('javascript:')) return false;
            if (href === '#' || href === '') return false;
            if (href.includes('challenge-files.picoctf.net')) return true;
            if (href.includes('artifacts.picoctf.net')) return true;
            if (href.includes('picoctf.net') && href.match(/:[0-9]+/)) return true;
            return true;
        })
        .map(a => ({
            text: a.innerText.trim() || "Download",
            url: a.href
        }));

    // Regex scan for non-hyperlinked URLs
    const urlRegex = /(https?:\\/\\/[^\\s]+)/g;
    const matches = desc.match(urlRegex);
    if (matches) {
        matches.forEach(url => {
            if (!files.some(f => f.url === url)) {
                files.push({ text: "Extracted URL", url: url });
            }
        });
    }

    return {
        description: desc,
        description_html: html,
        files: files,
        code_blocks: codeBlocks
    };
}"""

# Also try to detect the flag-submission input field
JS_EXTRACT_FLAG_INPUT = """(modal) => {
    // Look for flag-submission input
    const input = modal.querySelector('input[placeholder*="flag"], input[placeholder*="Flag"], input[name*="flag"], input[type="text"]');
    return input ? true : false;
}"""

# ---------------------------------------------------------------------------
# Card selectors to try (PicoCTF UI changes occasionally)
# ---------------------------------------------------------------------------
CARD_SELECTORS = [
    '.problem-list-card',
    'div[class*="ChallengeCard"]',
    'div[class*="challenge-card"]',
    'div[data-cy="challenge-card"]',
    '.card',
    'div[role="listitem"]',
]


def sanitize_name(name: str) -> str:
    """Sanitize a string for use as a directory/file name."""
    if not name:
        return "unknown"
    return re.sub(r'[^\w\-_]', '_', name)


def download_artifact(url: str, dest_path: str | Path) -> tuple[bool, str | None]:
    """Download a single file from *url* to *dest_path*.

    Returns (success, error_message).
    """
    try:
        if "wikipedia.org" in url:
            return False, "Skipped Wikipedia link"

        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"      ↓ Downloading {url} → {dest_path.name}")
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True, None

    except Exception as e:
        print(f"      ✗ Download failed: {e}")
        return False, str(e)


# ---------------------------------------------------------------------------
# High-level async helpers
# ---------------------------------------------------------------------------

async def login(page):
    """Navigate to PicoCTF login and wait for the user to complete CAPTCHA.

    The browser is launched visible (headless=False) so the user can interact
    with Cloudflare and the login form manually.
    """
    print("\n🔐  Navigating to PicoCTF login...")
    await page.goto("https://play.picoctf.org/login", wait_until="domcontentloaded")

    print("\n" + "=" * 60)
    print("  MANUAL ACTION REQUIRED")
    print("  1. Solve the Cloudflare CAPTCHA in the browser")
    print("  2. Enter your credentials and click Sign In")
    print("  3. Wait until you reach the Dashboard / Practice page")
    print("=" * 60)

    input("\n  Press ENTER here once you are fully logged in → ")
    print("  ✓ Continuing...\n")


async def iter_challenge_pages(page):
    """Async generator that yields one page of challenges at a time.

    The browser stays on each page until the caller advances via ``async for``,
    so challenges can be expanded while still in the DOM.

    Yields:
        (page_num, challenges)  - where challenges is a list[dict]
    """
    print("📋  Navigating to PicoGym practice page...")
    await page.goto("https://play.picoctf.org/practice", wait_until="networkidle")
    await page.wait_for_timeout(3000)

    seen_names: set[str] = set()
    page_num = 1
    max_pages = 50

    while page_num <= max_pages:
        print(f"   Page {page_num}...")

        # Find a working card selector
        found_selector = None
        for sel in CARD_SELECTORS:
            try:
                await page.wait_for_selector(sel, timeout=3000)
                found_selector = sel
                break
            except Exception:
                continue

        if not found_selector:
            print("   ⚠  Could not find any challenge cards - stopping.")
            return

        # Extract challenge metadata via JS
        challenges = await page.evaluate(JS_EXTRACT_CHALLENGES, found_selector)
        print(f"   Found {len(challenges)} challenges on page {page_num}")

        # De-duplicate
        new_challenges = [c for c in challenges if c["name"] not in seen_names]
        if not new_challenges:
            print("   ⚠  No new challenges - last page reached.")
            return

        for c in new_challenges:
            seen_names.add(c["name"])

        # Yield this page's challenges - browser stays on this page
        yield page_num, new_challenges

        # Partial page = last page
        if len(challenges) < 12:
            print("   ✓  Partial page - last page reached.")
            return

        # Advance to next page
        next_selectors = [
            'button[aria-label="Next page"]:not([disabled])',
            'li.page-item.active + li.page-item a',
            'button:has-text("Next")', 'a:has-text("Next")',
            'button:has-text("›")',
        ]

        next_btn = None
        for sel in next_selectors:
            try:
                btn = await page.query_selector(sel)
                if btn and not await btn.get_attribute("disabled"):
                    next_btn = btn
                    break
            except Exception:
                continue

        if next_btn:
            await next_btn.click()
            await page.wait_for_timeout(3000)
            page_num += 1
        else:
            return


async def get_challenge_details(page, challenge: dict, artifacts_dir: Path, launch_instance: bool = True) -> dict:
    """Click a challenge card to open its modal and extract full details.

    Downloads any linked artifacts into *artifacts_dir*/<category>/<name>/.
    Returns the enriched challenge dict with keys:
        description, description_html, files, hints, code_blocks, local_files
    """
    chal_name = challenge["name"]
    # Removed redundant console output - master.py now handles progress logging

    try:
        card_handle = await page.evaluate_handle(JS_FIND_CARD_BY_TITLE, {"safe_name": chal_name})
        card_el = card_handle.as_element()

        if not card_el:
            print(f"      ✗ Card element not found for '{chal_name}'")
            return {**challenge, "error": "card_not_found"}

        await card_el.click()

        # Wait for modal
        modal_selector = 'div[role="dialog"], div.modal, div[class*="popup"]'
        try:
            await page.wait_for_selector(modal_selector, timeout=5000)
        except Exception:
            print(f"      ⚠ Modal did not appear for '{chal_name}'")
            return {**challenge, "error": "modal_timeout"}

        modal = await page.query_selector(modal_selector)
        if not modal:
            return {**challenge, "error": "modal_not_found"}

        # Wait for content to load inside modal
        try:
            await page.wait_for_selector(
                'div[role="dialog"] .description, div[role="dialog"] p.body-sm, div[role="dialog"] button[title*="Hint"]',
                timeout=5000,
            )
        except Exception:
            pass  # Content may still be partially usable

        # --- Expand hints ---
        hints: list[str] = []
        try:
            hint_btns = await modal.query_selector_all('button[title*="Hint"], button[class*="hint"]')
            if hint_btns:
                for h_btn in hint_btns:
                    await h_btn.click()
                    await page.wait_for_timeout(200)
                hints = await page.evaluate(JS_EXTRACT_HINTS, modal)
        except Exception:
            pass

        # --- Launch Instance (for dynamic challenges) ---
        if launch_instance:
            try:
                launch_btn = await modal.query_selector('button:has-text("Launch Instance")')
                if launch_btn:
                    print("      ⏳ Launching instance (waiting 10s)...")
                    await launch_btn.click()
                    await page.wait_for_timeout(10000)
            except Exception:
                pass

        # --- Extract details ---
        details = await page.evaluate(JS_EXTRACT_DETAILS, modal)

        # --- Download artifacts ---
        local_files: list[dict] = []
        if details["files"]:
            chal_dir = artifacts_dir / sanitize_name(challenge.get("category", "misc")) / sanitize_name(chal_name)
            chal_dir.mkdir(parents=True, exist_ok=True)

            for file_obj in details["files"]:
                url = file_obj["url"]
                filename = url.split("/")[-1].split("?")[0]
                if not filename or len(filename) > 100:
                    filename = "downloaded_file"
                dest = chal_dir / filename
                success, error = download_artifact(url, dest)
                local_files.append({
                    "text": file_obj["text"],
                    "url": url,
                    "local_path": str(dest) if success else None,
                    "error": error,
                })

        # --- Check for flag input ---
        has_flag_input = await page.evaluate(JS_EXTRACT_FLAG_INPUT, modal)

        # --- Close modal ---
        await _close_modal(page, modal)

        enriched = {
            **challenge,
            "description": details["description"],
            "description_html": details["description_html"],
            "files": details["files"],
            "hints": hints,
            "code_blocks": details["code_blocks"],
            "local_files": local_files,
            "has_flag_input": has_flag_input,
        }

        n_files = len(local_files)
        n_hints = len(hints)
        print(f"      ✓ {n_files} file(s), {n_hints} hint(s)")
        return enriched

    except Exception as e:
        print(f"      ✗ Error expanding '{chal_name}': {e}")
        await page.keyboard.press("Escape")
        return {**challenge, "error": str(e)}


async def _close_modal(page, modal):
    """Best-effort close the currently open modal."""
    close_selectors = [
        'button[aria-label="Close"]',
        'button.close',
        'div[role="dialog"] button:first-child',
        'button:has-text("Close")',
        'button:has-text("✕")',
    ]
    for sel in close_selectors:
        try:
            btn = await modal.query_selector(sel)
            if btn:
                await btn.click()
                await page.wait_for_timeout(500)
                return
        except Exception:
            continue
    # Fallback
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(500)
