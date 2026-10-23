#!/usr/bin/env python3
"""
Flag Submitter - types a flag into the PicoCTF challenge modal and reads the result.
"""


async def submit_flag(page, challenge_name: str, flag: str, page_num: int = None) -> dict:
    """Open the challenge modal, type the flag, click Submit, and read the result.

    Returns a dict: {"submitted": bool, "accepted": bool, "message": str}
    """
    from solver.scraper import JS_FIND_CARD_BY_TITLE

    print(f"      📤  Submitting flag for '{challenge_name}'...")

    try:
        # Navigate to the correct page if provided
        if page_num:
            current_url = page.url
            target_url = f"https://play.picoctf.org/practice?page={page_num}"
            if target_url not in current_url:
                await page.goto(target_url, wait_until="networkidle")
                await page.wait_for_timeout(2000)

        # Search through pagination pages to find the challenge card
        card_el = None
        for page_attempt in range(5):  # up to 5 pages
            card_handle = await page.evaluate_handle(JS_FIND_CARD_BY_TITLE, {"safe_name": challenge_name})
            card_el = card_handle.as_element()
            if card_el:
                break
            # Try clicking next page
            next_btn = await page.query_selector('button[aria-label="Next"], a:has-text("Next"), li.page-item:last-child a')
            if next_btn:
                try:
                    await next_btn.click()
                    await page.wait_for_timeout(2000)
                except Exception:
                    break
            else:
                break

        if not card_el:
            return {"submitted": False, "accepted": False, "message": "card_not_found"}

        await card_el.click()

        modal_selector = 'div[role="dialog"], div.modal, div[class*="popup"]'
        try:
            await page.wait_for_selector(modal_selector, timeout=5000)
        except Exception:
            return {"submitted": False, "accepted": False, "message": "modal_timeout"}

        modal = await page.query_selector(modal_selector)
        if not modal:
            return {"submitted": False, "accepted": False, "message": "modal_not_found"}

        await page.wait_for_timeout(1000)

        # Find the flag input field
        input_selectors = [
            'input[placeholder*="flag" i]',
            'input[placeholder*="Flag" i]',
            'input[name*="flag" i]',
            'input[type="text"]',
        ]

        flag_input = None
        for sel in input_selectors:
            flag_input = await modal.query_selector(sel)
            if flag_input:
                break

        if not flag_input:
            await _close(page, modal)
            return {"submitted": False, "accepted": False, "message": "no_flag_input"}

        # Clear any existing text and type the flag
        await flag_input.click()
        await flag_input.fill("")
        await flag_input.type(flag, delay=30)

        # Find and click the submit button
        submit_selectors = [
            'button:has-text("Submit")',
            'button:has-text("Check")',
            'button[type="submit"]',
            'input[type="submit"]',
        ]

        submit_btn = None
        for sel in submit_selectors:
            submit_btn = await modal.query_selector(sel)
            if submit_btn:
                break

        if not submit_btn:
            await _close(page, modal)
            return {"submitted": False, "accepted": False, "message": "no_submit_button"}

        # Snapshot modal text BEFORE submitting, so we can diff for new feedback
        text_before = (await modal.inner_text()).lower()

        await submit_btn.click()

        # Wait for the result feedback
        await page.wait_for_timeout(3000)

        accepted = False
        rejected = False

        _reject_phrases = [
            "incorrect", "wrong", "try again", "invalid", "not correct",
            "sorry",
        ]
        _accept_phrases = [
            "that is the correct flag", "congrat", "nice work",
            "well done", "flag accepted", "you got it",
            "that's the flag", "you solved", "hurray",
            "solved this challenge",
        ]

        # --- Step 1: Check toasts/notifications FIRST (most reliable) ---
        toast_selectors = [
            '.toast-body', '.toast', '[class*="notification"]',
            '[class*="feedback"]', '[class*="alert-"]:not([class*="alert-info"])',
        ]
        for sel in toast_selectors:
            try:
                els = await page.query_selector_all(sel)
                for el in els:
                    toast_text = (await el.inner_text()).lower().strip()
                    if not toast_text:
                        continue
                    print(f"      📋  Toast/alert text: {toast_text[:200]}")
                    if any(w in toast_text for w in _reject_phrases):
                        rejected = True
                        break
                    if any(w in toast_text for w in _accept_phrases):
                        accepted = True
                        break
                if accepted or rejected:
                    break
            except Exception:
                continue

        # --- Step 2: Check modal text diff ---
        if not accepted and not rejected:
            text_after = (await modal.inner_text()).lower()

            # Isolate new feedback text
            if len(text_after) > len(text_before):
                new_text = text_after[len(text_before):]
            elif text_after != text_before:
                new_text = ""
                for i, (a, b) in enumerate(zip(text_before, text_after)):
                    if a != b:
                        new_text = text_after[i:]
                        break
            else:
                new_text = ""

            if new_text:
                print(f"      📋  Feedback text: {new_text[:200]}")
                # Check rejected FIRST - "incorrect" contains "correct"
                if any(phrase in new_text for phrase in _reject_phrases):
                    rejected = True
                elif any(phrase in new_text for phrase in _accept_phrases):
                    accepted = True

        # --- Step 3: Full page text scan as last resort ---
        if not accepted and not rejected:
            try:
                full_page_text = (await page.inner_text('body')).lower()
                if any(w in full_page_text for w in _reject_phrases):
                    rejected = True
                elif any(w in full_page_text for w in _accept_phrases):
                    accepted = True
            except Exception:
                pass

        if accepted:
            print(f"      ✅  Flag accepted!")
            message = "accepted"
        elif rejected:
            print(f"      ❌  Flag rejected")
            message = "rejected"
        else:
            print(f"      ❓  Unclear result - check manually")
            message = "unclear"

        await _close(page, modal)

        return {
            "submitted": True,
            "accepted": accepted,
            "message": message,
        }

    except Exception as e:
        print(f"      ✗ Submit error: {e}")
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass
        return {"submitted": False, "accepted": False, "message": str(e)}


async def _close(page, modal):
    """Best-effort close modal."""
    close_selectors = [
        'button[aria-label="Close"]',
        'button.close',
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
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(500)
