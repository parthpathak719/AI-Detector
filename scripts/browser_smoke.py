"""Optional: pip install playwright; playwright install chromium. Start server first."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright


async def main():
    out = Path("test-results")
    out.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 1100})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" and "422" not in message.text else None)
        await page.goto("http://127.0.0.1:8000")
        await page.get_by_text("Model ready", exact=True).wait_for()
        await page.evaluate("document.fonts.ready")
        await page.screenshot(path=str(out / "desktop.png"), full_page=True)
        await page.get_by_role("button", name="Try an example").click()
        passage = await page.locator("#text").input_value()
        await page.get_by_role("button", name="Analyze content").click()
        await page.locator("#result:not([hidden])").wait_for(timeout=60000)
        result = await page.locator("#prediction").inner_text()
        assert result in ("AI-leaning model signal", "Human-leaning model signal")
        await page.screenshot(path=str(out / "result.png"), full_page=True)
        await page.get_by_role("tab", name="Upload a file").click()
        await page.locator("#file").set_input_files({"name": "passage.txt", "mimeType": "text/plain", "buffer": passage.encode()})
        await page.get_by_role("button", name="Analyze content").click()
        await page.locator("#result:not([hidden])").wait_for(timeout=60000)
        assert await page.locator("#prediction").inner_text() == result
        long_text = (passage + "\n") * 40
        await page.evaluate("""text => {
            const transfer = new DataTransfer();
            transfer.items.add(new File([text], 'long-notes.md', {type: 'text/markdown'}));
            document.getElementById('drop-zone').dispatchEvent(new DragEvent('drop', {bubbles:true, cancelable:true, dataTransfer:transfer}));
        }""", long_text)
        assert 'long-notes.md' in await page.locator('#file-name').inner_text()
        await page.get_by_role("button", name="Analyze content").click()
        await page.locator("#result:not([hidden])").wait_for(timeout=120000)
        assert int(await page.locator("#result-segments").inner_text()) > 1
        assert await page.locator(".segment").count() == 0
        async with page.expect_download() as download_info:
            await page.get_by_role("button", name="Download PDF report").click()
        download = await download_info.value
        assert download.suggested_filename == "lexora-analysis.pdf"
        await download.save_as(out / 'lexora-analysis.pdf')
        assert (out / 'lexora-analysis.pdf').read_bytes().startswith(b'%PDF-')
        assert await page.locator('#edit-source').count() == 0
        assert await page.locator('#extracted-text').count() == 0
        assert await page.locator('#segment-list').count() == 0
        assert 'research' not in (await page.locator('body').inner_text()).lower()
        assert await page.get_by_text('JSONL format', exact=True).count() == 0
        from io import BytesIO
        from pypdf import PdfReader
        report_text = '\n'.join(p.extract_text() for p in PdfReader(BytesIO((out / 'lexora-analysis.pdf').read_bytes())).pages)
        assert 'Segment 01' in report_text and 'Segment 09' in report_text
        await page.screenshot(path=str(out / "document-result.png"), full_page=True)
        import json
        jsonl = json.dumps({'text': passage, 'label': 'ai'}).encode()
        await page.locator('#file').set_input_files({'name': 'data.jsonl', 'mimeType': 'application/x-ndjson', 'buffer': jsonl})
        await page.get_by_role('button', name='Analyze content').click()
        await page.locator('#result:not([hidden])').wait_for(timeout=60000)
        assert await page.locator('#prediction').inner_text() == result
        await page.locator('.limitations summary').click()
        assert '1 JSONL record(s)' in await page.locator('#warnings').inner_text()
        # Each JSONL record and each file gets a separate result, after preview.
        batch_jsonl = '\n'.join(json.dumps({'id': name, 'text': value}) for name,value in [('essay',passage),('short','too short')])
        await page.locator('#file').set_input_files([
            {'name':'first.txt','mimeType':'text/plain','buffer':passage.encode()},
            {'name':'records.jsonl','mimeType':'application/x-ndjson','buffer':batch_jsonl.encode()},
        ])
        await page.locator('#submit:not([disabled])').wait_for()
        await page.locator('#jsonl-mode').select_option('independent')
        await page.locator('#submit:not([disabled])').wait_for()
        assert await page.locator('#preview-item option').count()==3
        await page.locator('#preview-item').select_option('1')
        assert await page.locator('#preview-text').input_value()==passage
        assert await page.locator('#preview-language').input_value()=='en'
        assert await page.locator('#preview-text').get_attribute('readonly') is not None
        assert await page.locator('#result').is_hidden()
        await page.get_by_role('button',name='Analyze content').click()
        await page.locator('#submit:not([disabled])').wait_for(timeout=60000)
        assert '2 complete · 1 failed' in await page.locator('#batch-status').inner_text()
        assert await page.locator('.batch-row button:visible').count()==2
        await page.locator('.batch-row button').nth(0).click()
        assert await page.locator('#result-source').inner_text()=='first.txt'
        await page.screenshot(path=str(out/'batch-result.png'),full_page=True)
        # Stop only queued work, leaving the current completed result usable.
        hit=asyncio.Event(); release=asyncio.Event()
        async def hold_first(route):
            hit.set()
            await release.wait()
            await route.continue_()
        await page.route('**/api/analyze',hold_first)
        await page.get_by_role('button',name='Analyze content').click()
        await asyncio.wait_for(hit.wait(),10)
        await page.get_by_role('button',name='Stop after current item',exact=True).click()
        release.set()
        await page.locator('#submit:not([disabled])').wait_for(timeout=60000)
        assert '1 complete · 0 failed · stopped' in await page.locator('#batch-status').inner_text()
        assert await page.get_by_text('Not analyzed',exact=True).count()==2
        await page.unroute('**/api/analyze',hold_first)
        await page.locator('#file').set_input_files({'name': 'bad.jsonl', 'mimeType': 'application/x-ndjson', 'buffer': b'{broken'})
        await page.locator('#error:not([hidden])').wait_for()
        assert 'line 1' in await page.locator('#error').inner_text()
        assert "16 handcrafted" not in await page.locator("body").inner_text()
        await page.get_by_role("tab", name="Paste text").click()
        await page.locator("#text").fill("too short")
        await page.get_by_role("button", name="Analyze content").click()
        await page.locator("#error:not([hidden])").wait_for()
        assert await page.locator("#result").is_hidden()
        await page.set_viewport_size({"width": 390, "height": 844})
        await page.get_by_role("button", name="Clear", exact=True).click()
        await page.screenshot(path=str(out / "mobile.png"), full_page=True)
        assert not await page.evaluate("document.documentElement.scrollWidth > innerWidth")
        small = await page.evaluate("""() => [...document.querySelectorAll('p,button,small,label,summary,select,textarea')]
          .filter(e => e.getClientRects().length && parseFloat(getComputedStyle(e).fontSize) < 14)
          .map(e => e.id || e.className)""")
        assert not small, small
        assert not errors, errors
        print("Browser checks passed: real inference, upload parity, validation error, stale-result clearing, long-document segmentation, report export, mobile overflow, no JavaScript errors.")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
