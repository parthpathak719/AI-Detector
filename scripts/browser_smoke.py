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
        assert await page.locator(".segment").count() > 1
        async with page.expect_download() as download_info:
            await page.get_by_role("button", name="Export report").click()
        download = await download_info.value
        assert download.suggested_filename == "origin-analysis.json"
        await page.screenshot(path=str(out / "document-result.png"), full_page=True)
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
        assert not errors, errors
        print("Browser checks passed: real inference, upload parity, validation error, stale-result clearing, long-document segmentation, report export, mobile overflow, no JavaScript errors.")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
