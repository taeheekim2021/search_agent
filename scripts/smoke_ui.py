"""Browser smoke test against an explicitly started mock server on localhost:8000."""

import argparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.base_url)
        expect(page.locator("#mode")).to_contain_text("모의 모드")
        results = []
        for query, count in [
            ("5살 아이가 볼 공룡 영상", 3),
            ("토리 캐릭터 영상", 4),
            ("18살 공룡 영상", 0),
            ("5살 아이가 볼 공룡 영상", 3),
        ]:
            page.locator("#query").fill(query)
            page.locator("#submit").click()
            expect(page.locator("#status")).to_contain_text(f"{count}개 결과")
            expect(page.locator("article")).to_have_count(count)
            results.append((query, page.locator("article h2").all_text_contents()))
        assert results[0][1] == results[-1][1]
        page.locator("#k").fill("1")
        page.locator("#n").fill("5")
        page.locator("#submit").click()
        expect(page.locator("#status")).to_contain_text("검색 실패")
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert not errors, errors
        print(
            {
                "mode": "mock",
                "browser": "Chromium",
                "checks": results,
                "repeat_stable": True,
                "validation_error": True,
                "mobile_overflow": False,
                "javascript_errors": errors,
            }
        )
        browser.close()


if __name__ == "__main__":
    main()
