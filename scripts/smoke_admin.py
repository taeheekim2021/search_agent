"""Browser smoke test against an explicitly started sample/mock admin server.

Read SEARCH_ADMIN_API_KEY from the environment. The real-server scenario blocks
index writes and only previews ingestion. A separate UI-only scenario intercepts
every API response to test write confirmations; it never contacts OpenSearch or
executes models. Neither scenario validates actual Qwen/BGE inference.
"""

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright

MALICIOUS_TITLE = '검증용 <img src="admin-xss-fixture" onerror="window.__admin_xss=1">'


def preview_manifest():
    return {
        "version": 1,
        "entries": [
            {
                "source_id": "admin-browser-fixture",
                "canonical_url": "https://example.org/admin-browser-fixture",
                "original_url": "https://example.org/admin-browser-fixture/watch",
                "title": MALICIOUS_TITLE,
                "description": "Browser rendering fixture; metadata preview only.",
                "tags": ["공룡"],
                "author": "Test fixture",
                "license": "CC0-1.0",
                "license_url": "https://example.org/license",
                "attribution": "Synthetic browser fixture",
                "rights_verified": True,
                "metadata_provenance": {
                    field: {"method": "editorial", "note": "Synthetic browser fixture"}
                    for field in ["title", "description", "tags"]
                },
            }
        ],
    }


def capture_screenshot(page, directory, name):
    if directory:
        assert not page.locator("#admin-key").input_value(), (
            "Screenshot rejected while a credential input contains text."
        )
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=directory / name, full_page=True, animations="disabled")


def check_mobile_overflow(page, tab, directory):
    if page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"):
        return
    capture_screenshot(page, directory, f"failure-mobile-{tab}.png")
    boxes = page.evaluate("""() => [...document.querySelectorAll('body *')].flatMap(element => {
        const box = element.getBoundingClientRect();
        if (!box.width || !box.height || (box.left >= 0 && box.right <= innerWidth)) return [];
        return [{tag: element.tagName.toLowerCase(), id: element.id,
            class: element.getAttribute('class') || '',
            left: Math.round(box.left), right: Math.round(box.right)}];
    }).slice(0, 40)""")
    raise AssertionError(f"Mobile overflow in {tab}: {json.dumps(boxes, ensure_ascii=False)}")


def synthetic_write_ui(browser, base_url, sample_status, screenshots_dir):
    """UI-only fixture: intercept every API request; never contact OpenSearch or run models."""
    status = json.loads(json.dumps(sample_status))
    status.update(backend="opensearch", mode="real", sample=False, content_count=0)
    status["notice"] = "Synthetic browser responses only. No OpenSearch or model is running."
    status["capabilities"] = dict.fromkeys(status["capabilities"], True)
    status["index"].update(name="synthetic-ui-only", state="available", exists=True, health="green")
    for model in status["models"].values():
        model.update(state="not_loaded", loaded=False, inference_readiness="not_checked")
    writes, unexpected, errors = [], [], []
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, service_workers="block")

    def intercept(route):
        request, path = route.request, urlsplit(route.request.url).path
        payload = request.post_data_json if request.method == "POST" else {}
        if not isinstance(payload, dict):
            unexpected.append((request.method, path))
            route.abort()
            return
        if request.method == "GET" and path == "/api/admin/status":
            response = status
        elif request.method == "GET" and path == "/api/admin/contents":
            response = {"items": [], "total": 0, "offset": 0, "limit": 20, "has_more": False}
        elif request.method == "POST" and path == "/api/admin/ingest":
            entries = [
                dict(item, content_id=f"synthetic-ui-{i}")
                for i, item in enumerate(payload["manifest"]["entries"])
            ]
            if payload.get("dry_run") is True:
                response = {"dry_run": True, "entries": entries, "status": "planned"}
            elif payload.get("dry_run") is False and payload.get("confirmed") is True:
                writes.append((path, payload))
                entries[0].update(status="indexed", stage="complete")
                entries[1].update(
                    status="failed", stage="index_upsert", error_type="OpenSearchError"
                )
                response = {
                    "dry_run": False,
                    "entries": entries,
                    "indexed": 1,
                    "status": "partial_failure",
                    "media_download_performed": False,
                }
            else:
                unexpected.append((request.method, path))
                route.abort()
                return
        elif (
            request.method == "POST"
            and path in {"/api/admin/index/ensure", "/api/admin/index/refresh"}
            and payload == {"confirmed": True}
        ):
            writes.append((path, payload))
            response = {
                "index": "synthetic-ui-only",
                "status": "ready" if path.endswith("ensure") else "refreshed",
            }
        else:
            unexpected.append((request.method, path))
            route.abort()
            return
        route.fulfill(status=200, json=response)

    # No API request in this context is ever continued to the actual server.
    context.route("**/api/**", intercept)
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(base_url + "/admin")
    page.locator("#admin-key").fill("synthetic-browser-fixture-key-not-a-server-key")
    page.locator("#connect-button").click()
    expect(page.locator("#workspace")).to_be_visible()
    page.locator("#tab-contents").click()
    expect(page.locator("#content-empty")).to_be_visible()
    manifest = preview_manifest()
    manifest["entries"].append(
        dict(
            manifest["entries"][0],
            title="Second synthetic item",
            canonical_url="https://example.org/second-synthetic-ui-item",
        )
    )
    page.locator("#manifest-json").fill(json.dumps(manifest, ensure_ascii=False))
    page.locator("#preview-ingest").click()
    expect(page.locator("#execute-ingest")).to_be_enabled()
    manifest["entries"][0]["title"] = "Edited after preview: synthetic fixture"
    page.locator("#manifest-json").fill(json.dumps(manifest, ensure_ascii=False))
    expect(page.locator("#execute-ingest")).to_be_disabled()
    expect(page.locator("#ingest-result")).to_be_hidden()
    assert not writes
    page.locator("#preview-ingest").click()
    expect(page.locator("#execute-ingest")).to_be_enabled()

    def cancel_then_confirm(button_id, endpoint):
        before = len(writes)
        page.locator(f"#{button_id}").click()
        expect(page.locator("#confirm-dialog")).to_be_visible()
        assert len(writes) == before
        page.locator("#cancel-confirm").click()
        expect(page.locator("#confirm-dialog")).to_be_hidden()
        assert len(writes) == before
        page.locator(f"#{button_id}").click()
        with page.expect_response(lambda response: urlsplit(response.url).path == endpoint):
            page.locator("#accept-confirm").click()
        assert len(writes) == before + 1 and writes[-1][0] == endpoint

    cancel_then_confirm("execute-ingest", "/api/admin/ingest")
    expect(page.locator("#ingest-feedback")).to_contain_text("일부 항목이 실패")
    expect(page.locator("#ingest-result")).to_contain_text("적재 1개 / 전체 2개")
    expect(page.locator("#ingest-result")).to_contain_text("index_upsert")
    expect(page.locator("#execute-ingest")).to_be_disabled()
    assert writes[0][1]["manifest"] == manifest
    capture_screenshot(page, screenshots_dir, "08-synthetic-partial-failure-ui.png")
    page.locator("#tab-index").click()
    for button, endpoint in [
        ("ensure-index", "/api/admin/index/ensure"),
        ("refresh-index", "/api/admin/index/refresh"),
    ]:
        expect(page.locator(f"#{button}")).to_be_enabled()
        cancel_then_confirm(button, endpoint)
        expect(page.locator("#index-feedback")).to_contain_text("synthetic-ui-only")
        expected_status = "ready" if endpoint.endswith("ensure") else "refreshed"
        expect(page.locator("#index-operation-raw")).to_contain_text(
            f'"status": "{expected_status}"'
        )
        expect(page.locator(f"#{button}")).to_be_enabled()
    assert len(writes) == 3 and not unexpected and not errors
    context.close()
    return {
        "scenario": "synthetic_browser_responses_only",
        "confirmed_requests": len(writes),
        "cancelled_actions": 3,
        "preview_invalidation": True,
        "partial_failure_rendered": True,
        "network_api_writes": 0,
        "actual_opensearch_validated": False,
        "actual_model_inference_validated": False,
        "javascript_errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        help="Optional directory for screenshots without credentials.",
    )
    args = parser.parse_args()
    key = os.environ.get("SEARCH_ADMIN_API_KEY", "")
    if len(key) < 32:
        parser.error("Set SEARCH_ADMIN_API_KEY to the sample/mock server's administrator key.")
    base_url = args.base_url.rstrip("/")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or None
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors, forbidden_writes, requests = [], [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        if args.screenshots_dir:
            args.screenshots_dir.mkdir(parents=True, exist_ok=True)

        def capture(name):
            capture_screenshot(page, args.screenshots_dir, name)

        def api_route(route):
            request = route.request
            path = urlsplit(request.url).path
            payload = request.post_data_json if request.method == "POST" else None
            requests.append((request.method, path))
            if request.method not in {"GET", "HEAD"}:
                preview = (
                    path == "/api/admin/ingest"
                    and isinstance(payload, dict)
                    and payload.get("dry_run", True) is True
                )
                search = path == "/api/admin/search"
                if not (preview or search):
                    forbidden_writes.append(path)
                    route.abort()
                    return
            route.continue_()

        page.route("**/api/admin/**", api_route)
        initial = page.goto(base_url + "/admin")
        assert initial and initial.status == 200
        assert initial.headers["cache-control"] == "no-store"
        assert "'unsafe-inline'" not in initial.headers.get("content-security-policy", "")
        expect(page.locator("#connection-gate")).to_be_visible()
        expect(page.locator("#workspace")).to_be_hidden()
        capture("01-login.png")

        def response_from(path):
            return page.expect_response(lambda response: urlsplit(response.url).path == path)

        # Failed authentication leaves the data workspace inaccessible.
        page.locator("#admin-key").fill("incorrect-browser-fixture-key-0000000000")
        with response_from("/api/admin/status") as unauthorized:
            page.locator("#connect-button").click()
        assert unauthorized.value.status == 401
        expect(page.locator("#connection-feedback")).not_to_be_empty()
        expect(page.locator("#workspace")).to_be_hidden()

        def connect():
            page.locator("#admin-key").fill(key)
            with response_from("/api/admin/status") as status_response:
                page.locator("#connect-button").click()
            assert status_response.value.status == 200
            status = status_response.value.json()
            assert status["backend"] == "sample" and status["mode"] == "mock", (
                "This smoke test requires SEARCH_BACKEND=sample and SEARCH_MODE=mock."
            )
            expect(page.locator("#workspace")).to_be_visible()
            expect(page.locator("#admin-key")).to_have_value("")
            expect(page.locator("#stat-content")).to_contain_text(str(status["content_count"]))
            return status

        status = connect()
        capture("02-overview-desktop.png")
        with response_from("/api/admin/contents") as contents_response:
            page.locator("#tab-contents").click()
        assert contents_response.value.status == 200
        contents = contents_response.value.json()
        expect(page.locator("#content-rows tr")).to_have_count(len(contents["items"]))
        capture("03-contents-desktop.png")
        page.locator("#content-rows button").first.click()
        expect(page.locator("#content-dialog")).to_be_visible()
        expect(page.locator("#content-detail-title")).to_have_text(contents["items"][0]["title"])
        capture("04-content-detail-desktop.png")
        page.locator("#close-content-dialog").click()

        page.locator("#content-query").fill("SAMPLE-001")
        with response_from("/api/admin/contents") as filtered_response:
            page.locator("#content-search-button").click()
        assert filtered_response.value.json()["total"] == 1
        expect(page.locator("#content-rows tr")).to_have_count(1)

        page.locator("#tab-search").click()
        page.locator("#search-query").fill("5살 아이가 볼 공룡 영상")
        page.locator("#search-top-k").fill("5")
        page.locator("#search-top-n").fill("3")
        with response_from("/api/admin/search") as search_response:
            page.locator("#run-search").click()
        assert search_response.value.status == 200
        result = search_response.value.json()
        assert result["mode"] == "mock" and not result["model_inference_performed"]
        assert len(result["results"]) == 3
        expect(page.locator("#search-results article")).to_have_count(3)
        capture("05-search-desktop.png")

        page.locator("#tab-contents").click()
        page.locator("#manifest-json").fill(json.dumps(preview_manifest(), ensure_ascii=False))
        with response_from("/api/admin/ingest") as preview_response:
            page.locator("#preview-ingest").click()
        assert preview_response.value.status == 200
        preview = preview_response.value.json()
        assert preview["dry_run"] and preview["network_accessed"] is False
        assert preview["media_download_performed"] is False
        expect(page.locator("#ingest-result")).to_be_visible()
        expect(page.locator("#ingest-raw")).to_contain_text("admin-xss-fixture")
        expect(page.locator('img[src="admin-xss-fixture"]')).to_have_count(0)
        assert page.evaluate("window.__admin_xss || null") is None
        expect(page.locator("#execute-ingest")).to_be_disabled()
        capture("06-preview-desktop.png")
        page.locator("#tab-index").click()
        expect(page.locator("#ensure-index")).to_be_disabled()
        expect(page.locator("#refresh-index")).to_be_disabled()

        page.set_viewport_size({"width": 390, "height": 844})
        for tab in ["overview", "contents", "search", "index"]:
            page.locator(f"#tab-{tab}").click()
            check_mobile_overflow(page, tab, args.screenshots_dir)
            if tab == "overview":
                capture("07-overview-mobile.png")
        assert key not in page.content()
        assert (
            page.evaluate("Object.keys(localStorage).length + Object.keys(sessionStorage).length")
            == 0
        )

        # A reload clears the memory-only credential; disconnect also clears the workspace.
        page.reload()
        expect(page.locator("#connection-gate")).to_be_visible()
        expect(page.locator("#workspace")).to_be_hidden()
        expect(page.locator("#admin-key")).to_have_value("")
        connect()
        page.locator("#disconnect").click()
        expect(page.locator("#connection-gate")).to_be_visible()
        expect(page.locator("#workspace")).to_be_hidden()
        assert not errors, errors
        assert not forbidden_writes, forbidden_writes
        print(
            json.dumps(
                {
                    "scenario": "sample_server_http",
                    "backend": status["backend"],
                    "mode": status["mode"],
                    "content_count": status["content_count"],
                    "search_results": len(result["results"]),
                    "authenticated_browse_and_search": True,
                    "metadata_preview": True,
                    "actual_model_inference_validated": False,
                    "index_mutations": 0,
                    "credential_persisted": False,
                    "xss_rendered_as_text": True,
                    "mobile_overflow": False,
                    "javascript_errors": errors,
                    "admin_requests": len(requests),
                },
                ensure_ascii=False,
            )
        )
        print(json.dumps(synthetic_write_ui(browser, base_url, status, args.screenshots_dir)))
        browser.close()


if __name__ == "__main__":
    main()
