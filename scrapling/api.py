"""
REST API server for Scrapling's fetcher classes.

Exposes Fetcher, DynamicFetcher, and StealthyFetcher via FastAPI endpoints.
Run with: scrapling api --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import os
import secrets
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional

try:
    from fastapi import Depends, FastAPI, HTTPException, Security
    from fastapi.responses import JSONResponse
    from fastapi.security import APIKeyHeader
    from pydantic import BaseModel, Field
except (ImportError, ModuleNotFoundError) as e:
    raise ModuleNotFoundError(
        'You need to install scrapling with the "api" extra to enable the REST API server. '
        "Install it with: pip install scrapling[api]"
    ) from e


# ---------------------------------------------------------------------------
# API Key authentication
# ---------------------------------------------------------------------------

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_api_key() -> Optional[str]:
    """Read the API key from SCRAPLING_API_KEY env var. Returns None if not set (auth disabled)."""
    return os.environ.get("SCRAPLING_API_KEY")


def _verify_api_key(api_key: Optional[str] = Security(_API_KEY_HEADER)) -> None:
    """Dependency that validates the X-API-Key header against SCRAPLING_API_KEY env var.

    If SCRAPLING_API_KEY is not set, authentication is disabled and all requests pass through.
    """
    expected = _get_api_key()
    if expected is None:
        # Auth disabled — no key configured
        return
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key. Provide it via the X-API-Key header.")
    if not secrets.compare_digest(api_key, expected):
        raise HTTPException(status_code=403, detail="Invalid API key.")


# ---------------------------------------------------------------------------
# Browser concurrency control
# ---------------------------------------------------------------------------

logger = logging.getLogger("scrapling.api")

_MAX_BROWSERS = int(os.environ.get("SCRAPLING_MAX_BROWSERS", "1"))
_MAX_FETCHERS = int(os.environ.get("SCRAPLING_MAX_FETCHERS", "10"))
_QUEUE_TIMEOUT = int(os.environ.get("SCRAPLING_BROWSER_QUEUE_TIMEOUT", "60"))
# Bounded thread pool prevents OS thread exhaustion — all blocking fetcher/browser
# work runs through this pool instead of FastAPI spawning unbounded threads.
_worker_pool = ThreadPoolExecutor(max_workers=_MAX_BROWSERS + _MAX_FETCHERS)
_browser_semaphore = asyncio.Semaphore(_MAX_BROWSERS)
_fetcher_semaphore = asyncio.Semaphore(_MAX_FETCHERS)


async def _acquire_semaphore(sem: asyncio.Semaphore, timeout: int) -> bool:
    """Try to acquire an asyncio semaphore with a timeout. Returns True on success."""
    try:
        await asyncio.wait_for(sem.acquire(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class FetcherGetRequest(BaseModel):
    url: str
    headers: Optional[Dict[str, str]] = None
    cookies: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, str]] = None
    proxy: Optional[str] = None
    timeout: Optional[int] = 30
    follow_redirects: Optional[bool] = True
    verify: Optional[bool] = True
    impersonate: Optional[str] = None
    stealthy_headers: Optional[bool] = True
    css_selector: Optional[str] = None
    xpath_selector: Optional[str] = None


class FetcherDataRequest(FetcherGetRequest):
    data: Optional[Dict[str, str]] = None
    json_data: Optional[Any] = Field(None, alias="json_data")


class DynamicFetchRequest(BaseModel):
    url: str
    headless: Optional[bool] = True
    disable_resources: Optional[bool] = False
    network_idle: Optional[bool] = False
    load_dom: Optional[bool] = True
    timeout: Optional[int] = 30000
    wait: Optional[int] = None
    wait_selector: Optional[str] = None
    wait_selector_state: Optional[str] = "attached"
    locale: Optional[str] = None
    real_chrome: Optional[bool] = False
    cdp_url: Optional[str] = None
    proxy: Optional[str] = None
    extra_headers: Optional[Dict[str, str]] = None
    useragent: Optional[str] = None
    google_search: Optional[bool] = True
    css_selector: Optional[str] = None
    xpath_selector: Optional[str] = None


class StealthyFetchRequest(DynamicFetchRequest):
    allow_webgl: Optional[bool] = True
    hide_canvas: Optional[bool] = False
    block_webrtc: Optional[bool] = False
    solve_cloudflare: Optional[bool] = False


class ScrapeResponse(BaseModel):
    url: str
    status: int
    reason: str
    headers: Dict[str, str] = {}
    cookies: Any = {}
    body: str = ""
    selected_content: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _response_to_dict(response: Any, css_selector: Optional[str] = None, xpath_selector: Optional[str] = None) -> dict:
    """Convert a Scrapling Response object to a JSON-serializable dict."""
    # Extract headers as plain dict
    resp_headers = {}
    if response.headers:
        if isinstance(response.headers, dict):
            resp_headers = {str(k): str(v) for k, v in response.headers.items()}
        else:
            try:
                resp_headers = {str(k): str(v) for k, v in response.headers.items()}
            except Exception:
                resp_headers = {}

    # Extract cookies as plain dict
    resp_cookies = {}
    if response.cookies:
        if isinstance(response.cookies, dict):
            resp_cookies = response.cookies
        elif isinstance(response.cookies, (list, tuple)):
            for cookie in response.cookies:
                if isinstance(cookie, dict) and "name" in cookie and "value" in cookie:
                    resp_cookies[cookie["name"]] = cookie["value"]
                elif isinstance(cookie, dict):
                    resp_cookies.update(cookie)

    # Body as string
    body = ""
    try:
        body = (
            response.body.decode("utf-8", errors="replace") if isinstance(response.body, bytes) else str(response.body)
        )
    except Exception:
        body = str(response)

    # Selected content via CSS or XPath
    selected = None
    if css_selector:
        try:
            elements = response.css(css_selector)
            selected = [el.html if hasattr(el, "html") else str(el) for el in elements]
        except Exception as e:
            selected = [f"CSS selector error: {e}"]
    elif xpath_selector:
        try:
            elements = response.xpath(xpath_selector)
            selected = [el.html if hasattr(el, "html") else str(el) for el in elements]
        except Exception as e:
            selected = [f"XPath selector error: {e}"]

    return {
        "url": str(response.url),
        "status": response.status,
        "reason": response.reason,
        "headers": resp_headers,
        "cookies": resp_cookies,
        "body": body,
        "selected_content": selected,
    }


def _build_fetcher_kwargs(req: FetcherGetRequest) -> dict:
    """Build kwargs dict for Fetcher from request model, excluding None values."""
    kwargs: Dict[str, Any] = {}
    if req.headers:
        kwargs["headers"] = req.headers
    if req.cookies:
        kwargs["cookies"] = req.cookies
    if req.params:
        kwargs["params"] = req.params
    if req.proxy:
        kwargs["proxy"] = req.proxy
    if req.timeout is not None:
        kwargs["timeout"] = req.timeout
    if req.follow_redirects is not None:
        kwargs["follow_redirects"] = req.follow_redirects
    if req.verify is not None:
        kwargs["verify"] = req.verify
    if req.impersonate:
        kwargs["impersonate"] = req.impersonate
    if req.stealthy_headers is not None:
        kwargs["stealthy_headers"] = req.stealthy_headers
    return kwargs


def _build_dynamic_kwargs(req: DynamicFetchRequest) -> dict:
    """Build kwargs dict for DynamicFetcher from request model."""
    kwargs: Dict[str, Any] = {
        "headless": req.headless,
        "disable_resources": req.disable_resources,
        "network_idle": req.network_idle,
        "load_dom": req.load_dom,
        "timeout": req.timeout,
        "real_chrome": req.real_chrome,
        "google_search": req.google_search,
    }
    if req.wait is not None:
        kwargs["wait"] = req.wait
    if req.wait_selector:
        kwargs["wait_selector"] = req.wait_selector
    if req.wait_selector_state:
        kwargs["wait_selector_state"] = req.wait_selector_state
    if req.locale:
        kwargs["locale"] = req.locale
    if req.cdp_url:
        kwargs["cdp_url"] = req.cdp_url
    if req.proxy:
        kwargs["proxy"] = req.proxy
    if req.extra_headers:
        kwargs["extra_headers"] = req.extra_headers
    if req.useragent:
        kwargs["useragent"] = req.useragent
    return kwargs


def _build_stealthy_kwargs(req: StealthyFetchRequest) -> dict:
    """Build kwargs dict for StealthyFetcher from request model."""
    kwargs = _build_dynamic_kwargs(req)
    kwargs["allow_webgl"] = req.allow_webgl
    kwargs["hide_canvas"] = req.hide_canvas
    kwargs["block_webrtc"] = req.block_webrtc
    kwargs["solve_cloudflare"] = req.solve_cloudflare
    return kwargs


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    app = FastAPI(
        title="Scrapling REST API",
        description="REST API for Scrapling web scraping library. "
        "Exposes Fetcher (HTTP), DynamicFetcher (browser), and StealthyFetcher (stealth browser) endpoints.\n\n"
        "**Authentication:** Set the `SCRAPLING_API_KEY` env var to enable API key auth. "
        "When enabled, all scraping endpoints require an `X-API-Key` header. "
        "The `/api/health` endpoint is always public.",
        version="0.1.0",
    )

    # Shared dependency for all protected endpoints
    auth = [Depends(_verify_api_key)]

    # -----------------------------------------------------------------------
    # Fetcher endpoints (HTTP-based scraping)
    # -----------------------------------------------------------------------

    @app.post("/api/fetcher/get", response_model=ScrapeResponse, tags=["Fetcher"], dependencies=auth)
    async def fetcher_get(req: FetcherGetRequest):
        """Perform an HTTP GET request using Scrapling's Fetcher (curl_cffi)."""
        from scrapling.fetchers import Fetcher

        logger.info("[fetcher/get] url=%s", req.url)
        if not await _acquire_semaphore(_fetcher_semaphore, _QUEUE_TIMEOUT):
            logger.warning("[fetcher/get] 503 queue timeout url=%s", req.url)
            raise HTTPException(
                status_code=503,
                detail="Server busy — too many concurrent fetcher requests. Try again shortly.",
            )
        try:
            kwargs = _build_fetcher_kwargs(req)
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(_worker_pool, partial(Fetcher.get, req.url, **kwargs))
        except HTTPException:
            raise
        except Exception as e:
            logger.error("[fetcher/get] 502 url=%s error=%s\n%s", req.url, e, traceback.format_exc())
            raise HTTPException(status_code=502, detail=str(e))
        finally:
            _fetcher_semaphore.release()

        logger.info("[fetcher/get] 200 url=%s status=%s", req.url, response.status)
        return JSONResponse(content=_response_to_dict(response, req.css_selector, req.xpath_selector))

    @app.post("/api/fetcher/post", response_model=ScrapeResponse, tags=["Fetcher"], dependencies=auth)
    async def fetcher_post(req: FetcherDataRequest):
        """Perform an HTTP POST request using Scrapling's Fetcher (curl_cffi)."""
        from scrapling.fetchers import Fetcher

        logger.info("[fetcher/post] url=%s", req.url)
        if not await _acquire_semaphore(_fetcher_semaphore, _QUEUE_TIMEOUT):
            logger.warning("[fetcher/post] 503 queue timeout url=%s", req.url)
            raise HTTPException(
                status_code=503,
                detail="Server busy — too many concurrent fetcher requests. Try again shortly.",
            )
        try:
            kwargs = _build_fetcher_kwargs(req)
            if req.data:
                kwargs["data"] = req.data
            if req.json_data is not None:
                kwargs["json"] = req.json_data
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(_worker_pool, partial(Fetcher.post, req.url, **kwargs))
        except HTTPException:
            raise
        except Exception as e:
            logger.error("[fetcher/post] 502 url=%s error=%s\n%s", req.url, e, traceback.format_exc())
            raise HTTPException(status_code=502, detail=str(e))
        finally:
            _fetcher_semaphore.release()

        logger.info("[fetcher/post] 200 url=%s status=%s", req.url, response.status)
        return JSONResponse(content=_response_to_dict(response, req.css_selector, req.xpath_selector))

    @app.post("/api/fetcher/put", response_model=ScrapeResponse, tags=["Fetcher"], dependencies=auth)
    async def fetcher_put(req: FetcherDataRequest):
        """Perform an HTTP PUT request using Scrapling's Fetcher (curl_cffi)."""
        from scrapling.fetchers import Fetcher

        logger.info("[fetcher/put] url=%s", req.url)
        if not await _acquire_semaphore(_fetcher_semaphore, _QUEUE_TIMEOUT):
            logger.warning("[fetcher/put] 503 queue timeout url=%s", req.url)
            raise HTTPException(
                status_code=503,
                detail="Server busy — too many concurrent fetcher requests. Try again shortly.",
            )
        try:
            kwargs = _build_fetcher_kwargs(req)
            if req.data:
                kwargs["data"] = req.data
            if req.json_data is not None:
                kwargs["json"] = req.json_data
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(_worker_pool, partial(Fetcher.put, req.url, **kwargs))
        except HTTPException:
            raise
        except Exception as e:
            logger.error("[fetcher/put] 502 url=%s error=%s\n%s", req.url, e, traceback.format_exc())
            raise HTTPException(status_code=502, detail=str(e))
        finally:
            _fetcher_semaphore.release()

        logger.info("[fetcher/put] 200 url=%s status=%s", req.url, response.status)
        return JSONResponse(content=_response_to_dict(response, req.css_selector, req.xpath_selector))

    @app.post("/api/fetcher/delete", response_model=ScrapeResponse, tags=["Fetcher"], dependencies=auth)
    async def fetcher_delete(req: FetcherGetRequest):
        """Perform an HTTP DELETE request using Scrapling's Fetcher (curl_cffi)."""
        from scrapling.fetchers import Fetcher

        logger.info("[fetcher/delete] url=%s", req.url)
        if not await _acquire_semaphore(_fetcher_semaphore, _QUEUE_TIMEOUT):
            logger.warning("[fetcher/delete] 503 queue timeout url=%s", req.url)
            raise HTTPException(
                status_code=503,
                detail="Server busy — too many concurrent fetcher requests. Try again shortly.",
            )
        try:
            kwargs = _build_fetcher_kwargs(req)
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(_worker_pool, partial(Fetcher.delete, req.url, **kwargs))
        except HTTPException:
            raise
        except Exception as e:
            logger.error("[fetcher/delete] 502 url=%s error=%s\n%s", req.url, e, traceback.format_exc())
            raise HTTPException(status_code=502, detail=str(e))
        finally:
            _fetcher_semaphore.release()

        logger.info("[fetcher/delete] 200 url=%s status=%s", req.url, response.status)
        return JSONResponse(content=_response_to_dict(response, req.css_selector, req.xpath_selector))

    # -----------------------------------------------------------------------
    # DynamicFetcher endpoints (browser-based scraping)
    # -----------------------------------------------------------------------

    @app.post("/api/dynamic/fetch", response_model=ScrapeResponse, tags=["DynamicFetcher"], dependencies=auth)
    async def dynamic_fetch(req: DynamicFetchRequest):
        """Fetch a page using Scrapling's DynamicFetcher (Playwright/Chromium browser)."""
        from scrapling.fetchers import DynamicFetcher

        logger.info("[dynamic/fetch] url=%s headless=%s network_idle=%s", req.url, req.headless, req.network_idle)
        if not await _acquire_semaphore(_browser_semaphore, _QUEUE_TIMEOUT):
            logger.warning("[dynamic/fetch] 503 queue timeout url=%s (waited %ds)", req.url, _QUEUE_TIMEOUT)
            raise HTTPException(
                status_code=503,
                detail="Server busy — too many concurrent browser requests. Try again shortly.",
            )
        try:
            kwargs = _build_dynamic_kwargs(req)
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(_worker_pool, partial(DynamicFetcher.fetch, req.url, **kwargs))
        except HTTPException:
            raise
        except (BlockingIOError, OSError) as e:
            logger.error(
                "[dynamic/fetch] 503 resource exhaustion url=%s error=%s\n%s", req.url, e, traceback.format_exc()
            )
            raise HTTPException(
                status_code=503,
                detail=f"Server out of resources — cannot spawn browser process: {e}",
            )
        except Exception as e:
            logger.error("[dynamic/fetch] 502 url=%s error=%s\n%s", req.url, e, traceback.format_exc())
            raise HTTPException(status_code=502, detail=str(e))
        finally:
            _browser_semaphore.release()

        logger.info("[dynamic/fetch] 200 url=%s status=%s", req.url, response.status)
        return JSONResponse(content=_response_to_dict(response, req.css_selector, req.xpath_selector))

    # -----------------------------------------------------------------------
    # StealthyFetcher endpoints (stealth browser scraping)
    # -----------------------------------------------------------------------

    @app.post("/api/stealthy/fetch", response_model=ScrapeResponse, tags=["StealthyFetcher"], dependencies=auth)
    async def stealthy_fetch(req: StealthyFetchRequest):
        """Fetch a page using Scrapling's StealthyFetcher (stealth Chromium with anti-bot bypass)."""
        from scrapling.fetchers import StealthyFetcher

        logger.info("[stealthy/fetch] url=%s headless=%s network_idle=%s", req.url, req.headless, req.network_idle)
        if not await _acquire_semaphore(_browser_semaphore, _QUEUE_TIMEOUT):
            logger.warning("[stealthy/fetch] 503 queue timeout url=%s (waited %ds)", req.url, _QUEUE_TIMEOUT)
            raise HTTPException(
                status_code=503,
                detail="Server busy — too many concurrent browser requests. Try again shortly.",
            )
        try:
            kwargs = _build_stealthy_kwargs(req)
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(_worker_pool, partial(StealthyFetcher.fetch, req.url, **kwargs))
        except HTTPException:
            raise
        except (BlockingIOError, OSError) as e:
            logger.error(
                "[stealthy/fetch] 503 resource exhaustion url=%s error=%s\n%s", req.url, e, traceback.format_exc()
            )
            raise HTTPException(
                status_code=503,
                detail=f"Server out of resources — cannot spawn browser process: {e}",
            )
        except Exception as e:
            logger.error("[stealthy/fetch] 502 url=%s error=%s\n%s", req.url, e, traceback.format_exc())
            raise HTTPException(status_code=502, detail=str(e))
        finally:
            _browser_semaphore.release()

        return JSONResponse(content=_response_to_dict(response, req.css_selector, req.xpath_selector))

    # -----------------------------------------------------------------------
    # Health check (always public — no auth required)
    # -----------------------------------------------------------------------

    @app.get("/api/health", tags=["Health"])
    def health():
        """Health check endpoint (no authentication required)."""
        return {"status": "ok"}

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the Scrapling REST API server."""
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info(
        "Starting Scrapling API — MAX_BROWSERS=%d, MAX_FETCHERS=%d, QUEUE_TIMEOUT=%ds, POOL_WORKERS=%d",
        _MAX_BROWSERS,
        _MAX_FETCHERS,
        _QUEUE_TIMEOUT,
        _MAX_BROWSERS + _MAX_FETCHERS,
    )

    app = create_app()
    uvicorn.run(app, host=host, port=port)
