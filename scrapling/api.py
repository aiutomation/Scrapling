"""
REST API server for Scrapling's fetcher classes.

Exposes Fetcher, DynamicFetcher, and StealthyFetcher via FastAPI endpoints.
Run with: scrapling api --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import os
import queue
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
_EXECUTION_TIMEOUT = int(os.environ.get("SCRAPLING_EXECUTION_TIMEOUT", "120"))
_MAX_RESPONSE_SIZE = int(os.environ.get("SCRAPLING_MAX_RESPONSE_SIZE", str(10 * 1024 * 1024)))  # 10MB
# Bounded thread pool prevents OS thread exhaustion — all blocking fetcher/browser
# work runs through this pool instead of FastAPI spawning unbounded threads.
_worker_pool = ThreadPoolExecutor(max_workers=_MAX_BROWSERS + _MAX_FETCHERS)

# Semaphores MUST be created lazily inside the running asyncio event loop.
# Creating asyncio.Semaphore() at module import time (before uvicorn starts its
# event loop) causes "Future attached to a different loop" errors on every await,
# which is why all requests return 502 immediately (~465ms — no browser work done).
_browser_semaphore: Optional[asyncio.Semaphore] = None
_fetcher_semaphore: Optional[asyncio.Semaphore] = None


def _get_browser_semaphore() -> asyncio.Semaphore:
    """Return the browser semaphore, creating it lazily in the running event loop."""
    global _browser_semaphore
    if _browser_semaphore is None:
        _browser_semaphore = asyncio.Semaphore(_MAX_BROWSERS)
    return _browser_semaphore


def _get_fetcher_semaphore() -> asyncio.Semaphore:
    """Return the fetcher semaphore, creating it lazily in the running event loop."""
    global _fetcher_semaphore
    if _fetcher_semaphore is None:
        _fetcher_semaphore = asyncio.Semaphore(_MAX_FETCHERS)
    return _fetcher_semaphore


async def _acquire_semaphore(sem: asyncio.Semaphore, timeout: int) -> bool:
    """Try to acquire an asyncio semaphore with a timeout. Returns True on success."""
    try:
        await asyncio.wait_for(sem.acquire(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


# ---------------------------------------------------------------------------
# Browser session pool — reuses browser instances to avoid per-request
# launch/teardown overhead. Sessions with default config are pooled;
# requests with custom browser-level settings get one-off sessions.
# ---------------------------------------------------------------------------


class _BrowserSessionPool:
    """Thread-safe pool of reusable browser sessions."""

    def __init__(self, max_dynamic: int, max_stealthy: int):
        self._dynamic: queue.Queue = queue.Queue(maxsize=max_dynamic)
        self._stealthy: queue.Queue = queue.Queue(maxsize=max_stealthy)

    def acquire_dynamic(self) -> Any:
        """Get a pooled DynamicSession or create a new one."""
        while True:
            try:
                session = self._dynamic.get_nowait()
                if session._is_alive:
                    session._reset_watchdog()
                    return session
                # Session died (watchdog closed it), discard and try next
                try:
                    session.close()
                except Exception:
                    pass
            except queue.Empty:
                break
        # Only detach the event loop right before creating a NEW session.
        # sync_playwright().start() checks for a running asyncio loop and
        # raises if one is found.  We must NOT detach when reusing a pooled
        # session — doing so kills patchright's internal greenlet event loop.
        _detach_event_loop()
        from scrapling.engines._browsers._controllers import DynamicSession
        from scrapling.engines.toolbelt.custom import BaseFetcher

        session = DynamicSession(
            headless=True,
            extra_flags=_CONTAINER_BROWSER_FLAGS,
            selector_config={**BaseFetcher._generate_parser_arguments()},
        )
        session.start()
        return session

    def acquire_stealthy(self) -> Any:
        """Get a pooled StealthySession or create a new one."""
        while True:
            try:
                session = self._stealthy.get_nowait()
                if session._is_alive:
                    session._reset_watchdog()
                    return session
                try:
                    session.close()
                except Exception:
                    pass
            except queue.Empty:
                break
        _detach_event_loop()
        from scrapling.engines._browsers._stealth import StealthySession
        from scrapling.engines.toolbelt.custom import BaseFetcher

        session = StealthySession(
            headless=True,
            extra_flags=_CONTAINER_BROWSER_FLAGS,
            selector_config={**BaseFetcher._generate_parser_arguments()},
        )
        session.start()
        return session

    def release(self, session: Any, pool_type: str) -> None:
        """Return a session to the pool, or close it if the pool is full."""
        pool = self._dynamic if pool_type == "dynamic" else self._stealthy
        try:
            if session._is_alive:
                pool.put_nowait(session)
                return
        except queue.Full:
            pass
        try:
            session.close()
        except Exception:
            pass

    def discard(self, session: Any) -> None:
        """Close a session without returning it to the pool (after errors)."""
        try:
            session.close()
        except Exception:
            pass

    def shutdown(self) -> None:
        """Close all pooled sessions."""
        for pool in (self._dynamic, self._stealthy):
            while not pool.empty():
                try:
                    pool.get_nowait().close()
                except Exception:
                    pass


_browser_pool = _BrowserSessionPool(max_dynamic=_MAX_BROWSERS, max_stealthy=_MAX_BROWSERS)

# Extra Chrome flags to reduce thread/process footprint in constrained containers.
# Without these, Chromium spawns many sub-processes and threads which can exhaust
# the container's PID/thread limits (pthread_create: Resource temporarily unavailable).
_CONTAINER_BROWSER_FLAGS = [
    "--renderer-process-limit=1",
    "--disable-gpu",
    "--disable-software-rasterizer",
]


def _is_poolable_dynamic(req: "DynamicFetchRequest") -> bool:
    """Check if request can reuse a pooled default-config browser session."""
    return (
        req.headless is True
        and req.real_chrome is False
        and req.cdp_url is None
        and req.locale is None
        and req.useragent is None
    )


def _is_poolable_stealthy(req: "StealthyFetchRequest") -> bool:
    """Check if stealthy request can reuse a pooled default-config session."""
    return (
        _is_poolable_dynamic(req) and req.allow_webgl is True and req.hide_canvas is False and req.block_webrtc is False
    )


def _build_fetch_only_kwargs(req: "DynamicFetchRequest") -> dict:
    """Build kwargs for session.fetch() — fetch-level overridable params only."""
    kwargs: Dict[str, Any] = {
        "disable_resources": req.disable_resources,
        "network_idle": req.network_idle,
        "load_dom": req.load_dom,
        "timeout": req.timeout,
        "google_search": req.google_search,
    }
    if req.wait is not None:
        kwargs["wait"] = req.wait
    if req.wait_selector:
        kwargs["wait_selector"] = req.wait_selector
    if req.wait_selector_state:
        kwargs["wait_selector_state"] = req.wait_selector_state
    if req.proxy:
        kwargs["proxy"] = req.proxy
    if req.extra_headers:
        kwargs["extra_headers"] = req.extra_headers
    return kwargs


def _build_stealthy_fetch_only_kwargs(req: "StealthyFetchRequest") -> dict:
    """Build kwargs for StealthySession.fetch() — fetch-level params only."""
    kwargs = _build_fetch_only_kwargs(req)
    kwargs["solve_cloudflare"] = req.solve_cloudflare
    return kwargs


def _detach_event_loop() -> None:
    """Detach the asyncio event loop from the current thread.

    When ``run_in_executor`` dispatches work to a ``ThreadPoolExecutor``
    thread, the thread can still "see" the parent's running event loop via
    ``asyncio.get_running_loop()`` (Python ≥3.12 behaviour).  Patchright's
    ``sync_playwright()`` checks for a running loop and raises
    ``"Playwright Sync API inside the asyncio loop"`` if one is found.

    We clear the loop at both the Python and C levels to prevent
    false-positive detection.
    """
    try:
        asyncio.set_event_loop(None)
    except Exception:
        pass
    # Also clear the C-level running loop reference that
    # asyncio.get_running_loop() / asyncio._get_running_loop() checks.
    try:
        asyncio._set_running_loop(None)  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        pass


def _is_retryable_session_error(e: Exception) -> bool:
    """Check if a pooled session error is retryable with a fresh session."""
    msg = str(e)
    return ("Target" in msg and "closed" in msg) or "no running event loop" in msg or "Event loop is closed" in msg


def _run_pooled_dynamic(url: str, req: "DynamicFetchRequest") -> Any:
    """Run a dynamic fetch using pooled session when possible, else one-off."""
    if not _is_poolable_dynamic(req):
        _detach_event_loop()
        from scrapling.fetchers import DynamicFetcher

        return DynamicFetcher.fetch(url, **_build_dynamic_kwargs(req))

    session = _browser_pool.acquire_dynamic()
    try:
        result = session.fetch(url, **_build_fetch_only_kwargs(req))
    except Exception as e:
        _browser_pool.discard(session)
        if _is_retryable_session_error(e):
            logger.warning(
                "[dynamic/fetch] Pooled session unusable (%s), retrying with fresh session for url=%s", e, url
            )
            session = _browser_pool.acquire_dynamic()
            try:
                result = session.fetch(url, **_build_fetch_only_kwargs(req))
            except Exception:
                _browser_pool.discard(session)
                raise
            _browser_pool.release(session, "dynamic")
            return result
        raise
    _browser_pool.release(session, "dynamic")
    return result


def _run_pooled_stealthy(url: str, req: "StealthyFetchRequest") -> Any:
    """Run a stealthy fetch using pooled session when possible, else one-off."""
    if not _is_poolable_stealthy(req):
        _detach_event_loop()
        from scrapling.fetchers import StealthyFetcher

        return StealthyFetcher.fetch(url, **_build_stealthy_kwargs(req))

    session = _browser_pool.acquire_stealthy()
    try:
        result = session.fetch(url, **_build_stealthy_fetch_only_kwargs(req))
    except Exception as e:
        _browser_pool.discard(session)
        if _is_retryable_session_error(e):
            logger.warning(
                "[stealthy/fetch] Pooled session unusable (%s), retrying with fresh session for url=%s", e, url
            )
            session = _browser_pool.acquire_stealthy()
            try:
                result = session.fetch(url, **_build_stealthy_fetch_only_kwargs(req))
            except Exception:
                _browser_pool.discard(session)
                raise
            _browser_pool.release(session, "stealthy")
            return result
        raise
    _browser_pool.release(session, "stealthy")
    return result


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

    # Body as string (truncated to _MAX_RESPONSE_SIZE to prevent OOM)
    body = ""
    try:
        raw = response.body
        if isinstance(raw, bytes):
            if len(raw) > _MAX_RESPONSE_SIZE:
                raw = raw[:_MAX_RESPONSE_SIZE]
            body = raw.decode("utf-8", errors="replace")
        else:
            body = str(raw)
            if len(body) > _MAX_RESPONSE_SIZE:
                body = body[:_MAX_RESPONSE_SIZE]
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
        if not await _acquire_semaphore(_get_fetcher_semaphore(), _QUEUE_TIMEOUT):
            logger.warning("[fetcher/get] 503 queue timeout url=%s", req.url)
            raise HTTPException(
                status_code=503,
                detail="Server busy — too many concurrent fetcher requests. Try again shortly.",
            )
        try:
            kwargs = _build_fetcher_kwargs(req)
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(_worker_pool, partial(Fetcher.get, req.url, **kwargs)),
                timeout=_EXECUTION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("[fetcher/get] execution timeout (%ds) url=%s", _EXECUTION_TIMEOUT, req.url)
            raise HTTPException(status_code=504, detail=f"Request timed out after {_EXECUTION_TIMEOUT}s.")
        except HTTPException:
            raise
        except Exception as e:
            logger.error("[fetcher/get] 502 url=%s error=%s\n%s", req.url, e, traceback.format_exc())
            raise HTTPException(status_code=502, detail=str(e))
        finally:
            _get_fetcher_semaphore().release()

        logger.info("[fetcher/get] 200 url=%s status=%s", req.url, response.status)
        return JSONResponse(content=_response_to_dict(response, req.css_selector, req.xpath_selector))

    @app.post("/api/fetcher/post", response_model=ScrapeResponse, tags=["Fetcher"], dependencies=auth)
    async def fetcher_post(req: FetcherDataRequest):
        """Perform an HTTP POST request using Scrapling's Fetcher (curl_cffi)."""
        from scrapling.fetchers import Fetcher

        logger.info("[fetcher/post] url=%s", req.url)
        if not await _acquire_semaphore(_get_fetcher_semaphore(), _QUEUE_TIMEOUT):
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
            response = await asyncio.wait_for(
                loop.run_in_executor(_worker_pool, partial(Fetcher.post, req.url, **kwargs)),
                timeout=_EXECUTION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("[fetcher/post] execution timeout (%ds) url=%s", _EXECUTION_TIMEOUT, req.url)
            raise HTTPException(status_code=504, detail=f"Request timed out after {_EXECUTION_TIMEOUT}s.")
        except HTTPException:
            raise
        except Exception as e:
            logger.error("[fetcher/post] 502 url=%s error=%s\n%s", req.url, e, traceback.format_exc())
            raise HTTPException(status_code=502, detail=str(e))
        finally:
            _get_fetcher_semaphore().release()

        logger.info("[fetcher/post] 200 url=%s status=%s", req.url, response.status)
        return JSONResponse(content=_response_to_dict(response, req.css_selector, req.xpath_selector))

    @app.post("/api/fetcher/put", response_model=ScrapeResponse, tags=["Fetcher"], dependencies=auth)
    async def fetcher_put(req: FetcherDataRequest):
        """Perform an HTTP PUT request using Scrapling's Fetcher (curl_cffi)."""
        from scrapling.fetchers import Fetcher

        logger.info("[fetcher/put] url=%s", req.url)
        if not await _acquire_semaphore(_get_fetcher_semaphore(), _QUEUE_TIMEOUT):
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
            response = await asyncio.wait_for(
                loop.run_in_executor(_worker_pool, partial(Fetcher.put, req.url, **kwargs)),
                timeout=_EXECUTION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("[fetcher/put] execution timeout (%ds) url=%s", _EXECUTION_TIMEOUT, req.url)
            raise HTTPException(status_code=504, detail=f"Request timed out after {_EXECUTION_TIMEOUT}s.")
        except HTTPException:
            raise
        except Exception as e:
            logger.error("[fetcher/put] 502 url=%s error=%s\n%s", req.url, e, traceback.format_exc())
            raise HTTPException(status_code=502, detail=str(e))
        finally:
            _get_fetcher_semaphore().release()

        logger.info("[fetcher/put] 200 url=%s status=%s", req.url, response.status)
        return JSONResponse(content=_response_to_dict(response, req.css_selector, req.xpath_selector))

    @app.post("/api/fetcher/delete", response_model=ScrapeResponse, tags=["Fetcher"], dependencies=auth)
    async def fetcher_delete(req: FetcherGetRequest):
        """Perform an HTTP DELETE request using Scrapling's Fetcher (curl_cffi)."""
        from scrapling.fetchers import Fetcher

        logger.info("[fetcher/delete] url=%s", req.url)
        if not await _acquire_semaphore(_get_fetcher_semaphore(), _QUEUE_TIMEOUT):
            logger.warning("[fetcher/delete] 503 queue timeout url=%s", req.url)
            raise HTTPException(
                status_code=503,
                detail="Server busy — too many concurrent fetcher requests. Try again shortly.",
            )
        try:
            kwargs = _build_fetcher_kwargs(req)
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(_worker_pool, partial(Fetcher.delete, req.url, **kwargs)),
                timeout=_EXECUTION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("[fetcher/delete] execution timeout (%ds) url=%s", _EXECUTION_TIMEOUT, req.url)
            raise HTTPException(status_code=504, detail=f"Request timed out after {_EXECUTION_TIMEOUT}s.")
        except HTTPException:
            raise
        except Exception as e:
            logger.error("[fetcher/delete] 502 url=%s error=%s\n%s", req.url, e, traceback.format_exc())
            raise HTTPException(status_code=502, detail=str(e))
        finally:
            _get_fetcher_semaphore().release()

        logger.info("[fetcher/delete] 200 url=%s status=%s", req.url, response.status)
        return JSONResponse(content=_response_to_dict(response, req.css_selector, req.xpath_selector))

    # -----------------------------------------------------------------------
    # DynamicFetcher endpoints (browser-based scraping)
    # -----------------------------------------------------------------------

    @app.post("/api/dynamic/fetch", response_model=ScrapeResponse, tags=["DynamicFetcher"], dependencies=auth)
    async def dynamic_fetch(req: DynamicFetchRequest):
        """Fetch a page using Scrapling's DynamicFetcher (Playwright/Chromium browser)."""
        logger.info("[dynamic/fetch] url=%s headless=%s network_idle=%s", req.url, req.headless, req.network_idle)
        if not await _acquire_semaphore(_get_browser_semaphore(), _QUEUE_TIMEOUT):
            logger.warning("[dynamic/fetch] 503 queue timeout url=%s (waited %ds)", req.url, _QUEUE_TIMEOUT)
            raise HTTPException(
                status_code=503,
                detail="Server busy — too many concurrent browser requests. Try again shortly.",
            )
        try:
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(_worker_pool, partial(_run_pooled_dynamic, req.url, req)),
                timeout=_EXECUTION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("[dynamic/fetch] execution timeout (%ds) url=%s", _EXECUTION_TIMEOUT, req.url)
            raise HTTPException(
                status_code=504,
                detail=f"Browser request timed out after {_EXECUTION_TIMEOUT}s.",
            )
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
            _get_browser_semaphore().release()

        logger.info("[dynamic/fetch] 200 url=%s status=%s", req.url, response.status)
        return JSONResponse(content=_response_to_dict(response, req.css_selector, req.xpath_selector))

    # -----------------------------------------------------------------------
    # StealthyFetcher endpoints (stealth browser scraping)
    # -----------------------------------------------------------------------

    @app.post("/api/stealthy/fetch", response_model=ScrapeResponse, tags=["StealthyFetcher"], dependencies=auth)
    async def stealthy_fetch(req: StealthyFetchRequest):
        """Fetch a page using Scrapling's StealthyFetcher (stealth Chromium with anti-bot bypass)."""
        logger.info("[stealthy/fetch] url=%s headless=%s network_idle=%s", req.url, req.headless, req.network_idle)
        if not await _acquire_semaphore(_get_browser_semaphore(), _QUEUE_TIMEOUT):
            logger.warning("[stealthy/fetch] 503 queue timeout url=%s (waited %ds)", req.url, _QUEUE_TIMEOUT)
            raise HTTPException(
                status_code=503,
                detail="Server busy — too many concurrent browser requests. Try again shortly.",
            )
        try:
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(_worker_pool, partial(_run_pooled_stealthy, req.url, req)),
                timeout=_EXECUTION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("[stealthy/fetch] execution timeout (%ds) url=%s", _EXECUTION_TIMEOUT, req.url)
            raise HTTPException(
                status_code=504,
                detail=f"Browser request timed out after {_EXECUTION_TIMEOUT}s.",
            )
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
            _get_browser_semaphore().release()

        return JSONResponse(content=_response_to_dict(response, req.css_selector, req.xpath_selector))

    # -----------------------------------------------------------------------
    # Health check (always public — no auth required)
    # -----------------------------------------------------------------------

    @app.get("/api/health", tags=["Health"])
    def health():
        """Health check endpoint (no authentication required)."""
        return {"status": "ok"}

    @app.on_event("shutdown")
    def shutdown_event():
        """Clean up browser pool and thread pool on server shutdown."""
        logger.info("Shutting down — closing browser pool and thread pool...")
        _browser_pool.shutdown()
        _worker_pool.shutdown(wait=False, cancel_futures=True)

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the Scrapling REST API server."""
    import sys

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info(
        "Starting Scrapling API — host=%s, port=%d, MAX_BROWSERS=%d, MAX_FETCHERS=%d, "
        "QUEUE_TIMEOUT=%ds, EXECUTION_TIMEOUT=%ds, MAX_RESPONSE_SIZE=%dMB, POOL_WORKERS=%d",
        host,
        port,
        _MAX_BROWSERS,
        _MAX_FETCHERS,
        _QUEUE_TIMEOUT,
        _EXECUTION_TIMEOUT,
        _MAX_RESPONSE_SIZE // (1024 * 1024),
        _MAX_BROWSERS + _MAX_FETCHERS,
    )

    try:
        app = create_app()
        uvicorn.run(app, host=host, port=port)
    except Exception:
        logger.critical("Server failed to start", exc_info=True)
        sys.exit(1)
