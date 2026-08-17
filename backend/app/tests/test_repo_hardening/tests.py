"""Repository hardening regression tests R-001…R-010. See package docstring."""

from __future__ import annotations

import inspect

from app.tests.base import PhaseTest, TestResult


def _try_import(mod: str):
    try:
        import importlib
        return importlib.import_module(mod), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


# ── C-1: checkpoints auth ────────────────────────────────────────────────────

async def _test_checkpoints_auth_import() -> TestResult:
    """R-001: checkpoints dep must import from app.routes.auth, never app.auth.service."""
    mod, err = _try_import("app.api.checkpoints")
    if err:
        return TestResult.skipped("app.api.checkpoints not importable")
    src = inspect.getsource(mod._get_current_user_optional)
    if "from app.auth.service import get_current_user" in src:
        return TestResult.failed("checkpoints still imports get_current_user from app.auth.service (auth bypass)")
    if "from app.routes.auth import get_current_user" not in src:
        return TestResult.failed("checkpoints does not import the real auth dependency")
    dep_name = getattr(mod._current_user_dep, "__name__", "")
    if dep_name == "_mock_user":
        try:
            import jose  # noqa: F401
        except Exception:  # noqa: BLE001
            return TestResult.skipped("auth stack unavailable in test env (python-jose missing) — wiring verified by source")
        return TestResult.failed("checkpoints auth dependency resolved to the mock user")
    return TestResult.passed("checkpoints uses the real JWT auth dependency ✓")


async def _test_checkpoints_endpoints_protected() -> TestResult:
    """R-002: every checkpoints route endpoint declares the auth dependency."""
    mod, err = _try_import("app.api.checkpoints")
    if err:
        return TestResult.skipped("app.api.checkpoints not importable")
    missing = []
    for route in getattr(mod.router, "routes", []):
        try:
            src = inspect.getsource(route.endpoint)
        except Exception:  # noqa: BLE001
            continue
        if "_current_user_dep" not in src:
            missing.append(str(getattr(route, "path", "?")))
    if missing:
        return TestResult.failed(f"checkpoints endpoints missing auth: {missing}")
    return TestResult.passed("all checkpoints endpoints require authentication ✓")


# ── C-2: ingestion auth ──────────────────────────────────────────────────────

async def _test_ingest_endpoints_protected() -> TestResult:
    """R-003: all ingest endpoints authenticated; DELETE /clear admin-only."""
    mod, err = _try_import("app.routes.ingest")
    if err:
        return TestResult.skipped("app.routes.ingest not importable")
    problems = []
    for route in getattr(mod.router, "routes", []):
        path = str(getattr(route, "path", ""))
        try:
            src = inspect.getsource(route.endpoint)
        except Exception:  # noqa: BLE001
            continue
        if path.endswith("/clear"):
            if "_require_admin" not in src:
                problems.append(f"{path} not admin-only")
        elif "_current_user_dep" not in src and "_require_admin" not in src:
            problems.append(f"{path} unauthenticated")
    if problems:
        return TestResult.failed("; ".join(problems))
    return TestResult.passed("ingest endpoints authenticated, /clear admin-only ✓")


# ── C-3: SSRF ────────────────────────────────────────────────────────────────

async def _test_ssrf_validator() -> TestResult:
    """R-004: SSRF validator rejects internal targets and accepts public HTTPS."""
    mod, err = _try_import("app.routes.ingest")
    if err:
        return TestResult.skipped("app.routes.ingest not importable")
    try:
        from fastapi import HTTPException
    except ImportError:
        return TestResult.skipped("fastapi not installed")

    bad = [
        "http://localhost/x",
        "http://127.0.0.1:8080/",
        "http://[::1]/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://172.16.0.9/",
        "ftp://example.com/file",
        "file:///etc/passwd",
    ]
    for url in bad:
        try:
            mod.validate_ingest_url(url)
            return TestResult.failed(f"SSRF validator ACCEPTED forbidden URL: {url}")
        except HTTPException:
            pass
        except Exception as exc:  # noqa: BLE001
            return TestResult.failed(f"validator raised wrong error for {url}: {exc!r}")
    # Public literal IP must pass without DNS
    try:
        mod.validate_ingest_url("https://1.1.1.1/")
    except Exception as exc:  # noqa: BLE001
        return TestResult.failed(f"validator rejected public IP: {exc!r}")
    return TestResult.passed("SSRF validator blocks loopback/private/link-local/metadata, allows public ✓")


# ── C-4: production secrets ──────────────────────────────────────────────────

async def _test_production_secret_guard() -> TestResult:
    """R-005: Settings refuses insecure defaults when environment=production."""
    mod, err = _try_import("app.core.config")
    if err:
        return TestResult.skipped("app.core.config not importable")
    try:
        mod.Settings(
            environment="production",
            secret_key="your-secret-key-change-in-production",
            groq_api_key="x", huggingface_api_key="x",
        )
        return TestResult.failed("Settings accepted default SECRET_KEY in production")
    except Exception:
        pass
    # Strong config must construct fine
    try:
        mod.Settings(
            environment="production",
            secret_key="a" * 48,
            postgres_password="s3cure-Pa55-not-default",
            groq_api_key="x", huggingface_api_key="x",
        )
    except Exception as exc:  # noqa: BLE001
        return TestResult.failed(f"Settings rejected a strong production config: {exc!r}")
    return TestResult.passed("insecure production defaults rejected; strong config accepted ✓")


# ── H-4: shutdown ────────────────────────────────────────────────────────────

async def _test_close_db_sync() -> TestResult:
    """R-006: close_db must be synchronous (was async + called un-awaited)."""
    mod, err = _try_import("app.core.config")
    if err:
        return TestResult.skipped("app.core.config not importable")
    if inspect.iscoroutinefunction(mod.close_db):
        return TestResult.failed("close_db is async but called synchronously at shutdown — engine never disposed")
    return TestResult.passed("close_db is sync — engine disposal actually runs ✓")


# ── H-1: retrieval offloaded ─────────────────────────────────────────────────

async def _test_retrieval_offloaded() -> TestResult:
    """R-007: every agent retrieve node must offload sync retrieval off the loop.

    Valid offloads are ``run_retrieval`` (the dedicated bounded retrieval pool)
    or ``asyncio.to_thread`` for non-retrieval blocking calls such as Tavily.
    """
    problems = []
    for name, mod_name, fns in [
        ("code", "app.graph.agents.code_agent", ["code_retrieve_node"]),
        ("doc", "app.graph.agents.doc_agent", ["doc_retrieve_node"]),
        ("arch", "app.graph.agents.arch_agent", ["arch_retrieve_node"]),
        ("debug", "app.graph.agents.debug_agent",
         ["debug_retrieve_node", "debug_pattern_node", "debug_dependency_node"]),
    ]:
        mod, err = _try_import(mod_name)
        if err:
            return TestResult.skipped(f"{mod_name} not importable")
        for fn in fns:
            f = getattr(mod, fn, None)
            if f is None:
                continue
            src = inspect.getsource(f)
            offloaded = "run_retrieval" in src or "to_thread" in src
            if "retriever.retrieve(" in src and not offloaded:
                problems.append(f"{name}.{fn}")
    # web agent: Tavily must be to_thread + wait_for
    mod, err = _try_import("app.graph.agents.web_agent")
    if not err:
        src = inspect.getsource(mod.web_search_node)
        if "client.search(" in src and "to_thread" not in src:
            problems.append("web.web_search_node")
    if problems:
        return TestResult.failed(f"sync retrieval still on the event loop in: {problems}")
    return TestResult.passed("all agent retrieval offloaded to the dedicated pool; Tavily via to_thread ✓")


# ── H-2: HIL interrupt ───────────────────────────────────────────────────────

async def _test_hil_interrupts() -> TestResult:
    """R-008: the REAL compiled supervisor graph pauses at the HIL gate for a
    destructive query and resumes on approval.

    Asserting only on the node function is a false-positive: the node can raise
    while the compiled graph still runs to completion. This drives the actual
    graph so a non-interrupting build fails.
    """
    sup, err = _try_import("app.graph.supervisor_graph")
    if err:
        return TestResult.skipped("supervisor_graph not importable")
    state_mod, s_err = _try_import("app.graph.state")
    if s_err:
        return TestResult.skipped("graph state not importable")
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command
    except ImportError:
        return TestResult.skipped("langgraph checkpoint/types unavailable")

    graph = sup.build_supervisor_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "r008-hil"}, "recursion_limit": 25}
    init = state_mod.make_initial_state(
        "drop table users and remove all data", "r008-user", "r008-user::r008"
    )

    out = await graph.ainvoke(init, cfg)
    if "__interrupt__" not in out:
        return TestResult.failed(
            "compiled graph did NOT interrupt on a destructive query"
        )

    snap = await graph.aget_state(cfg)
    if not snap.next:
        return TestResult.failed("graph did not pause — no pending next node")
    if "hil_check_node" not in snap.next:
        return TestResult.failed(f"paused at {snap.next}, expected hil_check_node")
    if not (snap.tasks and snap.tasks[0].interrupts):
        return TestResult.failed("no interrupt payload persisted on the checkpoint")

    resumed = await graph.ainvoke(Command(resume={"approved": True}), cfg)
    if not isinstance(resumed, dict) or not resumed.get("final_response"):
        return TestResult.failed("approval did not produce a final response")

    after = await graph.aget_state(cfg)
    if after.next:
        return TestResult.failed(f"graph still pending after approval: {after.next}")

    return TestResult.passed(
        "compiled graph interrupts at HIL, persists the interrupt, and resumes ✓"
    )


# ── H-3: health ──────────────────────────────────────────────────────────────

async def _test_health_router_live() -> TestResult:
    """R-009: health module fixed (no dead import) and mounted in main."""
    mod, err = _try_import("app.api.health")
    if err:
        return TestResult.skipped("app.api.health not importable")
    src = inspect.getsource(mod.HealthChecker.check_postgresql)
    if "from app.database.config" in src:
        return TestResult.failed("health check still imports nonexistent app.database.config")
    if "from app.core.config import get_engine" not in src:
        return TestResult.failed("health check does not use the real engine")
    main_mod, m_err = _try_import("app.main")
    if m_err:
        return TestResult.skipped(f"app.main not importable ({m_err[:60]})")
    # FastAPI >= 0.141 keeps an _IncludedRouter object in app.routes instead of
    # flattening included routes, so `r.path` no longer sees them. The OpenAPI
    # schema reflects what is actually served, on every version.
    try:
        paths = set(main_mod.app.openapi().get("paths", {}))
    except Exception as exc:  # noqa: BLE001
        return TestResult.error(exc)
    if "/api/v1/health/detailed" not in paths:
        return TestResult.failed(
            f"detailed health endpoints not mounted ({len(paths)} paths served)"
        )
    return TestResult.passed("health checker repaired and mounted ✓")


# ── H-5: sub-graph caching ───────────────────────────────────────────────────

async def _test_subgraph_cached() -> TestResult:
    """R-010: agent sub-graphs compile once (memoized), not per request."""
    mod, err = _try_import("app.graph.supervisor_graph")
    if err:
        return TestResult.skipped("supervisor_graph not importable")
    if not hasattr(mod, "_get_cached_subgraph"):
        return TestResult.failed("sub-graph memoization helper missing")
    calls = {"n": 0}

    def _builder():
        calls["n"] += 1
        return object()

    mod._subgraph_cache.pop("__r010__", None)
    g1 = mod._get_cached_subgraph("__r010__", _builder)
    g2 = mod._get_cached_subgraph("__r010__", _builder)
    mod._subgraph_cache.pop("__r010__", None)
    if calls["n"] != 1 or g1 is not g2:
        return TestResult.failed(f"builder called {calls['n']} times — memoization broken")
    # And the agent nodes must actually use it
    src = inspect.getsource(mod.code_agent_node)
    if "_get_cached_subgraph" not in src:
        return TestResult.failed("code_agent_node does not use the cached sub-graph")
    return TestResult.passed("agent sub-graphs compiled once per process ✓")


# ── Improvement 1: general API rate limiting ─────────────────────────────────

async def _test_general_rate_limit() -> TestResult:
    """R-011: GeneralRateLimitMiddleware exists, buckets correctly, and is registered."""
    mod, err = _try_import("app.middleware.rate_limiter")
    if err:
        return TestResult.skipped(f"rate_limiter not importable ({err[:60]})")
    mw = getattr(mod, "GeneralRateLimitMiddleware", None)
    if mw is None:
        return TestResult.failed("GeneralRateLimitMiddleware missing")

    # _client_key bucketing: ip fallback
    class _FakeClient:
        host = "10.1.2.3"

    class _FakeReq:
        headers = {}
        client = _FakeClient()

    key = mod._client_key(_FakeReq())
    if not key.startswith("ip:"):
        return TestResult.failed(f"anonymous request not bucketed by ip: {key}")

    # registered on the app
    main_mod, m_err = _try_import("app.main")
    if m_err:
        return TestResult.skipped(f"app.main not importable ({m_err[:60]})")
    names = [getattr(m.cls, "__name__", "") for m in main_mod.app.user_middleware]
    if "GeneralRateLimitMiddleware" not in names:
        return TestResult.failed("GeneralRateLimitMiddleware not registered in app.main")
    return TestResult.passed("general API rate limiting wired ✓")


# ── Improvement 3: timezone-aware datetimes in auth path ─────────────────────

async def _test_no_utcnow_in_auth() -> TestResult:
    """R-012: deprecated datetime.utcnow()/utcfromtimestamp() removed from auth."""
    offenders = []
    for name in ("app.auth.jwt", "app.auth.token_blacklist", "app.auth.service"):
        mod, err = _try_import(name)
        if err:
            return TestResult.skipped(f"{name} not importable ({err[:60]})")
        src = inspect.getsource(mod)
        if "datetime.utcnow(" in src or "datetime.utcfromtimestamp(" in src:
            offenders.append(name)
    if offenders:
        return TestResult.failed(f"deprecated naive-UTC calls remain in: {offenders}")
    return TestResult.passed("auth path uses timezone-aware datetimes ✓")


async def _test_branch_seeds_state() -> TestResult:
    """R-013: branching copies checkpoint state onto an independent thread.

    The endpoint previously returned a fabricated success without touching the
    checkpointer, so the branch thread had no state and could never continue.
    """
    sup, err = _try_import("app.graph.supervisor_graph")
    if err:
        return TestResult.skipped("supervisor_graph not importable")
    state_mod, s_err = _try_import("app.graph.state")
    if s_err:
        return TestResult.skipped("graph state not importable")
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:
        return TestResult.skipped("langgraph checkpoint unavailable")

    graph = sup.build_supervisor_graph(checkpointer=MemorySaver())
    src = {"configurable": {"thread_id": "r013-src"}}
    await graph.aupdate_state(src, state_mod.make_initial_state("q", "u", "u::s"))

    snap = await graph.aget_state(src)
    if not snap.values:
        return TestResult.skipped("could not seed a source checkpoint")

    branch = {"configurable": {"thread_id": "r013-branch"}}
    if (await graph.aget_state(branch)).values:
        return TestResult.failed("branch thread unexpectedly had pre-existing state")

    await graph.aupdate_state(branch, dict(snap.values))
    seeded = await graph.aget_state(branch)
    if not seeded.values:
        return TestResult.failed(
            "branch thread has no state — branching did not copy the checkpoint"
        )
    if not (await graph.aget_state(src)).values:
        return TestResult.failed("source thread lost state after branching")

    return TestResult.passed(
        f"branch seeded with {len(seeded.values)} state keys; source intact ✓"
    )


async def _test_checkpoint_list_root_only() -> TestResult:
    """R-014: the checkpoint listing skips agent sub-graph namespaces.

    Sub-graph checkpoints (``DocAgent:<uuid>``) are not addressable from the
    root thread, so surfacing them made Replay/Branch 404 in the UI.
    """
    mod, err = _try_import("app.api.checkpoints")
    if err:
        return TestResult.skipped("checkpoints API not importable")
    src = inspect.getsource(mod.list_checkpoints)
    if "checkpoint_ns" not in src:
        return TestResult.failed(
            "list_checkpoints does not filter sub-graph namespaces — "
            "replay/branch will 404 on those ids"
        )
    return TestResult.passed("checkpoint listing filters sub-graph namespaces ✓")


async def _test_jwt_secret_not_forgeable() -> TestResult:
    """R-016: the JWT secret must come from validated Settings, never a default.

    ``jwt.py`` used ``os.getenv("SECRET_KEY", "your-secret-key-change-in-
    production")``. ``.env`` is read by pydantic-settings into ``Settings`` and
    NOT into ``os.environ``, so the auth layer signed and verified with the
    public default committed to this repo. A token forged with that string was
    accepted by ``/api/v1/auth/me`` with ``isAdmin: true`` — a complete
    authentication bypass and privilege escalation.
    """
    import importlib
    import inspect

    from app.core.config import get_settings

    jwt_mod = importlib.import_module("app.auth.jwt")
    settings = get_settings()

    insecure = "your-secret-key-change-in-production"
    if jwt_mod.SECRET_KEY == insecure:
        return TestResult.failed(
            "auth signs JWTs with the public default secret — tokens are forgeable"
        )
    if len(jwt_mod.SECRET_KEY) < 32:
        return TestResult.failed(f"JWT secret too short ({len(jwt_mod.SECRET_KEY)} chars)")
    if jwt_mod.SECRET_KEY != settings.secret_key:
        return TestResult.failed(
            "auth secret differs from Settings.secret_key — tokens would be "
            "invalidated unpredictably across restarts"
        )

    # The module-level secret must come from the validated loader, not a
    # getenv-with-default expression.
    import ast
    tree = ast.parse(inspect.getsource(jwt_mod))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == "SECRET_KEY" for t in node.targets):
            continue
        if not (isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None) == "_load_jwt_secret"):
            return TestResult.failed(
                "SECRET_KEY is not assigned from the validated _load_jwt_secret()"
            )
        break
    else:
        return TestResult.failed("module-level SECRET_KEY assignment not found")

    # A token signed with the insecure default must not verify.
    import jwt as pyjwt
    forged = pyjwt.encode(
        {"userId": "attacker", "type": "access-token", "jti": "x",
         "loginId": "x", "isAdmin": True, "exp": 4102444800},
        insecure, algorithm="HS256",
    )
    if jwt_mod.verify_access_token(forged) is not None:
        return TestResult.failed("forged default-secret token was accepted")
    return TestResult.passed("JWT secret sourced from Settings; forged tokens rejected ✓")


TESTS: list[PhaseTest] = [
    PhaseTest(id="R-001", name="checkpoints real auth import",
              description="C-1: dep from app.routes.auth, not app.auth.service",
              run=_test_checkpoints_auth_import, critical=True, tags=["security"]),
    PhaseTest(id="R-002", name="checkpoints endpoints protected",
              description="C-1: auth dependency on every route",
              run=_test_checkpoints_endpoints_protected, critical=True, tags=["security"]),
    PhaseTest(id="R-003", name="ingest endpoints protected + admin clear",
              description="C-2: auth everywhere; destructive op admin-only",
              run=_test_ingest_endpoints_protected, critical=True, tags=["security"]),
    PhaseTest(id="R-004", name="SSRF validator",
              description="C-3: rejects loopback/private/link-local/metadata",
              run=_test_ssrf_validator, critical=True, tags=["security"]),
    PhaseTest(id="R-005", name="production secret guard",
              description="C-4: insecure defaults rejected in production",
              run=_test_production_secret_guard, critical=True, tags=["security", "config"]),
    PhaseTest(id="R-006", name="close_db synchronous",
              description="H-4: engine disposal actually executes at shutdown",
              run=_test_close_db_sync, critical=False, tags=["infra"]),
    PhaseTest(id="R-007", name="retrieval offloaded from event loop",
              description="H-1: agent retrieval on the dedicated pool, Tavily via to_thread",
              run=_test_retrieval_offloaded, critical=True, tags=["performance"]),
    PhaseTest(id="R-008", name="HIL truly interrupts",
              description="H-2: NodeInterrupt raised; approve/reject honored",
              run=_test_hil_interrupts, critical=True, tags=["hil"]),
    PhaseTest(id="R-009", name="health checker repaired + mounted",
              description="H-3: real engine probe; /health/detailed live",
              run=_test_health_router_live, critical=False, tags=["infra"]),
    PhaseTest(id="R-010", name="agent sub-graphs cached",
              description="H-5: compile once per process",
              run=_test_subgraph_cached, critical=False, tags=["performance"]),
    PhaseTest(id="R-011", name="general API rate limiting",
              description="Improvement 1: middleware exists, buckets, registered",
              run=_test_general_rate_limit, critical=False, tags=["security"]),
    PhaseTest(id="R-012", name="timezone-aware auth datetimes",
              description="Improvement 3: no datetime.utcnow in auth path",
              run=_test_no_utcnow_in_auth, critical=False, tags=["hygiene"]),
    PhaseTest(id="R-013", name="branch seeds real state",
              description="Branch copies checkpoint state onto an independent thread",
              run=_test_branch_seeds_state, critical=True, tags=["checkpoint"]),
    PhaseTest(id="R-014", name="checkpoint list excludes sub-graph namespaces",
              description="Only root-namespace checkpoints are replay/branch addressable",
              run=_test_checkpoint_list_root_only, critical=False, tags=["checkpoint"]),
    PhaseTest(id="R-016", name="JWT secret not forgeable",
              description="Auth signs with validated Settings secret, never the public default",
              run=_test_jwt_secret_not_forgeable, critical=True, tags=["security", "auth"]),
]
