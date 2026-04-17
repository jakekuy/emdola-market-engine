"""
EMDOLA FastAPI backend.

Endpoints:
  POST /api/batch/start                      → {batch_id}
  GET  /api/batch/list                       → list[str]
  GET  /api/batch/{batch_id}/status          → BatchStatus
  GET  /api/batch/{batch_id}/results         → BatchResult
  POST /api/batch/{batch_id}/replay/{run_no} → {batch_id}

  WS   /ws/{batch_id}                        → tick events during simulation

WebSocket message types:
  {"type": "tick",          "run": int, "tick": int, "prices": {...},
   "type_activity": {...},  "shock_active": bool}
  {"type": "run_complete",  "run": int, "final_prices": {...}}
  {"type": "batch_complete","sanity_checks": [...]}
  {"type": "error",         "message": str}

Spec references: §13.1 (API design), §9.3 (seeds/WebSocket).
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.config.sector_config import SECTORS, AGENT_TYPES
from backend.models.calibration import CalibrationInput
from backend.models.results import BatchResult, BatchStatus
from backend.simulation.engine import BatchEngine
from backend.output.storage import save_batch, load_batch, list_batches

load_dotenv()


# ── In-memory batch state ──────────────────────────────────────────────────────

class _BatchState:
    """
    Tracks the live state of one batch run.
    Thread-safe reads are fine for simple int fields (GIL); the event queue
    uses asyncio.run_coroutine_threadsafe for cross-thread writes.
    """
    def __init__(self, total_runs: int, total_ticks: int) -> None:
        self.status: str = "running"       # "running" | "complete" | "error"
        self.total_runs: int = total_runs
        self.total_ticks: int = total_ticks
        self.runs_complete: int = 0
        self.current_tick: int = 0
        self.error_message: str = ""
        self.result: BatchResult | None = None
        # One queue per batch — WebSocket handler drains it.
        self.queue: asyncio.Queue = asyncio.Queue()
        # Pause control — set = running, clear = paused.
        self.pause_event: threading.Event = threading.Event()
        self.pause_event.set()


# Global state registry — {batch_id → _BatchState}
_batches: dict[str, _BatchState] = {}


# ── In-memory profile generation state ────────────────────────────────────────

class _ProfileState:
    """Tracks the live state of one profile generation job."""
    def __init__(self) -> None:
        self.status: str = "running"       # "running" | "complete" | "error"
        self.error_message: str = ""
        self.profiles = None               # ProfileSet once complete
        self.queue: asyncio.Queue = asyncio.Queue()


# Global profile store — {profiles_id → _ProfileState}
_profile_store: dict[str, _ProfileState] = {}

# Shared executor for simulation threads (CPU-bound work off the event loop).
_executor = ThreadPoolExecutor(max_workers=4)


# ── App lifecycle ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="EMDOLA", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as _Request

class _NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: _Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return response

app.add_middleware(_NoCacheMiddleware)

# Serve frontend static files.
import pathlib as _pathlib
_FRONTEND = _pathlib.Path(__file__).parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(str(_FRONTEND / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Request / response helpers ─────────────────────────────────────────────────

class ProfilesGenerateResponse(BaseModel):
    profiles_id: str


class SavedProfilesMeta(BaseModel):
    generated_at: str
    scenario_name: str
    scenario_summary: str
    source: str  # "default" or "custom"


class LoadSavedProfilesResponse(BaseModel):
    profiles_id: str
    profiles: dict
    source: str  # "default" or "custom"


class BatchStartResponse(BaseModel):
    batch_id: str


class ReplayStartResponse(BaseModel):
    batch_id: str


# ── POST /api/profiles/generate ──────────────────────────────────────────────

@app.post("/api/profiles/generate", response_model=ProfilesGenerateResponse)
async def generate_profiles_endpoint(calibration: CalibrationInput) -> ProfilesGenerateResponse:
    """
    Accept a CalibrationInput, launch LLM profile generation in a background
    thread.  Returns profiles_id immediately.

    Connect to WS /ws/profiles/{profiles_id} to receive per-type progress
    events and the final profiles_complete payload.
    """
    profiles_id = f"profiles_{uuid.uuid4().hex[:12]}"
    state = _ProfileState()
    _profile_store[profiles_id] = state

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor,
        _run_profiles_thread,
        profiles_id,
        calibration,
        state,
        loop,
    )

    return ProfilesGenerateResponse(profiles_id=profiles_id)


# ── WS /ws/profiles/{profiles_id} ────────────────────────────────────────────

@app.websocket("/ws/profiles/{profiles_id}")
async def ws_profiles(websocket: WebSocket, profiles_id: str) -> None:
    """Stream per-type progress events for profile generation."""
    await websocket.accept()

    state = _profile_store.get(profiles_id)
    if state is None:
        await websocket.send_json({"type": "error", "message": f"Profiles '{profiles_id}' not found."})
        await websocket.close()
        return

    _TYPE_TIMEOUT  = 600.0   # 10 min per individual type
    _TOTAL_TIMEOUT = 1500.0  # 25 min total

    loop      = asyncio.get_running_loop()
    start     = loop.time()
    timeout_msg: str | None = None
    try:
        while True:
            elapsed   = loop.time() - start
            time_left = _TOTAL_TIMEOUT - elapsed
            if time_left <= 0:
                timeout_msg = "Profile generation timed out — total time (25 min) exceeded."
                break
            event = await asyncio.wait_for(
                state.queue.get(), timeout=min(_TYPE_TIMEOUT, time_left)
            )
            await websocket.send_json(event)
            if event.get("type") in ("profiles_complete", "error"):
                break
    except asyncio.TimeoutError:
        elapsed = loop.time() - start
        if elapsed >= _TOTAL_TIMEOUT - 5:
            timeout_msg = "Profile generation timed out — total time (25 min) exceeded."
        else:
            timeout_msg = "Profile generation timed out — one type took longer than 10 minutes. Try again."
    except WebSocketDisconnect:
        pass
    finally:
        if timeout_msg:
            try:
                await websocket.send_json({"type": "error", "message": timeout_msg})
            except Exception:
                pass
        await websocket.close()


# ── GET /api/profiles/saved ──────────────────────────────────────────────────

@app.get("/api/profiles/saved", response_model=SavedProfilesMeta)
async def get_saved_profiles_meta() -> SavedProfilesMeta:
    """
    Return metadata for the saved profile set, if one exists.
    Raises 404 if no profiles have been saved yet.
    """
    meta = _load_saved_profiles_meta()
    if meta is None:
        raise HTTPException(status_code=404, detail="No saved profiles found.")
    return SavedProfilesMeta(**meta)


# ── POST /api/profiles/saved/load ────────────────────────────────────────────

@app.post("/api/profiles/saved/load", response_model=LoadSavedProfilesResponse)
async def load_saved_profiles() -> LoadSavedProfilesResponse:
    """
    Load the active profile set from disk into memory.
    Tries user-generated profiles first; falls back to locked defaults.
    Returns profiles_id ('profiles_saved'), source ('default'|'custom'),
    and the full profiles dict for rendering in the frontend profile review view.
    Raises 404 if neither profile file exists.
    """
    result = _read_saved_profiles_file()
    if result is None:
        raise HTTPException(status_code=404, detail="No saved profiles found.")
    data, source = result

    from backend.models.profile import ProfileSet, PersonaProfile, AgentCharacteristics

    # Reconstruct ProfileSet from stored dict.
    personas_dict = {}
    for agent_type, persona_list in data["profiles"].items():
        personas_dict[agent_type] = [
            PersonaProfile(
                agent_type=agent_type,
                persona_number=p["persona_number"],
                narrative=p["narrative"],
                characteristics=AgentCharacteristics(**{
                    k: v for k, v in p["characteristics"].items()
                }),
            )
            for p in persona_list
        ]

    profile_set = ProfileSet(personas=personas_dict)

    state = _ProfileState()
    state.profiles = profile_set
    state.status = "complete"
    _profile_store["profiles_saved"] = state

    return LoadSavedProfilesResponse(
        profiles_id="profiles_saved",
        profiles=data["profiles"],
        source=source,
    )


# ── DELETE /api/profiles/saved ────────────────────────────────────────────────

@app.delete("/api/profiles/saved")
async def delete_saved_profiles() -> dict:
    """
    Delete user-generated profiles, reverting to the locked defaults.
    No-op if no user profiles exist.
    """
    if _PROFILES_PATH.exists():
        _PROFILES_PATH.unlink()
    return {"status": "ok"}


# ── POST /api/batch/start ─────────────────────────────────────────────────────

@app.post("/api/batch/start", response_model=BatchStartResponse)
async def start_batch(
    calibration: CalibrationInput,
    profiles_id: str | None = None,
    fast: bool = False,
) -> BatchStartResponse:
    """
    Accept a CalibrationInput, launch profile generation + batch simulation
    in a background thread.  Returns the batch_id immediately.

    Connect to WS /ws/{batch_id} to receive live tick events.
    fast=True suppresses per-tick WebSocket events — runs complete silently,
    then batch_complete fires.  Reduces browser overhead for longer batches.
    """
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    state = _BatchState(
        total_runs=calibration.run.num_runs,
        total_ticks=calibration.run.total_ticks,
    )
    _batches[batch_id] = state

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor,
        _run_batch_thread,
        batch_id,
        calibration,
        profiles_id,
        state,
        loop,
        fast,
    )

    return BatchStartResponse(batch_id=batch_id)


# ── GET /api/batch/list ───────────────────────────────────────────────────────

@app.get("/api/batch/list", response_model=list[str])
async def get_batch_list() -> list[str]:
    """Return saved batch IDs from disk, most-recent first."""
    return list_batches()


# ── GET /api/batch/{batch_id}/status ─────────────────────────────────────────

@app.get("/api/batch/{batch_id}/status", response_model=BatchStatus)
async def get_batch_status(batch_id: str) -> BatchStatus:
    """Return live status of a running or completed batch."""
    state = _batches.get(batch_id)
    if state is None:
        # May be a completed batch that was loaded from disk.
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
    return BatchStatus(
        batch_id=batch_id,
        status=state.status,
        runs_complete=state.runs_complete,
        total_runs=state.total_runs,
        current_tick=state.current_tick,
        error_message=state.error_message,
    )


# ── GET /api/batch/{batch_id}/results ────────────────────────────────────────

@app.get("/api/batch/{batch_id}/results", response_model=BatchResult)
async def get_batch_results(batch_id: str) -> BatchResult:
    """
    Return the full BatchResult for a completed batch.
    Checks in-memory state first, then falls back to disk.
    """
    state = _batches.get(batch_id)
    if state is not None and state.result is not None:
        return state.result

    # Try loading from disk.
    try:
        return load_batch(batch_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Results for batch '{batch_id}' not found.",
        )


# ── POST /api/batch/{batch_id}/pause  &  /resume ─────────────────────────────

@app.post("/api/batch/{batch_id}/pause")
async def pause_batch(batch_id: str) -> dict:
    """Pause a running simulation after the current tick completes."""
    state = _batches.get(batch_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Batch not found.")
    state.pause_event.clear()
    return {"paused": True}


@app.post("/api/batch/{batch_id}/resume")
async def resume_batch(batch_id: str) -> dict:
    """Resume a paused simulation."""
    state = _batches.get(batch_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Batch not found.")
    state.pause_event.set()
    return {"paused": False}


# ── POST /api/batch/{batch_id}/replay/{run_number} ───────────────────────────

@app.post(
    "/api/batch/{batch_id}/replay/{run_number}",
    response_model=ReplayStartResponse,
)
async def replay_run(batch_id: str, run_number: int) -> ReplayStartResponse:
    """
    Re-execute a single run from stored seed data.  Streams via a new
    WebSocket batch_id (replay_{batch_id}_{run_number}).

    The original BatchResult must be available (in-memory or on disk) for the
    seed records.
    """
    # Load original result for seeds.
    state = _batches.get(batch_id)
    original: BatchResult | None = None
    if state and state.result:
        original = state.result
    else:
        try:
            original = load_batch(batch_id)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Original batch '{batch_id}' not found.",
            )

    if run_number < 1 or run_number > original.num_runs:
        raise HTTPException(
            status_code=400,
            detail=f"run_number must be 1–{original.num_runs}.",
        )

    # Locate the seed record for this run.
    seed_record = None
    for rr in original.run_results:
        if rr.run_number == run_number:
            seed_record = rr.seed_record
            break
    if seed_record is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Run {run_number} seed not available "
                "(batch may have used aggregate_only storage)."
            ),
        )

    replay_id = f"replay_{batch_id}_r{run_number}"
    replay_state = _BatchState(total_runs=1, total_ticks=original.total_ticks)
    _batches[replay_id] = replay_state

    # Reconstruct calibration from stored batch metadata is not possible without
    # saving it — for now we load CalibrationInput from disk if present,
    # otherwise raise.  Phase 7 note: save calibration JSON alongside results.
    try:
        calibration = _load_calibration(batch_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail="Calibration not found on disk — replay requires saved calibration.",
        )

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor,
        _run_replay_thread,
        replay_id,
        calibration,
        original,
        seed_record.numpy_seed,
        run_number,
        replay_state,
        loop,
    )

    return ReplayStartResponse(batch_id=replay_id)


# ── WebSocket /ws/{batch_id} ──────────────────────────────────────────────────

@app.websocket("/ws/{batch_id}")
async def ws_batch(websocket: WebSocket, batch_id: str) -> None:
    """
    Stream tick events for a running batch.  Drains the batch state queue
    until a batch_complete or error message arrives, then closes.
    """
    await websocket.accept()

    state = _batches.get(batch_id)
    if state is None:
        await websocket.send_json({"type": "error", "message": f"Batch '{batch_id}' not found."})
        await websocket.close()
        return

    try:
        while True:
            event = await asyncio.wait_for(state.queue.get(), timeout=300.0)
            await websocket.send_json(event)
            if event.get("type") in ("batch_complete", "error"):
                break
    except asyncio.TimeoutError:
        await websocket.send_json({"type": "error", "message": "WebSocket timeout — no events."})
    except WebSocketDisconnect:
        pass  # Client disconnected — simulation continues unaffected.
    finally:
        await websocket.close()


# ── Background thread: full batch ─────────────────────────────────────────────

def _run_profiles_thread(
    profiles_id: str,
    calibration: CalibrationInput,
    state: _ProfileState,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Called in a ThreadPoolExecutor thread.
    Runs all 8 LLM profile generation calls, emitting per-type progress
    events to state.queue.  On completion, emits profiles_complete with
    the full serialised ProfileSet.
    """
    def _push(event: dict) -> None:
        asyncio.run_coroutine_threadsafe(state.queue.put(event), loop)

    def progress_callback(agent_type: str, step: int, total: int) -> None:
        _push({
            "type": "profile_type_complete",
            "agent_type": agent_type,
            "step": step,
            "total": total,
        })

    def profile_status_callback(message: str) -> None:
        _push({"type": "profile_status", "message": message})

    try:
        from backend.llm.profile_generator import generate_profiles
        profiles = generate_profiles(
            calibration,
            progress_callback=progress_callback,
            status_callback=profile_status_callback,
        )
        state.profiles = profiles
        state.status = "complete"
        _save_profiles_to_disk(profiles, calibration)
        _push({
            "type": "profiles_complete",
            "profiles_id": profiles_id,
            "profiles": _profiles_to_dict(profiles),
        })
    except Exception as exc:
        state.status = "error"
        state.error_message = str(exc)
        _push({"type": "error", "message": str(exc)})


def _run_batch_thread(
    batch_id: str,
    calibration: CalibrationInput,
    profiles_id: str | None,
    state: _BatchState,
    loop: asyncio.AbstractEventLoop,
    fast: bool = False,
) -> None:
    """
    Called in a ThreadPoolExecutor thread.
    1. Use pre-generated profiles (if profiles_id provided) or generate fresh.
    2. Run simulation batch with tick_callback → push events to state.queue.
    3. Mark batch complete.
    """

    def _push(event: dict) -> None:
        """Thread-safe push to the asyncio queue."""
        asyncio.run_coroutine_threadsafe(state.queue.put(event), loop)

    def tick_callback(data: dict) -> None:
        """
        Called after each tick of the display run (run 1) by run.py.
        data already contains: type, run, tick, prices (dict), shock_active,
        type_activity (dict).  Relay it directly.
        """
        state.current_tick = data.get("tick", 0)
        _push(data)

    try:
        from backend.llm.profile_generator import generate_profiles

        # Use pre-generated profiles if available; otherwise generate fresh.
        pre_state = _profile_store.get(profiles_id) if profiles_id else None
        if pre_state is not None and pre_state.profiles is not None:
            profiles = pre_state.profiles
        else:
            profiles = generate_profiles(calibration)

        def background_tick_callback(data: dict) -> None:
            _push({
                "type": "run_tick",
                "run": data.get("run", 0),
                "tick": data.get("tick", 0),
                "prices": data.get("prices", {}),
            })

        def run_start_callback(run_number: int, persona_seed: str) -> None:
            _push({
                "type": "run_start",
                "run": run_number,
                "persona_seed": persona_seed,
            })

        def run_complete_callback(run_result) -> None:
            state.runs_complete += 1
            _push({
                "type": "run_complete",
                "run": run_result.run_number,
                "final_prices": _prices_dict(run_result.final_prices),
                "persona_seed": run_result.seed_record.persona_seed,
            })
            # Fire immediately when the last run finishes so the frontend
            # shows feedback during aggregate/sanity/narrative processing.
            if state.runs_complete >= state.total_runs:
                _push({
                    "type": "batch_processing",
                    "message": "All runs complete — running sanity checks…",
                })

        def pre_narrative_callback() -> None:
            _push({
                "type": "batch_processing",
                "message": "Sanity checks complete — generating investment analysis…",
            })

        def narrative_status_callback(message: str) -> None:
            _push({"type": "batch_processing", "message": message})

        engine = BatchEngine(save_results=True, enable_narrative=True)
        result = engine.run_batch(
            calibration=calibration,
            profiles=profiles,
            batch_id=batch_id,
            tick_callback=None if fast else tick_callback,
            background_tick_callback=None if fast else background_tick_callback,
            run_complete_callback=run_complete_callback,
            run_start_callback=run_start_callback,
            pre_narrative_callback=pre_narrative_callback,
            narrative_status_callback=narrative_status_callback,
            pause_event=state.pause_event,
        )
        state.result = result
        state.status = "complete"

        # Save calibration alongside results for replay support.
        _save_calibration(batch_id, calibration)

        _push({
            "type": "batch_complete",
            "sanity_checks": [
                {
                    "check_name": sc.check_name,
                    "passed": sc.passed,
                    "score": sc.score,
                    "strength_label": sc.strength_label,
                    "message": sc.message,
                }
                for sc in result.sanity_checks
            ],
        })

    except Exception as exc:
        state.status = "error"
        state.error_message = str(exc)
        _push({"type": "error", "message": str(exc)})


# ── Background thread: replay ─────────────────────────────────────────────────

def _run_replay_thread(
    replay_id: str,
    calibration: CalibrationInput,
    original: BatchResult,
    numpy_seed: int,
    run_number: int,
    state: _BatchState,
    loop: asyncio.AbstractEventLoop,
) -> None:
    def _push(event: dict) -> None:
        asyncio.run_coroutine_threadsafe(state.queue.put(event), loop)

    def tick_callback(data: dict) -> None:
        state.current_tick = data.get("tick", 0)
        _push(data)

    try:
        # Load saved profiles from disk — avoids a 10–20 min LLM call on every replay.
        profiles = None
        saved_state = _profile_store.get("profiles_saved")
        if saved_state and getattr(saved_state, "profiles", None):
            profiles = saved_state.profiles
        else:
            result = _read_saved_profiles_file()
            if result:
                data, _ = result
                from backend.models.profile import ProfileSet, PersonaProfile, AgentCharacteristics
                personas_dict = {}
                for agent_type, persona_list in data["profiles"].items():
                    personas_dict[agent_type] = [
                        PersonaProfile(
                            agent_type=agent_type,
                            persona_number=p["persona_number"],
                            narrative=p["narrative"],
                            characteristics=AgentCharacteristics(**{
                                k: v for k, v in p["characteristics"].items()
                            }),
                        )
                        for p in persona_list
                    ]
                profiles = ProfileSet(personas=personas_dict)
        if profiles is None:
            # Fallback: generate fresh profiles via LLM.
            from backend.llm.profile_generator import generate_profiles
            profiles = generate_profiles(calibration)

        engine = BatchEngine(save_results=False, enable_narrative=False)
        result = engine.run_single(
            calibration=calibration,
            profiles=profiles,
            numpy_seed=numpy_seed,
            run_number=run_number,
            tick_callback=tick_callback,
        )
        state.status = "complete"
        _push({
            "type": "run_complete",
            "run": run_number,
            "final_prices": _prices_dict(result.final_prices),
            "persona_seed": result.seed_record.persona_seed,
        })
        _push({"type": "batch_complete", "sanity_checks": []})

    except Exception as exc:
        state.status = "error"
        state.error_message = str(exc)
        _push({"type": "error", "message": str(exc)})


# ── Calibration persistence helpers ───────────────────────────────────────────

def _save_calibration(batch_id: str, calibration: CalibrationInput) -> None:
    """Save calibration JSON alongside the batch results for replay support."""
    from backend.output.storage import DEFAULT_DATA_DIR
    batch_dir = DEFAULT_DATA_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "calibration.json").write_text(
        calibration.model_dump_json(indent=2), encoding="utf-8"
    )


def _load_calibration(batch_id: str) -> CalibrationInput:
    """Load calibration from disk. Raises FileNotFoundError if not present."""
    from backend.output.storage import DEFAULT_DATA_DIR
    path = DEFAULT_DATA_DIR / batch_id / "calibration.json"
    if not path.exists():
        raise FileNotFoundError(f"calibration.json not found for batch '{batch_id}'.")
    return CalibrationInput.model_validate_json(path.read_text(encoding="utf-8"))


# ── Utility ───────────────────────────────────────────────────────────────────

# ── Saved profiles helpers ────────────────────────────────────────────────────

_PROFILES_PATH = _pathlib.Path("data/profiles/profiles.json")
_DEFAULT_PROFILES_PATH = _pathlib.Path("data/profiles/profiles_default.json")


def _save_profiles_to_disk(profiles, calibration: CalibrationInput) -> None:
    """Persist a ProfileSet and calibration metadata to the canonical profiles file."""
    import datetime
    _PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Build a brief human-readable scenario summary.
    summary_parts = [f"{calibration.run.total_ticks} days"]
    if calibration.shocks:
        from backend.config.sector_config import SECTORS as _SECTORS
        shock_strs = []
        for shock in calibration.shocks:
            sector_names = [_SECTORS[i] for i in shock.affected_sectors if i < len(_SECTORS)]
            sectors = ", ".join(sector_names) if sector_names else "all sectors"
            shock_strs.append(f"{shock.shock_type} shock on {sectors} (day {shock.onset_tick})")
        summary_parts.append("; ".join(shock_strs))
    else:
        summary_parts.append("no shocks")

    payload = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "scenario_name": calibration.scenario_name,
        "scenario_summary": " · ".join(summary_parts),
        "profiles": _profiles_to_dict(profiles),
    }
    _PROFILES_PATH.write_text(
        __import__("json").dumps(payload, indent=2), encoding="utf-8"
    )


def _load_saved_profiles_meta() -> dict | None:
    """
    Return metadata for the active profile set, or None if neither file exists.
    Tries user-generated profiles first; falls back to locked defaults.
    Includes 'source': 'custom' or 'default'.
    """
    for path, source in ((_PROFILES_PATH, "custom"), (_DEFAULT_PROFILES_PATH, "default")):
        if not path.exists():
            continue
        try:
            data = __import__("json").loads(path.read_text(encoding="utf-8"))
            return {
                "generated_at": data.get("generated_at", ""),
                "scenario_name": data.get("scenario_name", ""),
                "scenario_summary": data.get("scenario_summary", ""),
                "source": source,
            }
        except Exception:
            continue
    return None


def _read_saved_profiles_file() -> tuple[dict, str] | None:
    """
    Return (full profiles file content, source) for the active profile set,
    or None if neither file exists/is readable.
    Tries user-generated profiles first; falls back to locked defaults.
    """
    for path, source in ((_PROFILES_PATH, "custom"), (_DEFAULT_PROFILES_PATH, "default")):
        if not path.exists():
            continue
        try:
            data = __import__("json").loads(path.read_text(encoding="utf-8"))
            return data, source
        except Exception:
            continue
    return None


def _prices_dict(prices: list[float]) -> dict[str, float]:
    """Convert a list of sector prices to a sector-keyed dict."""
    if not prices:
        return {}
    return {sector: round(price, 4) for sector, price in zip(SECTORS, prices)}


def _profiles_to_dict(profiles) -> dict:
    """Serialise a ProfileSet to a plain dict for JSON transmission."""
    result = {}
    for agent_type, persona_list in profiles.personas.items():
        result[agent_type] = [
            {
                "persona_number": p.persona_number,
                "narrative": p.narrative,
                "characteristics": {
                    k: v
                    for k, v in p.characteristics.model_dump().items()
                    if v is not None
                },
            }
            for p in persona_list
        ]
    return result
