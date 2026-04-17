/**
 * api.js — REST + WebSocket client for the EMDOLA backend.
 *
 * All functions return Promises.  WebSocket connections return a { close() } handle.
 */
const api = (() => {
  const BASE = '';  // Same-origin — served by FastAPI

  async function _json(url, options = {}) {
    const res = await fetch(BASE + url, options);
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${body}`);
    }
    return res.json();
  }

  /**
   * GET /api/profiles/saved → { generated_at, scenario_name, scenario_summary }
   * Resolves to null if no saved profiles exist (404).
   */
  async function getSavedProfilesMeta() {
    try {
      return await _json('/api/profiles/saved');
    } catch (e) {
      return null;
    }
  }

  /**
   * POST /api/profiles/saved/load → { profiles_id, profiles }
   * Loads the saved profile set from disk into backend memory.
   */
  async function loadSavedProfiles() {
    return _json('/api/profiles/saved/load', { method: 'POST' });
  }

  /** DELETE /api/profiles/saved  → removes user-generated profiles, reverts to defaults */
  async function resetToDefaultProfiles() {
    return _json('/api/profiles/saved', { method: 'DELETE' });
  }

  /** POST /api/profiles/generate  → { profiles_id } */
  async function generateProfiles(calibration) {
    return _json('/api/profiles/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(calibration),
    });
  }

  /** POST /api/batch/start  → { batch_id } */
  async function startBatch(calibration, profilesId, fast = false) {
    const params = [];
    if (profilesId) params.push(`profiles_id=${encodeURIComponent(profilesId)}`);
    if (fast)       params.push('fast=true');
    const url = '/api/batch/start' + (params.length ? '?' + params.join('&') : '');
    return _json(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(calibration),
    });
  }

  /** GET /api/batch/list  → string[] */
  async function listBatches() {
    return _json('/api/batch/list');
  }

  /** GET /api/batch/{id}/status  → BatchStatus */
  async function getBatchStatus(batchId) {
    return _json(`/api/batch/${batchId}/status`);
  }

  /** GET /api/batch/{id}/results  → BatchResult */
  async function getBatchResults(batchId) {
    return _json(`/api/batch/${batchId}/results`);
  }

  /** POST /api/batch/{id}/replay/{runNo}  → { batch_id } */
  async function replayRun(batchId, runNumber) {
    return _json(`/api/batch/${batchId}/replay/${runNumber}`, { method: 'POST' });
  }

  /**
   * Open a WebSocket to /ws/profiles/{profilesId}.
   * Receives profile_type_complete and profiles_complete events.
   */
  function connectProfilesWS(profilesId, onMessage, onClose) {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${location.host}/ws/profiles/${profilesId}`);
    ws.onmessage = (evt) => {
      try { onMessage(JSON.parse(evt.data)); }
      catch (e) { console.error('WS parse error', e); }
    };
    ws.onclose = () => onClose && onClose();
    ws.onerror = (e) => console.error('WebSocket error', e);
    return { close: () => ws.close() };
  }

  /**
   * Open a WebSocket to /ws/{batchId}.
   * Receives tick, run_complete, batch_complete, error events.
   */
  function connectWS(batchId, onMessage, onClose) {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${location.host}/ws/${batchId}`);
    ws.onmessage = (evt) => {
      try { onMessage(JSON.parse(evt.data)); }
      catch (e) { console.error('WS parse error', e); }
    };
    ws.onclose = () => onClose && onClose();
    ws.onerror = (e) => console.error('WebSocket error', e);
    return { close: () => ws.close() };
  }

  return {
    getSavedProfilesMeta, loadSavedProfiles, resetToDefaultProfiles,
    generateProfiles, startBatch,
    listBatches, getBatchStatus, getBatchResults, replayRun,
    connectProfilesWS, connectWS,
  };
})();
