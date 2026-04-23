/**
 * app.js — view controller, comms bar, and top-level state management.
 *
 * Flow (non-linear — all tabs freely navigable):
 *   Setup (5 tabs: Profiles | Market | Shocks | Agents | Run)
 *     → Runtime → Output
 *
 * Step 1: startProfileGeneration() — POST /api/profiles/generate, stream progress via WS.
 * Step 2: startBatch() — POST /api/batch/start?profiles_id=..., stream ticks via WS.
 */
const app = (() => {
  const _isEmbed = new URLSearchParams(window.location.search).get('embed') === 'true';
  if (_isEmbed) document.body.classList.add('embed-mode');

  let _currentBatchId    = null;
  let _currentProfilesId = null;
  let _currentCalibration = null;
  let _wsHandle = null;
  let _simRunning = false;
  let _displayRunDone = false;
  let _isPaused = false;
  let _runsCompleted = 0;  // count of run_complete events received (not the run number)
  let _batchStartTime = null;
  let _latestTick = 0;     // most recent tick from run 1 (used for ETA)

  // Profile generation timer state.
  let _profileGenStartTime     = null;
  let _profileGenTimerInterval = null;
  let _profileGenTypesComplete = 0;

  function _fmtSecs(ms) {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;
  }

  function _startProfileTimer() {
    _profileGenStartTime     = Date.now();
    _profileGenTypesComplete = 0;
    _profileGenTimerInterval = setInterval(_tickProfileTimer, 1000);
  }

  function _stopProfileTimer() {
    clearInterval(_profileGenTimerInterval);
    _profileGenTimerInterval = null;
    _profileGenStartTime     = null;
  }

  function _tickProfileTimer() {
    if (!_profileGenStartTime) return;
    const elapsed = Date.now() - _profileGenStartTime;
    const done    = _profileGenTypesComplete;
    let sub;
    if (done === 0) {
      sub = `${_fmtSecs(elapsed)} elapsed  ·  This may take 10–20 minutes depending on scenario complexity`;
    } else {
      const avgMs     = elapsed / done;
      const remaining = avgMs * (8 - done);
      sub = `${done} of 8 complete  ·  ${_fmtSecs(elapsed)} elapsed  ·  ~${_fmtSecs(remaining)} remaining`;
    }
    const subEl = document.getElementById('comms-sub');
    if (subEl) subEl.textContent = sub;
  }

  // ── Comms bar ───────────────────────────────────────────────────────────────

  const TYPE_LABELS_FULL = {
    R1:'passive retail', R2:'self-directed retail', R3:'high net worth retail', R4:'speculative retail',
    I1:'asset managers', I2:'hedge funds',          I3:'pension funds',        I4:'sovereign wealth funds',
  };

  function _setComms(type, message, sub) {
    const bar    = document.getElementById('comms-bar');
    const msgEl  = document.getElementById('comms-message');
    const subEl  = document.getElementById('comms-sub');
    const iconEl = document.getElementById('comms-icon');
    const spinEl = document.getElementById('comms-spinner');

    bar.className = `comms-bar ${type}`;
    msgEl.textContent = message;
    subEl.textContent = sub || '';

    const isLoading = type === 'loading';
    spinEl.style.display = isLoading ? 'block' : 'none';
    iconEl.style.display = isLoading ? 'none' : 'block';
  }

  // ── Personas footer rendering ────────────────────────────────────────────────

  function _renderPersonasFooter(savedMeta) {
    const footer = document.getElementById('personas-footer');
    if (!footer) return;

    if (savedMeta) {
      const isDefault = savedMeta.source === 'default';
      const dt = new Date(savedMeta.generated_at);
      const dateStr = dt.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
      const timeStr = dt.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });

      const statusLabel = isDefault ? '✓ Preloaded agents active' : '✓ Custom personas loaded';
      const statusMeta  = isDefault
        ? `Ready to run`
        : `Generated ${dateStr} at ${timeStr} · ${savedMeta.scenario_summary}`;

      const resetBtn = isDefault
        ? ''
        : `<button class="btn-secondary" onclick="app.resetToDefaultProfiles()">Reset to defaults</button>`;

      footer.innerHTML = `
        <div class="personas-footer-inner">
          <div class="personas-footer-status">
            <span class="saved-profiles-label">${statusLabel}</span>
            <span class="saved-profiles-meta">${statusMeta}</span>
          </div>
          <div class="personas-footer-actions">
            <button class="btn-gold" onclick="app.useSavedProfiles()">Use preloaded personas</button>
            <button class="btn-secondary" onclick="app.startProfileGeneration()">Regenerate personas</button>
            ${resetBtn}
          </div>
          <span class="personas-footer-hint">Regenerate fires 8 LLM calls — uses your scenario context to create tailored personas. Defaults are never overwritten.</span>
        </div>`;
    } else {
      footer.innerHTML = `
        <div class="personas-footer-inner personas-footer-empty">
          <button id="personas-generate-btn" class="btn-gold" onclick="app.startProfileGeneration()">Generate agent personas</button>
          <span class="personas-footer-hint">Fires 8 LLM calls — uses your scenario name, description, and shocks to create tailored investor personas.</span>
        </div>`;
    }
  }

  // ── Setup tab comms ─────────────────────────────────────────────────────────

  const _TAB_COMMS = {
    'tab-profiles': [
      'Scenarios & Personas tab — generate LLM-calibrated investor personas.',
      'Set your scenario name and description first, then generate personas. The model uses your scenario context — including any shocks you define — to create realistic personas for each investor type.',
    ],
    'tab-market': [
      'Market tab — set price sensitivity and signal strength.',
      'Configure how strongly each sector responds to buying and selling pressure, and how much background news and sentiment affects the market.',
    ],
    'tab-shocks': [
      'Shocks tab — define events or trends that disrupt the market.',
      'Shocks are included in the context passed to the model when generating personas, so defining them first produces more scenario-relevant personas.',
    ],
    'tab-agents': [
      'Agents tab — adjust the investor population.',
      'Override the number of agents or capital base per investor type. Leave blank to use realistic defaults (30 agents each).',
    ],
    'tab-run': [
      'Run tab — set simulation parameters, then launch.',
      'Choose simulation duration and number of Monte Carlo runs. Each run draws a different combination of agent personas. More runs give more robust distributional results.',
    ],
  };

  function _updateTabComms() {
    const activeTab = document.querySelector('#view-setup .tab-btn.active')?.dataset.tab || 'tab-profiles';
    const [msg, sub] = _TAB_COMMS[activeTab] || _TAB_COMMS['tab-profiles'];
    _setComms('info', msg, sub);
    _updateSetupFooter(activeTab);
    if (activeTab === 'tab-run') _updateRunSummary();
  }

  function _updateRunSummary() {
    const el = document.getElementById('run-summary');
    if (!el) return;

    const scenarioName = document.getElementById('scenario-name')?.value?.trim() || '—';
    const numRuns  = document.getElementById('num-runs')?.value || '?';
    const ticks    = document.getElementById('total-ticks')?.value || '?';
    const shocks   = document.querySelectorAll('.shock-card[id^="shock-card-"]').length;
    const profiles = _currentProfilesId ? 'Loaded' : 'Not generated';
    const profilesClass = _currentProfilesId ? 'ok' : 'warn';
    const profilesWarning = document.getElementById('run-tab-profiles-warning');
    if (profilesWarning) profilesWarning.style.display = _currentProfilesId ? 'none' : 'inline';

    el.innerHTML = `
      <div class="run-summary-item">
        <span class="run-summary-label">Scenario</span>
        <span class="run-summary-value">${scenarioName}</span>
      </div>
      <div class="run-summary-item">
        <span class="run-summary-label">Agent profiles</span>
        <span class="run-summary-value ${profilesClass}">${profiles}</span>
      </div>
      <div class="run-summary-item">
        <span class="run-summary-label">Shocks</span>
        <span class="run-summary-value">${shocks} configured</span>
      </div>
      <div class="run-summary-item">
        <span class="run-summary-label">Runs</span>
        <span class="run-summary-value">${numRuns}</span>
      </div>
      <div class="run-summary-item">
        <span class="run-summary-label">Duration</span>
        <span class="run-summary-value">${ticks} days</span>
      </div>`;
  }

  function _updateSetupFooter(activeTab) {
    const tab = activeTab || document.querySelector('#view-setup .tab-btn.active')?.dataset.tab || 'tab-profiles';
    const isRun = tab === 'tab-run';
    const nextBtn = document.getElementById('next-tab-btn');
    const runBtn  = document.getElementById('run-sim-btn');
    if (nextBtn) nextBtn.style.display = isRun ? 'none' : '';
    if (runBtn)  runBtn.style.display  = isRun ? ''     : 'none';
  }

  function nextTab() {
    const tabs = [...document.querySelectorAll('#view-setup .tab-btn')];
    const active = tabs.findIndex(b => b.classList.contains('active'));
    if (active >= 0 && active < tabs.length - 1) tabs[active + 1].click();
  }

  // ── Nav bar ──────────────────────────────────────────────────────────────────

  function _updateNav(activeId) {
    ['nav-setup', 'nav-runtime', 'nav-output'].forEach(id => {
      document.getElementById(id)?.classList.toggle('active', id === activeId);
    });
  }

  function _setNavAvailable(id, available) {
    const el = document.getElementById(id);
    if (el) el.disabled = !available;
  }

  // ── Sim lock (disables setup inputs while simulation is running) ─────────────

  function _setSimLock(locked) {
    _simRunning = locked;
    const banner    = document.getElementById('setup-locked-banner');
    const container = document.querySelector('#view-setup .setup-container');
    if (banner)    banner.style.display = locked ? 'block' : 'none';
    if (container) container.classList.toggle('sim-running', locked);
  }

  // ── View transitions ─────────────────────────────────────────────────────────

  function showLanding() {
    _setView('view-landing');
    document.body.classList.add('landing-mode');
  }

  function enterSetup() {
    document.body.classList.remove('landing-mode');
    showSetup();
  }

  function showSetup() {
    document.body.classList.remove('landing-mode');
    _setView('view-setup');
    _updateNav('nav-setup');
    _updateTabComms();
    if (!_simRunning) {
      const runSimBtn = document.getElementById('run-sim-btn');
      if (runSimBtn) { runSimBtn.disabled = false; runSimBtn.textContent = 'Run simulation →'; }
    }
  }

  function showRuntime() {
    _setView('view-runtime');
    _updateNav('nav-runtime');
    window.parent.postMessage('emdola:run_started', '*');
    // Only update comms if we're not mid-simulation (mid-sim comms are managed by WS events).
    if (!_simRunning) {
      _setComms('info', 'Live view — simulation has completed.', 'Navigate to Results to see the full output.');
    }
  }

  function showOutput() {
    _setView('view-output');
    _updateNav('nav-output');
    _setComms('success', 'Simulation complete.', 'Results and investment briefing shown below.');
    window.parent.postMessage('emdola:results_ready', '*');
  }

  function _setView(id) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(id)?.classList.add('active');
    window.scrollTo(0, 0);
  }

  // ── Use saved profiles (no LLM call) ─────────────────────────────────────────

  async function useSavedProfiles() {
    _setComms('loading', 'Loading personas…', '');

    try {
      const { profiles_id, profiles: profileData, source } = await api.loadSavedProfiles();
      _currentProfilesId = profiles_id;

      profiles.init(profileData);
      const label = source === 'default' ? 'Preloaded personas active' : 'Custom personas loaded';
      _setComms('success',
        `${label} — 24 personas ready.`,
        'Review them in the Personas tab, configure the remaining tabs as needed, then click "Run simulation →".'
      );
    } catch (err) {
      _setComms('error', 'Could not load personas: ' + err.message, '');
    }
  }

  async function resetToDefaultProfiles() {
    _setComms('loading', 'Resetting to default personas…', '');
    try {
      await api.resetToDefaultProfiles();
      const savedMeta = await api.getSavedProfilesMeta();
      _renderPersonasFooter(savedMeta);
      _setComms('info',
        'Reset to default personas.',
        'Default validated agents are active. Regenerate at any time to create scenario-specific personas.'
      );
    } catch (err) {
      _setComms('error', 'Could not reset personas: ' + err.message, '');
    }
  }

  // ── Step 1: generate agent profiles ─────────────────────────────────────────

  async function _revertToDefaultsAfterError(errorMsg) {
    try {
      const { profiles_id, profiles: profileData, source } = await api.loadSavedProfiles();
      _currentProfilesId = profiles_id;
      profiles.init(profileData);
      const savedMeta = await api.getSavedProfilesMeta();
      _renderPersonasFooter(savedMeta);
      _setComms('error',
        'Persona generation failed — reverted to default personas.',
        errorMsg + ' Default validated agents are loaded and ready to use.'
      );
    } catch (_) {
      _setComms('error', 'Profile generation failed.', errorMsg);
    }
  }

  async function startProfileGeneration() {
    const calibration = setup.buildCalibration();

    const genBtn = document.getElementById('personas-generate-btn');
    if (genBtn) { genBtn.disabled = true; genBtn.textContent = 'Starting…'; }

    _setComms('loading', 'Connecting to the model…', 'Preparing to generate agent personas');

    try {
      const { profiles_id } = await api.generateProfiles(calibration);
      _currentProfilesId = profiles_id;

      _setComms('loading', 'Generating agent personas…', '');
      _startProfileTimer();

      _wsHandle = api.connectProfilesWS(profiles_id, _onProfilesWsMessage, _onProfilesWsClose);
    } catch (err) {
      if (genBtn) { genBtn.disabled = false; genBtn.textContent = 'Generate agent personas →'; }
      await _revertToDefaultsAfterError('Check that the server is running and the API key is configured.');
    }
  }

  function _onProfilesWsMessage(data) {
    switch (data.type) {
      case 'profile_type_complete': {
        _profileGenTypesComplete = data.step;
        const label = TYPE_LABELS_FULL[data.agent_type] || data.agent_type;
        const next  = data.step < data.total ? _nextTypeLabel(data.agent_type) : null;
        const primary = `Generating personas — ${data.step} of ${data.total} complete · Finished: ${label}${next ? ` · Up next: ${next}` : ' · Finalising…'}`;
        const msgEl = document.getElementById('comms-message');
        if (msgEl) msgEl.textContent = primary;
        break;
      }

      case 'profile_status': {
        const msgEl = document.getElementById('comms-message');
        if (msgEl) msgEl.textContent = data.message;
        break;
      }

      case 'profiles_complete':
        _stopProfileTimer();
        if (_wsHandle) { _wsHandle.close(); _wsHandle = null; }
        profiles.init(data.profiles);
        _setComms('success',
          'All 24 agent personas generated and saved.',
          'Review them below, configure the remaining tabs as needed, then click "Run simulation →".'
        );
        break;

      case 'error':
        _stopProfileTimer();
        if (_wsHandle) { _wsHandle.close(); _wsHandle = null; }
        _revertToDefaultsAfterError('The API call failed — this may indicate insufficient API credits or a configuration issue.');
        break;

      default:
        console.log('Unknown profiles WS event:', data);
    }
  }

  function _onProfilesWsClose() {
    // Natural close after profiles_complete is handled above.
  }

  function _nextTypeLabel(currentType) {
    const order = ['R1','R2','R3','R4','I1','I2','I3','I4'];
    const idx = order.indexOf(currentType);
    const next = order[idx + 1];
    return next ? TYPE_LABELS_FULL[next] : null;
  }

  // ── Step 2: run simulation ──────────────────────────────────────────────────

  async function startBatch() {
    if (!_currentProfilesId) {
      // Switch to Profiles tab and prompt the user.
      document.querySelectorAll('#view-setup .tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('#view-setup .tab-panel').forEach(p => p.classList.remove('active'));
      document.querySelector('[data-tab="tab-profiles"]')?.classList.add('active');
      document.getElementById('tab-profiles')?.classList.add('active');
      _setComms('error',
        'No agent personas loaded.',
        'Generate or load personas in the Personas tab before running the simulation.'
      );
      return;
    }

    // Read calibration fresh — user has just finished configuring the tabs.
    const calibration = setup.buildCalibration();
    _currentCalibration = calibration;

    const runSimBtn = document.getElementById('run-sim-btn');
    if (runSimBtn) { runSimBtn.disabled = true; runSimBtn.textContent = 'Starting simulation…'; }

    _displayRunDone = false;
    _isPaused = false;
    _runsCompleted = 0;
    _batchStartTime = null;
    _latestTick = 0;
    const pauseBtn = document.getElementById('rt-pause-btn');
    if (pauseBtn) { pauseBtn.textContent = '⏸ Pause'; pauseBtn.classList.remove('paused'); pauseBtn.disabled = false; }
    _setComms('loading', 'Starting simulation…', 'Initialising agent population');

    try {
      const fastMode = document.getElementById('fast-mode')?.checked || false;
      const { batch_id } = await api.startBatch(calibration, _currentProfilesId, fastMode);
      _currentBatchId = batch_id;

      _setNavAvailable('nav-runtime', true);
      _setSimLock(true);
      runtime.init(calibration);
      showRuntime();
      _setComms('loading', 'Simulation running…', `Run 1 of ${calibration.run?.num_runs || '?'}`);

      if (_wsHandle) _wsHandle.close();
      _batchStartTime = Date.now();  // start timer when WS connects, before any run_complete
      _wsHandle = api.connectWS(batch_id, _onWsMessage, _onWsClose);
    } catch (err) {
      _setComms('error', 'Could not start simulation: ' + err.message, '');
      if (runSimBtn) { runSimBtn.disabled = false; runSimBtn.textContent = 'Run simulation →'; }
    }
  }

  // ── WebSocket event handler (simulation) ─────────────────────────────────────

  function _onWsMessage(data) {
    switch (data.type) {
      case 'run_start':
        runtime.onRunStart(data);
        break;

      case 'run_tick':
        runtime.addSpaghettiPoint(data.run, data.tick, data.prices);
        break;

      case 'tick':
        runtime.onTick(data);
        _latestTick = data.tick;
        runtime.addSpaghettiPoint(1, data.tick, data.prices);
        if (!_displayRunDone && data.tick >= ((_currentCalibration?.run?.total_ticks || 250) - 1)) {
          _displayRunDone = true;
        }
        // Update comms bar with tick-based ETA (run 1 drives the batch timeline).
        if (data.tick > 5 && _batchStartTime) {
          const totalTicks = _currentCalibration?.run?.total_ticks || 250;
          const totalRuns  = _currentCalibration?.run?.num_runs || 0;
          const elapsedMs  = Date.now() - _batchStartTime;
          const msPerTick  = elapsedMs / data.tick;
          const remainingSec = Math.round(msPerTick * (totalTicks - data.tick) / 1000);
          const etaStr = remainingSec > 90
            ? `~${Math.round(remainingSec / 60)}m remaining`
            : `~${remainingSec}s remaining`;
          _setComms('loading', 'Simulation running…',
            `Run 1 of ${totalRuns} (day ${data.tick}/${totalTicks})  ·  ${etaStr}`);
        }
        break;

      case 'run_complete': {
        _runsCompleted++;
        runtime.onRunComplete(data);
        const totalRuns = _currentCalibration?.run?.num_runs || 0;
        // Update comms if run 1 hasn't started ticking yet (fast/early) OR if run 1 is done
        // (tick-based ETA is stale — show actual run count progress instead).
        if (_latestTick <= 5 || _displayRunDone) {
          const remainingRuns = totalRuns - _runsCompleted;
          let sub = `${_runsCompleted} of ${totalRuns} runs complete`;
          if (remainingRuns > 0 && _batchStartTime) {
            const elapsedMs   = Date.now() - _batchStartTime;
            const avgMsPerRun = elapsedMs / _runsCompleted;
            const etaSec      = Math.round(avgMsPerRun * remainingRuns / 1000);
            const etaStr      = etaSec > 90
              ? `~${Math.round(etaSec / 60)}m remaining`
              : `~${etaSec}s remaining`;
            sub += `  ·  ${etaStr}`;
          }
          _setComms('loading', 'Simulation running…', sub);
        }
        break;
      }

      case 'batch_processing':
        _setComms('loading', data.message || 'Processing…',
          data.message?.includes('analysis')
            ? 'Generating investment narrative from simulation output — this takes 1–3 minutes.'
            : 'Aggregating results across all runs.');
        break;

      case 'batch_complete': {
        _setSimLock(false);
        _isPaused = false;
        const pb = document.getElementById('rt-pause-btn');
        if (pb) { pb.textContent = '⏸ Pause'; pb.classList.remove('paused'); pb.disabled = true; }
        _setComms('loading', 'Analysis complete — loading results…', '');
        _fetchAndShowResults(_currentBatchId);
        break;
      }

      case 'error':
        _setSimLock(false);
        console.error('Simulation error:', data.message);
        _setComms('error', 'Simulation error: ' + data.message, 'Return to setup and try again.');
        showSetup();
        break;

      default:
        console.log('Unknown WS event:', data);
    }
  }

  function _onWsClose() {
    if (!document.getElementById('view-runtime')?.classList.contains('active')) return;
    if (!_currentBatchId) return;
    const batchId = _currentBatchId;
    setTimeout(async () => {
      try {
        const status = await api.getBatchStatus(batchId);
        if (status.status === 'complete') {
          _fetchAndShowResults(batchId);
        } else {
          _setComms('error',
            'Connection lost — simulation may still be running.',
            'Refresh the page to start a new run.');
        }
      } catch (_) {
        _setComms('error',
          'Connection lost — simulation may still be running.',
          'Refresh the page to start a new run.');
      }
    }, 1000);
  }

  async function _fetchAndShowResults(batchId) {
    try {
      const result = await api.getBatchResults(batchId);
      output.render(batchId, result);
      _setNavAvailable('nav-output', true);
      showOutput();
    } catch (err) {
      console.error('Failed to fetch results:', err);
      _setComms('error', 'Could not load results: ' + err.message, '');
    }
  }

  // ── Replay ───────────────────────────────────────────────────────────────────

  async function replayRun(runNumber) {
    if (!_currentBatchId) return;

    _setComms('loading', `Replaying run ${runNumber}…`, '');

    try {
      const { batch_id: replayId } = await api.replayRun(_currentBatchId, runNumber);
      showRuntime();
      runtime.init(_currentCalibration || { scenario_name: 'Replay', run: {} });

      if (_wsHandle) _wsHandle.close();
      _wsHandle = api.connectWS(replayId, _onWsMessage, _onWsClose);
    } catch (err) {
      _setComms('error', 'Replay error: ' + err.message, '');
    }
  }

  // ── Init ─────────────────────────────────────────────────────────────────────

  async function init() {
    lucide.createIcons();
    setup.init();

    // Update comms whenever the user switches calibration tabs.
    document.querySelectorAll('#view-setup .tab-btn').forEach(btn => {
      btn.addEventListener('click', _updateTabComms);
    });

    _updateSetupFooter();
    showLanding();

    const savedMeta = await api.getSavedProfilesMeta();
    _renderPersonasFooter(savedMeta);
    if (_isEmbed && savedMeta) useSavedProfiles();

    // Pre-populate comms for when user enters setup.
    if (savedMeta) {
      const isDefault = savedMeta.source === 'default';
      _setComms('info',
        isDefault
          ? 'Preloaded agents active — ready to run.'
          : 'Custom personas available from your last scenario.',
        isDefault
          ? 'Select a preset, configure the remaining tabs, then click "Run simulation →". Regenerate at any time for scenario-specific personas.'
          : `"${savedMeta.scenario_name}" — use them or regenerate after updating your scenario. Configure the remaining tabs, then click "Run simulation →".`
      );
    } else {
      _setComms('info',
        'Configure your scenario, generate agent personas, then run the simulation.',
        'Use the 5 tabs above. You can set calibration parameters before or after generating personas — the model uses your scenario context when creating personas.'
      );
    }
  }

  async function togglePause() {
    if (!_currentBatchId || !_simRunning) return;
    const btn = document.getElementById('rt-pause-btn');
    try {
      const action = _isPaused ? 'resume' : 'pause';
      await fetch(`/api/batch/${_currentBatchId}/${action}`, { method: 'POST' });
      _isPaused = !_isPaused;
      if (btn) {
        btn.textContent = _isPaused ? '▶ Resume' : '⏸ Pause';
        btn.classList.toggle('paused', _isPaused);
      }
    } catch (e) {
      console.error('Pause/resume failed:', e);
    }
  }

  document.addEventListener('DOMContentLoaded', init);

  function toggleCalibration(detailId, btn) {
    const el = document.getElementById(detailId);
    const open = el.style.display === 'none';
    el.style.display = open ? 'block' : 'none';
    btn.textContent = open ? 'Close ▴' : 'Customise ▾';
  }

  return { startProfileGeneration, useSavedProfiles, resetToDefaultProfiles, startBatch, replayRun, nextTab, showLanding, enterSetup, showSetup, showRuntime, showOutput, togglePause, toggleCalibration };
})();
