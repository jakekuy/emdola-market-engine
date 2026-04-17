/**
 * runtime.js — runtime view layout controller.
 *
 * Manages the ticker row, agent type cards, and progress indicators.
 * Called by app.js on each WebSocket tick event.
 */
const runtime = (() => {
  const SECTORS = ['En','Mat','Ind','CDi','Sta','HC','Fin','Tech','CS','Uti','RE'];
  // Full sector names as sent by the backend — must match sector_config.py order.
  const SECTOR_KEYS = ['Energy','Materials','Industrials','Consumer Discretionary','Consumer Staples','Health Care','Financials','Information Technology','Communication Services','Utilities','Real Estate'];
  const AGENT_TYPES = ['R1','R2','R3','R4','I1','I2','I3','I4'];
  const TYPE_LABELS = {
    R1:'Passive retail', R2:'Self-directed retail', R3:'HNW retail', R4:'Speculative retail',
    I1:'Asset manager',  I2:'Hedge fund',           I3:'Pension fund', I4:'Sovereign wealth',
  };

  /** Parse "R1n4a0b0c4R2n64a12b2c50..." → {R1:{a:0,b:0,c:4}, R2:{a:12,b:2,c:50}, ...} */
  function _parsePersonaSeed(seed) {
    const dist = {};
    const re = /([RI]\d)n\d+a(\d+)b(\d+)c(\d+)/g;
    let m;
    while ((m = re.exec(seed)) !== null) {
      dist[m[1]] = { a: parseInt(m[2]), b: parseInt(m[3]), c: parseInt(m[4]) };
    }
    return dist;
  }

  let prevPrices = {};
  let totalTicks = 250;
  let totalRuns = 10;
  let ticksCompleted = 0;
  let runsCompleted = 0;
  let _personaSeed = null;   // persona_seed string from most recent run_complete
  let _displayRun = 0;       // run number of most recent run_complete

  /** Initialise runtime view for a new batch. */
  function init(calibration) {
    totalTicks = calibration.run?.total_ticks || 250;
    totalRuns  = calibration.run?.num_runs || 10;
    ticksCompleted = 0;
    runsCompleted = 0;
    prevPrices = {};

    document.getElementById('rt-scenario-name').textContent = calibration.scenario_name || '';
    document.getElementById('rt-progress').textContent = `Run 0 / ${totalRuns}`;
    document.getElementById('rt-tick').textContent = `0 / ${totalTicks}`;
    document.getElementById('rt-progress-bar').style.width = '0%';
    document.getElementById('rt-shock-status').textContent = '—';

    _buildTickerRow();
    network.init('network-container');
    _initLiveChart();
  }

  function _buildTickerRow() {
    const row = document.getElementById('ticker-row');
    row.innerHTML = SECTORS.map(s =>
      `<div class="ticker-item" id="ticker-${s}">
         <span class="ticker-sector">${s}</span>
         <span class="ticker-price mono" id="price-${s}">100.00</span>
       </div>`
    ).join('');
  }

  function _buildAgentCards() {
    const cards = document.getElementById('agent-cards');
    cards.innerHTML = AGENT_TYPES.map(type =>
      `<div class="agent-card" data-type="${type}" onclick="runtime.highlightType('${type}')" title="Click to view archetype detail for this run">
         <div class="agent-card-header">
           <span class="agent-card-type" style="color:var(--${type.toLowerCase()})">${type}</span>
           <span class="agent-card-click-hint">view archetypes →</span>
         </div>
         <div class="agent-card-label">${TYPE_LABELS[type]}</div>
         <div class="agent-card-stat" id="card-stat-${type}">—</div>
       </div>`
    ).join('');
  }

  /** Handle a tick event from the WebSocket. */
  function onTick(data) {
    const prices = data.prices || {};
    const shockActive = data.shock_active || false;
    ticksCompleted = (data.tick || 0) + 1;  // 1-indexed: tick 249 shows as 250/250

    // Update ticker row.
    SECTORS.forEach((s, i) => {
      const priceEl = document.getElementById(`price-${s}`);
      if (!priceEl) return;
      const price = prices[SECTOR_KEYS[i]];
      if (price === undefined) return;
      const prev = prevPrices[s] || 100;
      priceEl.textContent = price.toFixed(2);
      priceEl.className = 'ticker-price mono' + (price > prev ? ' up' : price < prev ? ' down' : '');
      prevPrices[s] = price;
    });

    // Update progress.
    document.getElementById('rt-tick').textContent = `${ticksCompleted} / ${totalTicks}`;
    const pct = (runsCompleted / totalRuns * 90 + (ticksCompleted / totalTicks * 10)).toFixed(1);
    document.getElementById('rt-progress-bar').style.width = `${Math.min(pct, 100)}%`;

    // Shock status.
    document.getElementById('rt-shock-status').textContent = shockActive ? '⚡ Active' : '—';

    // Update agent cards.
    const activity = data.type_activity || {};
    AGENT_TYPES.forEach(type => {
      const a = activity[type];
      const el = document.getElementById(`card-stat-${type}`);
      if (el && a) {
        const ar = ((a.activity_ratio || 0) * 100).toFixed(0);
        el.innerHTML = `${ar}% active this tick<br><span style="color:var(--text-2);font-size:10px">↑ ${a.buy||0} buying &nbsp;·&nbsp; ↓ ${a.sell||0} selling</span>`;
      }
    });

    // Update network visualisation.
    network.update(data);
  }

  /** Handle a run_start event — fires before the first tick, gives early persona_seed. */
  function onRunStart(data) {
    if (data.persona_seed) {
      _personaSeed = data.persona_seed;
      _displayRun = data.run || 1;
    }
  }

  /** Handle a run_complete event. */
  function onRunComplete(data) {
    runsCompleted++;
    document.getElementById('rt-progress').textContent = `Run ${runsCompleted} / ${totalRuns}`;
    document.getElementById('rt-tick').textContent = `${totalTicks} / ${totalTicks}`;
    const pct = ((runsCompleted / totalRuns) * 100).toFixed(1);
    document.getElementById('rt-progress-bar').style.width = `${Math.min(pct, 100)}%`;
    if (data.persona_seed) {
      _personaSeed = data.persona_seed;
      _displayRun = data.run || runsCompleted;
    }
  }

  /**
   * Called when user clicks an agent card or network node.
   * If a run has completed and profiles are loaded, opens a popup showing
   * the specific persona that was active for that agent type in this run.
   */
  function highlightType(type) {
    if (typeof profiles === 'undefined') return;
    if (_personaSeed) {
      const dist = _parsePersonaSeed(_personaSeed);
      if (dist[type]) {
        profiles.openModalWithDistribution(type, dist[type], `Run ${_displayRun}`);
        return;
      }
    }
    profiles.openModal(type);
  }

  // ── Live spaghetti chart ──────────────────────────────────────────────────

  // Sector pills: 'Mkt' = equal-weighted market average, then individual GICS.
  const CHART_SECTOR_PILLS = ['Mkt','En','Mat','Ind','CDi','Sta','HC','Fin','Tech','CS','Uti','RE'];

  let _chartSector     = 'Mkt';
  let _spaghettiData   = {};   // { runNum: [{tick, prices}, ...] }
  let _chartInitialized = false;
  let _chartUpdatePending = false;
  let _chartG          = null;
  let _chartLinesG     = null;
  let _chartXScale     = null;
  let _chartYScale     = null;
  let _chartLine       = null;
  let _chartInnerW     = 0;
  let _chartInnerH     = 0;

  function _initLiveChart() {
    _spaghettiData    = {};
    _chartSector      = 'Mkt';
    _chartInitialized = false;

    const toggle = document.getElementById('live-chart-toggle');
    if (toggle) {
      toggle.innerHTML = CHART_SECTOR_PILLS.map(s =>
        `<span class="live-sector-pill${s === _chartSector ? ' active' : ''}"
               data-sector="${s}"
               onclick="runtime.selectChartSector('${s}')">${s === 'Mkt' ? 'All sectors (avg)' : s}</span>`
      ).join('');
    }

    requestAnimationFrame(() => {
      const container = document.getElementById('live-chart-container');
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const W = rect.width  || 500;
      const H = rect.height || 160;
      const m = { top: 8, right: 16, bottom: 26, left: 42 };
      _chartInnerW = W - m.left - m.right;
      _chartInnerH = H - m.top  - m.bottom;

      const svgSel = d3.select('#live-chart-svg').attr('width', W).attr('height', H);
      svgSel.selectAll('*').remove();

      _chartG = svgSel.append('g').attr('transform', `translate(${m.left},${m.top})`);

      _chartG.append('rect')
        .attr('width', _chartInnerW).attr('height', _chartInnerH)
        .attr('fill', '#0c0c0e');

      _chartXScale = d3.scaleLinear().domain([0, totalTicks]).range([0, _chartInnerW]);
      _chartYScale = d3.scaleLinear().domain([99, 101]).range([_chartInnerH, 0]);

      // Baseline at 100 (start price).
      _chartG.append('line').attr('class', 'live-baseline')
        .attr('x1', 0).attr('x2', _chartInnerW)
        .attr('y1', _chartYScale(100)).attr('y2', _chartYScale(100))
        .attr('stroke', 'rgba(255,255,255,0.1)').attr('stroke-dasharray', '4,3');

      _chartG.append('g').attr('class', 'live-x-axis')
        .attr('transform', `translate(0,${_chartInnerH})`)
        .call(d3.axisBottom(_chartXScale).ticks(5).tickFormat(d => `${d}`));

      _chartG.append('g').attr('class', 'live-y-axis')
        .call(d3.axisLeft(_chartYScale).ticks(4)
          .tickSize(-_chartInnerW)
          .tickFormat(d => d.toFixed(1)));

      _chartLinesG = _chartG.insert('g', '.live-x-axis').attr('class', 'live-lines');

      _chartLine = d3.line()
        .defined(d => d.value != null)
        .x(d => _chartXScale(d.tick))
        .y(d => _chartYScale(d.value))
        .curve(d3.curveMonotoneX);

      _chartInitialized = true;
    });
  }

  function _getChartValue(pricesDict, sector) {
    if (sector === 'Mkt') {
      const vals = SECTOR_KEYS.map(k => pricesDict[k]);
      const valid = vals.filter(v => v != null);
      if (!valid.length) return 100;
      return valid.reduce((a, b) => a + b, 0) / valid.length;
    }
    const idx = SECTORS.indexOf(sector);
    return idx >= 0 ? (pricesDict[SECTOR_KEYS[idx]] ?? 100) : 100;
  }

  function _scheduleChartUpdate() {
    if (!_chartUpdatePending) {
      _chartUpdatePending = true;
      requestAnimationFrame(() => {
        _chartUpdatePending = false;
        _updateLiveChart();
      });
    }
  }

  function _updateLiveChart() {
    if (!_chartInitialized || !_chartLinesG) return;

    const runNums = Object.keys(_spaghettiData).map(Number).sort((a, b) => a - b);
    if (!runNums.length) return;

    // Build display points and compute Y extent.
    let yMin = Infinity, yMax = -Infinity;
    const displayData = {};
    runNums.forEach(rn => {
      const pts = (_spaghettiData[rn] || []).map(pt => ({
        tick: pt.tick,
        value: _getChartValue(pt.prices, _chartSector),
      }));
      displayData[rn] = pts;
      pts.forEach(p => {
        if (p.value < yMin) yMin = p.value;
        if (p.value > yMax) yMax = p.value;
      });
    });

    // Update Y scale with padding.
    const pad = Math.max(0.3, (yMax - yMin) * 0.15);
    _chartYScale.domain([Math.min(yMin - pad, 99.5), Math.max(yMax + pad, 100.5)]);
    _chartG.select('.live-y-axis')
      .call(d3.axisLeft(_chartYScale).ticks(4)
        .tickSize(-_chartInnerW)
        .tickFormat(d => d.toFixed(1)));
    _chartG.select('.live-baseline')
      .attr('y1', _chartYScale(100)).attr('y2', _chartYScale(100));

    // Draw/update one path per run.
    runNums.forEach(rn => {
      const pts = displayData[rn];
      if (!pts || pts.length < 2) return;
      const isRun1 = rn === 1;
      const id = `ll-${rn}`;
      let path = _chartLinesG.select(`#${id}`);
      if (path.empty()) {
        path = _chartLinesG.append('path').attr('id', id).attr('fill', 'none');
        if (isRun1) path.raise();
      }
      path
        .attr('stroke',       isRun1 ? '#d4a80e' : 'rgba(255,255,255,0.22)')
        .attr('stroke-width', isRun1 ? 1.8 : 0.75)
        .attr('d', _chartLine(pts));
    });

    // Keep run 1 on top.
    _chartLinesG.select('#ll-1').raise();
  }

  /** Called by app.js for each price data point (run 1 via 'tick', others via 'run_tick'). */
  function addSpaghettiPoint(runNumber, tick, pricesDict) {
    if (!_spaghettiData[runNumber]) _spaghettiData[runNumber] = [];
    _spaghettiData[runNumber].push({ tick, prices: pricesDict });
    _scheduleChartUpdate();
  }

  /** Switch displayed sector; redraws immediately from stored data. */
  function selectChartSector(sector) {
    _chartSector = sector;
    document.querySelectorAll('.live-sector-pill').forEach(p => {
      p.classList.toggle('active', p.dataset.sector === sector);
    });
    _updateLiveChart();
  }

  /** Update just the day counter — called even when live view is skipped. */
  function tickProgress(tick) {
    ticksCompleted = tick;
    const el = document.getElementById('rt-tick');
    if (el) el.textContent = `${tick} / ${totalTicks}`;
    const pct = (runsCompleted / totalRuns * 90 + (tick / totalTicks * 10)).toFixed(1);
    const bar = document.getElementById('rt-progress-bar');
    if (bar) bar.style.width = `${Math.min(pct, 100)}%`;
  }

  return { init, onRunStart, onTick, onRunComplete, highlightType, addSpaghettiPoint, selectChartSector, tickProgress };
})();
