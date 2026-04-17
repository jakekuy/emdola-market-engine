/**
 * charts.js — Plotly.js charts for the output view.
 *
 * Spaghetti plot: all run price trajectories per sector (with rolling median).
 * Mispricing scatter: final mispricing per sector per run (strip plot).
 */
const charts = (() => {
  const SECTORS = ['En','Mat','Ind','CDi','Sta','HC','Fin','Tech','CS','Uti','RE'];
  // Full names as sent by the backend (must match sector_config.py order).
  const SECTOR_FULLNAMES = [
    'Energy','Materials','Industrials','Consumer Discretionary','Consumer Staples',
    'Health Care','Financials','Information Technology','Communication Services',
    'Utilities','Real Estate',
  ];
  const SECTOR_NAMES = {
    En:'Energy', Mat:'Materials', Ind:'Industrials', CDi:'Cons. Discret.',
    Sta:'Cons. Staples', HC:'Health Care', Fin:'Financials',
    Tech:'Info. Tech.', CS:'Comm. Services', Uti:'Utilities', RE:'Real Estate',
  };
  // GICS sector colours — red-to-gold spectrum matching CSS variables.
  const SECTOR_COLORS = [
    '#943040','#b83030','#c84028','#d06018','#c87010',
    '#b87018','#c89010','#d4a80e','#d4b020','#c8a828','#e0c030',
  ];
  const PLOTLY_LAYOUT_BASE = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: '#131316',
    font: { color: '#807c76', size: 11 },
    margin: { t: 20, r: 20, b: 40, l: 50 },
    xaxis: { gridcolor: '#1a1a1e', linecolor: '#2c2c34', tickcolor: '#2c2c34' },
    yaxis: { gridcolor: '#1a1a1e', linecolor: '#2c2c34', tickcolor: '#2c2c34' },
  };

  let _currentSector = 'Mkt';

  /** Initialise sector pill selector. */
  function initSectorSelector() {
    const container = document.getElementById('chart-sector-selector');
    const pills = ['Mkt', ...SECTORS];
    container.innerHTML = pills.map(s =>
      `<div class="sector-pill ${s === _currentSector ? 'active' : ''}"
            data-sector="${s}"
            onclick="charts.selectSector('${s}')">${s === 'Mkt' ? 'All sectors (avg)' : s}</div>`
    ).join('');
  }

  function selectSector(sector) {
    _currentSector = sector;
    document.querySelectorAll('.sector-pill').forEach(p => {
      p.classList.toggle('active', p.dataset.sector === sector);
    });
    if (_lastResult) renderSpaghetti(_lastResult);
  }

  let _lastResult = null;

  /** Render all charts from a BatchResult. */
  function render(result) {
    _lastResult = result;
    initSectorSelector();
    renderSpaghetti(result);
    renderMispricing(result);
  }

  /** Spaghetti plot: per-run price trajectories for the selected sector. */
  function renderSpaghetti(result) {
    const sector = _currentSector;
    const stats = result.aggregated_stats;
    if (!stats) return;

    const sectorFullName = sector === 'Mkt' ? SECTOR_FULLNAMES[0] : SECTOR_FULLNAMES[SECTORS.indexOf(sector)];
    const meanData = stats.mean_prices[sectorFullName] || [];

    // Use actual tick numbers from run snapshots so the mean line aligns with per-run traces.
    const refTicks = result.run_results?.[0]?.tick_snapshots?.map(s => s.tick);
    const ticks = (refTicks && refTicks.length === meanData.length)
      ? refTicks
      : meanData.map((_, i) => i);

    const traces = [];

    // Per-run trajectories.
    result.run_results.forEach((run, ri) => {
      const prices = run.tick_snapshots.map(s => {
        if (sector === 'Mkt') {
          const vals = s.prices.filter(p => p != null);
          return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
        }
        const idx = SECTORS.indexOf(sector);
        return idx >= 0 ? s.prices[idx] : null;
      }).filter(p => p !== null);
      if (prices.length === 0) return;

      const seed = run.seed_record?.numpy_seed ?? '?';
      const avgMisp = run.final_mispricing
        ? (run.final_mispricing.reduce((a, b) => a + Math.abs(b), 0) / run.final_mispricing.length * 100).toFixed(1)
        : '?';

      traces.push({
        x: run.tick_snapshots.map(s => s.tick),
        y: prices,
        type: 'scatter',
        mode: 'lines',
        line: { color: 'rgba(212,168,14,0.20)', width: 1.5 },
        showlegend: false,
        hovertemplate: `<b>Run ${run.run_number}</b><br>Seed: ${seed}<br>Avg mispricing: ±${avgMisp}%<br>Day %{x}<extra></extra>`,
        name: `Run ${run.run_number}`,
      });
    });

    // Mean line (bright).
    const meanPrices = sector === 'Mkt'
      ? (stats.mean_prices[SECTOR_FULLNAMES[0]] || []).map((_, i) =>
          SECTOR_FULLNAMES.reduce((sum, fn) => sum + (stats.mean_prices[fn]?.[i] ?? 100), 0) / SECTOR_FULLNAMES.length
        )
      : meanData;
    if (meanPrices.length > 0) {
      traces.push({
        x: ticks,
        y: meanPrices,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#d4a80e', width: 2 },
        name: sector === 'Mkt' ? 'All sectors (avg)' : 'Mean',
        showlegend: true,
        hovertemplate: `<b>Mean</b><br>Day %{x}<br>Index: %{y:.2f}<extra></extra>`,
      });
    }

    // P10–P90 band.
    const p10 = sector === 'Mkt'
      ? (stats.p10_prices[SECTOR_FULLNAMES[0]] || []).map((_, i) =>
          SECTOR_FULLNAMES.reduce((sum, fn) => sum + (stats.p10_prices[fn]?.[i] ?? 100), 0) / SECTOR_FULLNAMES.length)
      : (stats.p10_prices[sectorFullName] || []);
    const p90 = sector === 'Mkt'
      ? (stats.p90_prices[SECTOR_FULLNAMES[0]] || []).map((_, i) =>
          SECTOR_FULLNAMES.reduce((sum, fn) => sum + (stats.p90_prices[fn]?.[i] ?? 100), 0) / SECTOR_FULLNAMES.length)
      : (stats.p90_prices[sectorFullName] || []);
    if (p10.length > 0 && p90.length > 0) {
      traces.push({
        x: [...ticks, ...ticks.slice().reverse()],
        y: [...p90, ...p10.slice().reverse()],
        type: 'scatter',
        fill: 'toself',
        fillcolor: 'rgba(212,168,14,0.15)',
        line: { color: 'rgba(0,0,0,0)', width: 0 },
        showlegend: true,
        hoverinfo: 'skip',
        name: 'P10–P90 range',
      });
    }

    // P0 = 100 reference line.
    if (ticks.length > 0) {
      traces.push({
        x: [ticks[0], ticks[ticks.length - 1]],
        y: [100, 100],
        type: 'scatter',
        mode: 'lines',
        line: { color: 'rgba(255,255,255,0.2)', dash: 'dot', width: 1 },
        showlegend: false,
        hoverinfo: 'skip',
      });
    }

    const layout = {
      ...PLOTLY_LAYOUT_BASE,
      yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'Index (base 100)' },
      xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: 'Day' },
      legend: { font: { color: '#807c76', size: 11 }, bgcolor: 'transparent' },
      hovermode: 'closest',
      height: 280,
    };

    Plotly.newPlot('chart-spaghetti', traces, layout, { responsive: true, displayModeBar: false });
  }

  /** Mispricing box plot: distribution of final mispricing per sector across all runs. */
  function renderMispricing(result) {
    const stats = result.aggregated_stats;
    if (!stats) return;

    const runs = result.run_results || [];
    const numRuns = runs.length;
    const traces = [];

    // Sectors that received a direct shock — used to mark labels.
    const shockedSet = new Set(result.shocked_sectors || []);

    if (numRuns === 0) {
      // Fallback: mean mispricing from aggregated stats only (no per-run breakdown).
      const means = SECTOR_FULLNAMES.map(fn => {
        const traj = stats.mean_mispricing[fn] || [];
        return traj.length > 0 ? traj[traj.length - 1] * 100 : 0;
      });
      traces.push({
        x: SECTORS.map((s, i) => shockedSet.has(SECTOR_FULLNAMES[i]) ? s + ' ●' : s),
        y: means,
        type: 'bar',
        marker: { color: SECTOR_COLORS },
        hovertemplate: '%{x}<br>%{y:+.1f}%<extra></extra>',
      });
    } else {
      // Box plot — one box per sector; individual points visible up to 100 runs.
      const showPoints = numRuns <= 100 ? 'all' : 'outliers';
      SECTOR_FULLNAMES.forEach((fn, si) => {
        const isShocked = shockedSet.has(fn);
        const vals = runs
          .filter(r => r.final_mispricing)
          .map(r => (r.final_mispricing[si] ?? 0) * 100);
        traces.push({
          y: vals,
          type: 'box',
          name: isShocked ? SECTORS[si] + ' ●' : SECTORS[si],
          boxpoints: showPoints,
          jitter: 0.35,
          pointpos: 0,
          marker: { color: SECTOR_COLORS[si], size: 5, opacity: isShocked ? 0.8 : 0.45 },
          line: { color: SECTOR_COLORS[si], width: isShocked ? 1.5 : 0.8 },
          fillcolor: SECTOR_COLORS[si] + (isShocked ? '33' : '14'),
          hovertemplate: `<b>${fn}</b>${isShocked ? ' — directly shocked' : ' — coupling / emergent'}<br>%{y:+.1f}%<extra></extra>`,
          showlegend: false,
        });
      });
    }

    const layout = {
      ...PLOTLY_LAYOUT_BASE,
      xaxis: {
        ...PLOTLY_LAYOUT_BASE.xaxis,
        title: '',
        tickangle: -45,
      },
      yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'Final mispricing (%)' },
      shapes: [{
        type: 'line',
        xref: 'paper', x0: 0, x1: 1,
        y0: 0, y1: 0,
        line: { color: 'rgba(255,255,255,0.25)', dash: 'dot', width: 1 },
      }],
      hovermode: 'closest',
      showlegend: false,
      height: 220,
    };

    Plotly.newPlot('chart-mispricing', traces, layout, { responsive: true, displayModeBar: false });
  }

  return { render, selectSector, initSectorSelector };
})();
