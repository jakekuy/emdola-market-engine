/**
 * output.js — results view: narrative, sanity checks, type activity, replay.
 */
const output = (() => {
  const AGENT_TYPES = ['R1','R2','R3','R4','I1','I2','I3','I4'];
  const SANITY_CHECK_LABELS = {
    fundamentalist_reversion:   'Mean reversion',
    herding_amplification:      'Herding amplification',
    fat_tails:                  'Fat tails',
    cross_sector_correlation:   'Cross-sector correlation',
    agent_type_divergence:      'Agent type divergence',
  };
  const TYPE_COLORS = {
    R1:'#6a1422', R2:'#b83030', R3:'#e04040', R4:'#d86020',
    I1:'#7a5808', I2:'#b88010', I3:'#d4a80e', I4:'#ecc820',
  };

  let _currentBatchId = null;
  // Agent trace data stored here to avoid embedding JSON in onclick attributes.
  let _traceRows = [];  // flat array of { trade, agentTrades, shockedIdxs, typeSummaryByTypeTick }

  /** Render the output view from a BatchResult. */
  function render(batchId, result) {
    _currentBatchId = batchId;
    const scenarioName = result.scenario_name || '';
    document.getElementById('out-scenario-name').textContent = scenarioName
      ? `${scenarioName} (EMDOLA Market Engine Investment Thesis)`
      : 'EMDOLA Market Engine Investment Thesis';
    _renderSanityBadges(result.sanity_checks || []);
    const { signal, modelSignal, rest } = _splitNarrative(result.narrative || '');
    _renderSignal(signal);
    _renderModelSignal(modelSignal);
    _renderNarrative(rest);
    _renderSanityDetail(result.sanity_checks || []);
    _renderSignalConsistency(result);
    _renderReplayControls(result);
    _renderAgentTraces(result);
    charts.render(result);
  }

  /**
   * Split the LLM narrative into three parts:
   * - "Investment Thesis" → top band
   * - "Investment Signal" → model signal band above charts
   * - Everything else (Emergent Market Dynamics, Cross-Run Patterns,
   *   Mispricing Signal) → full briefing body
   */
  function _splitNarrative(text) {
    if (!text) return { signal: '', modelSignal: '', rest: '' };
    const sections = text.split(/(?=^## )/m);
    let signal = '';
    let modelSignal = '';
    const rest = [];
    for (const sec of sections) {
      if (/^## Investment Thesis/m.test(sec)) {
        signal = sec.replace(/^## Investment Thesis\s*/m, '').trim();
      } else if (/^## Investment Signal/m.test(sec)) {
        modelSignal = sec.replace(/^## Investment Signal\s*/m, '').trim();
      } else {
        rest.push(sec);
      }
    }
    return { signal, modelSignal, rest: rest.join('').trim() };
  }

  function _renderSignal(text) {
    const el = document.getElementById('out-signal-text');
    if (!el) return;
    if (!text) {
      el.textContent = 'No investment signal generated.';
      el.style.color = 'var(--text-2)';
      return;
    }
    el.style.color = '';
    const html = text
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n/g, '<br>');
    el.innerHTML = `<p>${html}</p>`;
  }

  function _renderModelSignal(text) {
    const el = document.getElementById('out-model-signal-text');
    if (!el) return;
    if (!text) { el.style.display = 'none'; return; }
    el.style.display = '';
    const html = text
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n/g, '<br>');
    el.innerHTML = `<p>${html}</p>`;
  }

  function _sanityClass(sc) {
    if (sc.strength_label === 'N/A') return 'na';
    return (sc.strength_label || '').toLowerCase() || (sc.passed ? 'moderate' : 'weak');
  }

  function _renderSanityBadges(checks) {
    const container = document.getElementById('out-sanity-badges');
    container.innerHTML = checks.map(sc =>
      `<span class="sanity-badge ${_sanityClass(sc)}" title="${sc.message}">${SANITY_CHECK_LABELS[sc.check_name] || sc.check_name}</span>`
    ).join('&nbsp;');
  }

  function _renderNarrative(text) {
    const el = document.getElementById('narrative-text');
    if (!text) {
      el.textContent = 'No narrative generated. (ANTHROPIC_API_KEY not set or narrative disabled.)';
      el.style.color = 'var(--text-2)';
      return;
    }
    el.style.color = '';
    const html = text
      .replace(/^## (.+)$/gm, '<h3 style="color:var(--text-0);margin:16px 0 8px">$1</h3>')
      .replace(/^# (.+)$/gm,  '<h2 style="color:var(--text-0);margin:20px 0 10px">$1</h2>')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/^- (.+)$/gm, '<li>$1</li>')
      .replace(/(<li>.*<\/li>)+/gs, '<ul style="margin:8px 0 8px 18px;line-height:1.6">$&</ul>')
      .replace(/\n\n/g, '</p><p style="margin-bottom:10px">')
      .replace(/\n/g, '<br>');
    el.innerHTML = `<p style="margin-bottom:10px">${html}</p>`;
  }

  function _renderSanityDetail(checks) {
    const el = document.getElementById('sanity-checks-detail');
    if (!checks.length) { el.textContent = '—'; return; }
    el.innerHTML = checks.map(sc => {
      const cls = _sanityClass(sc);
      const label = sc.strength_label || (sc.passed ? 'Moderate' : 'Weak');
      const scoreStr = (sc.strength_label !== 'N/A' && sc.score != null)
        ? ` ${(sc.score * 100).toFixed(0)}`
        : '';
      return `
      <div class="sanity-check-row">
        <span class="sanity-badge ${cls}" style="white-space:nowrap">${label}${scoreStr}</span>
        <div>
          <div style="color:var(--text-0);font-weight:600;margin-bottom:2px">${SANITY_CHECK_LABELS[sc.check_name] || sc.check_name.replace(/_/g,' ')}</div>
          <div class="sanity-check-msg">${sc.message}</div>
          ${sc.detail ? `<div class="sanity-check-msg" style="color:var(--text-2)">${sc.detail}</div>` : ''}
        </div>
      </div>`;
    }).join('');
  }

  /**
   * Signal consistency: for each sector, what % of runs ended with positive (or negative) mispricing?
   * Sorted by consistency strength. Directly answers whether a signal is structural or noise.
   */
  function _renderSignalConsistency(result) {
    const el = document.getElementById('type-activity-summary');
    const runs = (result.run_results || []).filter(r => r.final_mispricing);
    const SECTOR_ABBR  = ['En','Mat','Ind','CDi','Sta','HC','Fin','Tech','CS','Uti','RE'];
    const SECTOR_FULL  = ['Energy','Materials','Industrials','Consumer Discretionary','Consumer Staples',
                          'Health Care','Financials','Information Technology','Communication Services',
                          'Utilities','Real Estate'];
    if (!runs.length) { el.textContent = '—'; return; }

    const n = runs.length;
    const rows = SECTOR_ABBR.map((abbr, si) => {
      const posCount = runs.filter(r => (r.final_mispricing[si] ?? 0) > 0).length;
      const negCount = n - posCount;
      const dominant = posCount >= negCount ? 'pos' : 'neg';
      const dominantCount = dominant === 'pos' ? posCount : negCount;
      const pct = Math.round((dominantCount / n) * 100);
      const mean = runs.reduce((s, r) => s + (r.final_mispricing[si] ?? 0), 0) / n * 100;
      return { abbr, full: SECTOR_FULL[si], dominant, dominantCount, pct, mean, n };
    });

    // Sort: most consistent first
    rows.sort((a, b) => b.pct - a.pct);

    el.innerHTML = rows.map(({ abbr, full, dominant, pct, mean, n }) => {
      const isPos   = dominant === 'pos';
      const color   = isPos ? '#5a9a5a' : '#c84040';
      const label   = isPos ? '▲' : '▼';
      const barPct  = pct;  // already 50–100
      // Normalise bar width: 50% consistency = 0% bar width, 100% = 100% bar width
      const barW    = ((pct - 50) / 50 * 100).toFixed(1);
      return `
        <div class="type-activity-row" title="${full}: ${pct}% of ${n} runs ended ${isPos ? 'above' : 'below'} baseline (mean ${mean > 0 ? '+' : ''}${mean.toFixed(1)}%)">
          <span class="type-badge mono" style="color:var(--text-1);min-width:32px;font-size:10px">${abbr}</span>
          <span style="color:${color};font-size:10px;min-width:10px">${label}</span>
          <div class="activity-bar-outer" style="flex:1">
            <div class="activity-bar-inner" style="width:${barW}%;background:${color};opacity:0.7"></div>
          </div>
          <span style="color:${color};font-size:11px;min-width:36px;text-align:right">${pct}%</span>
        </div>`;
    }).join('');
  }

  function _renderReplayControls(result) {
    const el = document.getElementById('replay-controls');
    if (!result.run_results || result.run_results.length === 0) {
      el.innerHTML = '<span style="color:var(--text-2);font-size:12px">Replay unavailable (aggregate-only mode)</span>';
      return;
    }
    const options = result.run_results.map(r => {
      const avgMispricing = r.final_mispricing
        ? (r.final_mispricing.reduce((a, b) => a + Math.abs(b), 0) / r.final_mispricing.length * 100).toFixed(1)
        : '?';
      return `<option value="${r.run_number}">Run ${r.run_number} — seed ${r.seed_record?.numpy_seed ?? '?'} — avg mispricing ${avgMispricing}%</option>`;
    }).join('');
    el.innerHTML = `
      <div style="display:flex;gap:8px;align-items:center">
        <select id="replay-select" style="flex:1;background:var(--bg-2);border:1px solid var(--bg-3);border-radius:var(--radius);color:var(--text-1);font-size:12px;padding:5px 8px">
          ${options}
        </select>
        <button class="replay-run-btn" onclick="app.replayRun(parseInt(document.getElementById('replay-select').value))">
          ▶ Replay
        </button>
      </div>`;
  }

  // ── Agent traces ──────────────────────────────────────────────────────────

  const SECTOR_NAMES = ['Energy','Materials','Industrials','Consumer Discretionary',
    'Consumer Staples','Health Care','Financials','Information Technology',
    'Communication Services','Utilities','Real Estate'];
  const SECTOR_ABBR = ['En','Mat','Ind','CDi','Sta','HC','Fin','Tech','CS','Uti','RE'];
  const TYPE_LABELS_TRACE = {
    R1:'Passive retail', R2:'Self-directed', R3:'High net worth', R4:'Speculative',
    I1:'Asset manager',  I2:'Hedge fund',    I3:'Pension fund',   I4:'Sovereign wealth',
  };

  function _renderAgentTraces(result) {
    const section = document.getElementById('agent-traces-section');
    const el = document.getElementById('agent-traces-list');
    const run0 = (result.run_results || [])[0];
    if (!run0 || !run0.agent_traces || run0.agent_traces.length === 0) {
      section.style.display = 'none';
      return;
    }
    section.style.display = '';

    const shockedIdxs = new Set(
      (result.shocked_sectors || []).map(s => SECTOR_NAMES.indexOf(s)).filter(i => i >= 0)
    );
    const typeSummaryByTypeTick = {};
    for (const ts of (run0.type_summaries || [])) {
      typeSummaryByTypeTick[`${ts.agent_type}:${ts.tick}`] = ts;
    }

    // Group all trades by agent for modal context (full agent trade list on click).
    const byAgent = {};
    for (const t of run0.agent_traces) {
      (byAgent[t.agent_id] = byAgent[t.agent_id] || []).push(t);
    }

    const TAG_TOOLTIPS = {
      SHOCK:    'Traded in a directly shocked sector during the shock window',
      COUPLING: 'Traded in an unshocked sector during the shock window — macro sentiment spillover',
      REVERSAL: 'Reversed direction in this sector vs. prior trade — conviction shift',
      DISSENT:  'Traded against own type\'s net direction this tick — contrarian signal',
    };

    // Build flat list of trade rows with tags; store data in closure array.
    _traceRows = [];
    const tagCounts = { COUPLING: 0, DISSENT: 0, REVERSAL: 0, SHOCK: 0 };
    const rows = run0.agent_traces.map((t, i) => {
      const tags = [];
      if (t.shock_active && !shockedIdxs.has(t.sector_index)) tags.push('COUPLING');
      else if (t.shock_active) tags.push('SHOCK');
      const prev = (byAgent[t.agent_id] || []).filter(x => x.sector_index === t.sector_index && x.tick < t.tick);
      if (prev.length > 0 && prev[prev.length-1].direction !== t.direction) tags.push('REVERSAL');
      const ts = typeSummaryByTypeTick[`${t.agent_type}:${t.tick}`];
      if (ts) {
        const typeNet = ts.net_direction[t.sector_index] || 0;
        if (typeNet !== 0 && Math.sign(typeNet) !== t.direction) tags.push('DISSENT');
      }
      tags.forEach(tag => { if (tagCounts[tag] !== undefined) tagCounts[tag]++; });

      _traceRows.push({
        agentType: t.agent_type,
        archetypeNum: t.archetype_num,
        agentTrades: byAgent[t.agent_id] || [],
        shockedIdxs: Array.from(shockedIdxs),
        netDirMap: Object.fromEntries(
          Object.entries(typeSummaryByTypeTick).map(([k, v]) => [k, v.net_direction])
        ),
      });

      const idx = _traceRows.length - 1;
      const color = TYPE_COLORS[t.agent_type] || 'var(--text-2)';
      const variant = String.fromCharCode(64 + t.archetype_num);
      const dirLabel = t.direction > 0
        ? `<span style="color:var(--green)">▲</span>`
        : `<span style="color:var(--red)">▼</span>`;
      const sector = SECTOR_NAMES[t.sector_index] || `S${t.sector_index}`;
      const tagHtml = tags
        .map(tag => `<span class="trace-tag trace-tag-${tag.toLowerCase()}" title="${TAG_TOOLTIPS[tag] || ''}">${tag}</span>`)
        .join('');

      return `<div class="trace-agent-row" onclick="output._openTrace(${idx})" title="Click for full trade log">
        <span class="type-badge" style="color:${color};min-width:28px">${t.agent_type}</span>
        <span style="color:var(--text-2);font-size:10px;min-width:20px">${variant}</span>
        <span style="color:var(--text-2);font-size:11px;min-width:24px;text-align:right">t${t.tick}</span>
        <span style="font-size:11px;flex:1;padding-left:6px">${sector}</span>
        <span style="min-width:18px">${dirLabel}</span>
        <span>${tagHtml}</span>
      </div>`;
    });

    // Summary line: count by tag type for at-a-glance pattern reading.
    const summaryParts = ['COUPLING','DISSENT','REVERSAL','SHOCK']
      .filter(tag => tagCounts[tag] > 0)
      .map(tag => `<span class="trace-tag trace-tag-${tag.toLowerCase()}" title="${TAG_TOOLTIPS[tag]}">${tagCounts[tag]} ${tag}</span>`);
    document.getElementById('agent-traces-summary').innerHTML = summaryParts.join(' ');

    el.innerHTML = rows.join('');
  }

  function _openTrace(idx) {
    const row = _traceRows[idx];
    if (!row) return;
    profiles.openTraceModal(row.agentType, row.archetypeNum, row.agentTrades, row.shockedIdxs, row.netDirMap);
  }

  return { render, _openTrace };
})();
