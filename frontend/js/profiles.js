/**
 * profiles.js — profile review view controller.
 *
 * Displays 8 investor type cards. Clicking a type opens a modal showing
 * all 3 LLM-generated personas with full narrative and all characteristics.
 */
const profiles = (() => {
  const AGENT_TYPES = ['R1','R2','R3','R4','I1','I2','I3','I4'];
  const TYPE_COLORS = {
    R1:'var(--r1)', R2:'var(--r2)', R3:'var(--r3)', R4:'var(--r4)',
    I1:'var(--i1)', I2:'var(--i2)', I3:'var(--i3)', I4:'var(--i4)',
  };
  const TYPE_LABELS = {
    R1:'Passive retail',        R2:'Self-directed retail',
    R3:'High net worth retail', R4:'Speculative retail',
    I1:'Asset manager',         I2:'Hedge fund',
    I3:'Pension fund',          I4:'Sovereign wealth',
  };
  const TYPE_DESCRIPTIONS = {
    R1: 'Long-term passive investors, pension savers and index fund holders who rarely trade.',
    R2: 'Active platform traders, self-directed retail investors monitoring markets daily.',
    R3: 'Sophisticated high-net-worth investors with regular professional portfolio review.',
    R4: 'Momentum-driven speculators, high-frequency monitoring, concentrated positions.',
    I1: 'Long-only institutional asset managers running benchmark-relative daily processes.',
    I2: 'Hedge funds, opportunistic, high-conviction, concentrated across sectors.',
    I3: 'Pension funds and insurance companies, liability-driven, conservative equity allocation.',
    I4: 'Sovereign wealth funds, very large capital base, long time horizons, diversified mandates.',
  };

  // All 21 characteristics in display order, grouped by category.
  const STABLE_TRAITS = [
    ['risk_aversion',         'Risk aversion'],
    ['time_horizon',          'Time horizon'],
    ['belief_formation',      'Fundamentalist ↔ Chartist'],
    ['information_quality',   'Information quality'],
    ['herding_sensitivity',   'Herding sensitivity'],
    ['decision_style',        'Decision style (systematic ↔ intuitive)'],
    ['strategy_adaptability', 'Strategy adaptability'],
    ['conviction_threshold',  'Conviction threshold'],
    ['institutional_inertia', 'Institutional inertia'],
    ['memory_window',         'Memory window'],
  ];
  const COGNITIVE_BIASES = [
    ['overconfidence',        'Overconfidence'],
    ['anchoring',             'Anchoring'],
    ['confirmation_bias',     'Confirmation bias'],
    ['recency_bias',          'Recency bias'],
    ['salience_bias',         'Salience bias'],
    ['pattern_matching_bias', 'Pattern matching bias'],
  ];
  const EMOTIONAL_BIASES = [
    ['loss_aversion',              'Loss aversion'],
    ['winner_selling_loser_holding','Winner-selling / loser-holding'],
    ['regret_aversion',            'Regret aversion'],
    ['fomo',                       'FOMO'],
    ['mental_accounting',          'Mental accounting'],
    ['ownership_bias',             'Ownership bias'],
  ];

  let _profileData = null;

  /** Initialise the profiles view with data from the profiles_complete WS event. */
  function init(profileData) {
    _profileData = profileData;
    _buildTypeGrid();
  }

  function _buildTypeGrid() {
    const grid = document.getElementById('profiles-type-grid');
    grid.innerHTML = AGENT_TYPES.map(type => `
      <div class="profile-type-card" data-type="${type}"
           onclick="profiles.openModal('${type}')">
        <span class="profile-type-code" style="color:${TYPE_COLORS[type]}">${type}</span>
        <div class="profile-type-name">${TYPE_LABELS[type]}</div>
        <div class="profile-type-hint">Click to view 3 variants →</div>
      </div>
    `).join('');
  }

  /** Open the modal for the selected agent type — shows all 3 personas. */
  function openModal(type) {
    if (!_profileData) return;
    const personas = _profileData[type] || [];

    document.getElementById('modal-type-code').textContent = type;
    document.getElementById('modal-type-code').style.color = TYPE_COLORS[type];
    document.getElementById('modal-type-name').textContent = TYPE_LABELS[type];
    document.getElementById('modal-type-desc').textContent = TYPE_DESCRIPTIONS[type];

    const body = document.getElementById('modal-body');
    body.className = 'modal-body';
    body.innerHTML = personas.map(p => _renderPersonaCard(p, null)).join('');

    document.getElementById('profile-modal-overlay').style.display = 'flex';
    document.body.style.overflow = 'hidden';
  }

  /**
   * Open the modal showing all 3 variants for an agent type, annotated with
   * the archetype distribution from a completed run.
   * distribution: {a: n, b: n, c: n} — agent counts per archetype.
   * runLabel: e.g. "Run 1".
   */
  function openModalWithDistribution(type, distribution, runLabel) {
    if (!_profileData) return;
    const personas = _profileData[type] || [];

    document.getElementById('modal-type-code').textContent = type;
    document.getElementById('modal-type-code').style.color = TYPE_COLORS[type];
    document.getElementById('modal-type-name').textContent = TYPE_LABELS[type];
    document.getElementById('modal-type-desc').textContent = TYPE_DESCRIPTIONS[type];

    const total = (distribution.a || 0) + (distribution.b || 0) + (distribution.c || 0);
    const distByPersona = { 1: distribution.a || 0, 2: distribution.b || 0, 3: distribution.c || 0 };

    // Build a prominent distribution banner below the type description.
    const existingBanner = document.getElementById('modal-dist-banner');
    if (existingBanner) existingBanner.remove();
    const banner = document.createElement('div');
    banner.id = 'modal-dist-banner';
    banner.className = 'modal-dist-banner';
    banner.innerHTML = `
      <span class="dist-banner-label">Active persona split, ${runLabel}</span>
      <div class="dist-banner-bars">
        ${['A','B','C'].map((letter, i) => {
          const count = [distribution.a, distribution.b, distribution.c][i] || 0;
          const pct = total > 0 ? Math.round(count / total * 100) : 0;
          return `
            <div class="dist-bar-item">
              <span class="dist-bar-letter">${letter}</span>
              <div class="dist-bar-track">
                <div class="dist-bar-fill" style="width:${pct}%"></div>
              </div>
              <span class="dist-bar-count">${count} <span class="dist-bar-pct">(${pct}%)</span></span>
            </div>`;
        }).join('')}
      </div>`;

    const descEl = document.getElementById('modal-type-desc');
    descEl.insertAdjacentElement('afterend', banner);

    const body = document.getElementById('modal-body');
    body.className = 'modal-body';
    body.innerHTML = personas.map(p => _renderPersonaCard(p, distByPersona[p.persona_number])).join('');

    document.getElementById('profile-modal-overlay').style.display = 'flex';
    document.body.style.overflow = 'hidden';
  }

  function closeModal() {
    document.getElementById('profile-modal-overlay').style.display = 'none';
    document.body.style.overflow = '';
    const banner = document.getElementById('modal-dist-banner');
    if (banner) banner.remove();
  }

  function togglePersonaDetail(btn) {
    const chars = btn.nextElementSibling;
    const open = chars.style.display !== 'none';
    chars.style.display = open ? 'none' : 'block';
    btn.textContent = open ? 'Show characteristics ▾' : 'Hide characteristics ▴';
  }

  function _renderCharGroup(title, charList, chars) {
    const rows = charList.map(([key, label]) => {
      const val = chars[key];
      if (val === null || val === undefined) return '';
      const pct = Math.round(val * 100);
      return `
        <div class="char-row">
          <span class="char-label">${label}</span>
          <div class="char-bar-outer">
            <div class="char-bar-inner" style="width:${pct}%"></div>
          </div>
          <span class="char-value">${pct}%</span>
        </div>`;
    }).join('');

    if (!rows.trim()) return '';
    return `<div class="char-section-title">${title}</div>${rows}`;
  }

  function _renderPersonaCard(persona, agentCount) {
    const chars = persona.characteristics || {};
    const variant = String.fromCharCode(64 + persona.persona_number); // A, B, C
    const countBadge = agentCount != null
      ? `<span class="persona-count-badge">${agentCount} agent${agentCount !== 1 ? 's' : ''}</span>`
      : '';

    return `
      <div class="persona-card">
        <div class="persona-header-row">
          <div class="persona-label">Variant ${variant}</div>
          ${countBadge}
        </div>
        <p class="persona-narrative">${persona.narrative}</p>
        <button class="persona-detail-btn" onclick="profiles.togglePersonaDetail(this)">
          Show characteristics ▾
        </button>
        <div class="persona-chars" style="display:none">
          ${_renderCharGroup('Stable traits', STABLE_TRAITS, chars)}
          ${_renderCharGroup('Cognitive biases', COGNITIVE_BIASES, chars)}
          ${_renderCharGroup('Emotional biases', EMOTIONAL_BIASES, chars)}
        </div>
      </div>`;
  }

  // ── Agent trace modal ─────────────────────────────────────────────────────

  const SECTOR_NAMES_TRACE = [
    'Energy','Materials','Industrials','Consumer Discretionary','Consumer Staples',
    'Health Care','Financials','Information Technology','Communication Services',
    'Utilities','Real Estate',
  ];

  /**
   * Open the trace modal for one representative agent.
   *
   * agentType    : e.g. 'R2'
   * archetypeNum : 1, 2, or 3
   * trades       : array of AgentTraceRecord objects (interesting trades only)
   * shockedIdxs  : array of shocked sector indices (for COUPLING tag)
   * netDirMap    : { "R2:tick": [net_direction per sector, …], … } — for DISSENT tag
   */
  function openTraceModal(agentType, archetypeNum, trades, shockedIdxs, netDirMap) {
    const variant  = String.fromCharCode(64 + archetypeNum); // 1→A, 2→B, 3→C
    const color    = TYPE_COLORS[agentType] || 'var(--text-2)';
    const shockedSet = new Set(shockedIdxs);

    document.getElementById('trace-modal-type-code').textContent  = agentType;
    document.getElementById('trace-modal-type-code').style.color  = color;
    document.getElementById('trace-modal-type-name').textContent  = TYPE_LABELS[agentType] || '';
    document.getElementById('trace-modal-variant').textContent    = `Variant ${variant}`;

    // Compute LARGE threshold: top 33% of this agent's trades.
    const mags = trades.map(t => t.magnitude);
    const sorted = [...mags].sort((a, b) => a - b);
    const p67 = sorted[Math.floor(sorted.length * 0.67)] || 0;

    // Track previous direction per sector for REVERSAL detection.
    const prevDir = {};

    const rows = trades.map(t => {
      const tags = [];
      if (t.shock_active) tags.push('SHOCK');
      if (t.shock_active && !shockedSet.has(t.sector_index)) tags.push('COUPLING');
      if (t.magnitude >= p67 && mags.length > 2) tags.push('LARGE');
      if (prevDir[t.sector_index] !== undefined && prevDir[t.sector_index] !== t.direction) {
        tags.push('REVERSAL');
      }
      prevDir[t.sector_index] = t.direction;
      const key = `${agentType}:${t.tick}`;
      const netDir = netDirMap[key];
      if (netDir) {
        const typeNet = netDir[t.sector_index] || 0;
        if (typeNet !== 0 && Math.sign(typeNet) !== t.direction) tags.push('DISSENT');
      }

      const dirLabel = t.direction > 0
        ? '<span style="color:var(--green)">▲ Buy</span>'
        : '<span style="color:var(--red)">▼ Sell</span>';
      const tagHtml = tags
        .map(tag => `<span class="trace-tag trace-tag-${tag.toLowerCase()}">${tag}</span>`)
        .join(' ');
      const magStr = t.magnitude.toFixed(2);
      const sectorName = SECTOR_NAMES_TRACE[t.sector_index] || `S${t.sector_index}`;

      return `
        <tr class="trace-table-row">
          <td class="trace-td mono">${t.tick}</td>
          <td class="trace-td">${sectorName}</td>
          <td class="trace-td">${dirLabel}</td>
          <td class="trace-td mono" style="text-align:right">${magStr}B</td>
          <td class="trace-td">${tagHtml}</td>
        </tr>`;
    }).join('');

    const body = document.getElementById('trace-modal-body');
    if (!trades.length) {
      body.innerHTML = '<p style="color:var(--text-2);font-size:12px;padding:16px">No notable trades recorded for this agent.</p>';
    } else {
      body.innerHTML = `
        <table class="trace-table">
          <thead>
            <tr>
              <th class="trace-th">Tick</th>
              <th class="trace-th">Sector</th>
              <th class="trace-th">Action</th>
              <th class="trace-th" style="text-align:right">Size</th>
              <th class="trace-th">Tags</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>`;
    }

    document.getElementById('trace-modal-overlay').style.display = 'flex';
    document.body.style.overflow = 'hidden';
  }

  function closeTraceModal() {
    document.getElementById('trace-modal-overlay').style.display = 'none';
    document.body.style.overflow = '';
  }

  // Close modals on Escape key.
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { closeModal(); closeTraceModal(); }
  });

  return { init, openModal, openModalWithDistribution, closeModal, togglePersonaDetail, openTraceModal, closeTraceModal };
})();
