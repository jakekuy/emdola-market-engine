/**
 * setup.js — 4-tab calibration form.
 *
 * Builds the lambda grid, shock add/remove UI, agent override grid.
 * setup.buildCalibration() returns a CalibrationInput JSON object.
 */
const setup = (() => {
  const SECTORS = ['En','Mat','Ind','CDi','Sta','HC','Fin','Tech','CS','Uti','RE'];
  const SECTOR_NAMES = {
    En:'Energy', Mat:'Materials', Ind:'Industrials', CDi:'Cons. Discret.',
    Sta:'Cons. Staples', HC:'Health Care', Fin:'Financials',
    Tech:'Info. Tech.', CS:'Comm. Services', Uti:'Utilities', RE:'Real Estate',
  };
  const LAMBDA_DEFAULTS = [1.50,2.00,1.20,1.20,1.00,1.20,0.80,0.80,1.20,1.80,2.50];
  const AGENT_TYPES = ['R1','R2','R3','R4','I1','I2','I3','I4'];
  const TYPE_LABELS = {
    R1:'Passive retail', R2:'Self-directed retail', R3:'HNW retail', R4:'Speculative retail',
    I1:'Asset manager',  I2:'Hedge fund',           I3:'Pension fund', I4:'Sovereign wealth',
  };
  const TYPE_DESCRIPTIONS = {
    R1: 'Long-term passive investors — pension savers and index fund holders who rarely trade.',
    R2: 'Active platform traders — self-directed retail investors monitoring markets daily.',
    R3: 'Sophisticated high-net-worth investors with regular professional portfolio review.',
    R4: 'Momentum-driven speculators — high-frequency monitoring, concentrated positions.',
    I1: 'Long-only institutional asset managers running benchmark-relative daily processes.',
    I2: 'Hedge funds — opportunistic, high-conviction, concentrated across sectors.',
    I3: 'Pension funds and insurance companies — liability-driven, conservative equity allocation.',
    I4: 'Sovereign wealth funds — very large capital base, long time horizons, diversified mandates.',
  };
  const DEFAULT_AUM = { R1:500, R2:133, R3:133, R4:50, I1:2667, I2:100, I3:583, I4:121 };

  let shockCount = 0;

  // ── Plain-language context helpers ────────────────────────────────────────────

  function _lambdaLabel(val) {
    if (val <= 0.08) return 'Very liquid — prices change little even on large trades';
    if (val <= 0.11) return 'Liquid — gradual price response to buying/selling pressure';
    if (val <= 0.14) return 'Moderate — balanced price sensitivity';
    if (val <= 0.19) return 'Responsive — meaningful price moves on moderate activity';
    return 'Illiquid — prices can shift sharply on relatively small volumes';
  }

  function _nmLabel(val) {
    if (val <= 0.05) return 'Off — price moves driven by agent behaviour only, no external signals';
    if (val <= 0.20) return 'Low — faint background noise from news and commentary';
    if (val <= 0.45) return 'Moderate — normal level of market commentary and analyst activity';
    if (val <= 0.70) return 'High — active media coverage strongly influences agent signals';
    return 'Very high — sentiment and narrative dominate over fundamentals';
  }

  function _updateLambdaCtx(sector) {
    const val = parseFloat(document.getElementById(`lambda-${sector}`)?.value) || 0;
    const ctx = document.getElementById(`lambda-ctx-${sector}`);
    if (!ctx) return;
    ctx.textContent = _lambdaLabel(val);
    ctx.className = 'lambda-context' + (val >= 0.20 ? ' responsive' : '');
  }

  function init() {
    _buildLambdaGrid();
    _buildAgentOverrideGrid();
    _initTabs();
    _initNmContext();
  }

  function _initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
      });
    });
  }

  function _buildLambdaGrid() {
    const grid = document.getElementById('lambda-grid');
    SECTORS.forEach((s, i) => {
      const defVal = LAMBDA_DEFAULTS[i];
      grid.innerHTML += `
        <div class="lambda-cell" data-sector="${s.toLowerCase()}">
          <span class="lambda-sector-name">${SECTOR_NAMES[s] || s}</span>
          <input id="lambda-${s}" type="number" min="0.01" max="1" step="0.01"
                 value="${defVal}" oninput="setup.updateLambdaCtx('${s}')" />
          <div class="lambda-context" id="lambda-ctx-${s}">${_lambdaLabel(defVal)}</div>
        </div>`;
    });
  }

  function _initNmContext() {
    const input = document.getElementById('nm-intensity');
    if (!input) return;
    const ctx = document.createElement('div');
    ctx.id = 'nm-context';
    ctx.className = 'param-context';
    ctx.textContent = _nmLabel(parseFloat(input.value) || 0.4);
    input.parentElement.appendChild(ctx);
    input.addEventListener('input', () => {
      ctx.textContent = _nmLabel(parseFloat(input.value) || 0);
    });
  }

  function _buildAgentOverrideGrid() {
    const grid = document.getElementById('agent-override-grid');
    AGENT_TYPES.forEach(type => {
      const typeVar = `var(--${type.toLowerCase()})`;
      grid.innerHTML += `
        <div class="shock-card" data-type="${type}" style="margin-bottom:10px">
          <div class="shock-card-header">
            <div>
              <span class="shock-card-title">
                <span style="font-family:'SF Mono','Fira Code',monospace;color:${typeVar}">${type}</span>
                — ${TYPE_LABELS[type]}
              </span>
              <div class="agent-type-desc">${TYPE_DESCRIPTIONS[type]}</div>
            </div>
          </div>
          <div class="field-row">
            <div class="field-group">
              <label>Number of agents</label>
              <input id="agent-count-${type}" type="number" min="30" max="200" step="10" placeholder="Default (30)" />
              <span class="hint">30–200 per type</span>
            </div>
            <div class="field-group">
              <label>Total capital (billions USD)</label>
              <input id="agent-aum-${type}" type="number" min="1" placeholder="Default ($${DEFAULT_AUM[type]}B)" />
              <span class="hint">Total investment capital managed by this type across all agents</span>
            </div>
          </div>
        </div>`;
    });
  }

  function addShock() {
    shockCount++;
    const id = shockCount;
    const sectorCheckboxes = SECTORS.map((s, i) =>
      `<label class="sector-cb-label">
         <input type="checkbox" name="shock-sectors-${id}" value="${i}"> ${s}
       </label>`
    ).join('');

    const html = `
      <div class="shock-card" id="shock-card-${id}">
        <div class="shock-card-header">
          <span class="shock-card-title">Shock ${id}</span>
          <button class="btn-remove" onclick="setup.removeShock(${id})">Remove</button>
        </div>
        <div class="field-row">
          <div class="field-group">
            <label>Starts on day</label>
            <input id="shock-onset-${id}" type="number" min="0" value="50" />
          </div>
          <div class="field-group">
            <label>Lasts (trading days)</label>
            <input id="shock-duration-${id}" type="number" min="1" value="20" />
          </div>
        </div>
        <div class="field-row">
          <div class="field-group">
            <label>Magnitude <span class="hint">(positive = upward price pressure, negative = downward — 0.10 = 10% impact)</span></label>
            <input id="shock-magnitude-${id}" type="number" step="0.01" value="0.15" />
          </div>
          <div class="field-group">
            <label>Type</label>
            <select id="shock-type-${id}">
              <option value="acute">Acute — sudden repricing event</option>
              <option value="chronic">Chronic — slow-building trend, holds permanently</option>
            </select>
          </div>
        </div>
        <div class="field-row">
          <div class="field-group">
            <label>What it affects</label>
            <select id="shock-channel-${id}">
              <option value="both">Market prices and sentiment</option>
              <option value="market">Market prices only</option>
              <option value="influence">Sentiment only</option>
            </select>
          </div>
          <div class="field-group">
            <label>Fair value reverts? <span class="hint">(acute only — actual prices are emergent)</span></label>
            <select id="shock-reversion-${id}">
              <option value="false">No — repricing is permanent</option>
              <option value="true">Yes — reverts over duration window</option>
            </select>
          </div>
        </div>
        <div class="field-group">
          <label>Affected sectors</label>
          <div class="sector-checkboxes">${sectorCheckboxes}</div>
        </div>
      </div>`;
    document.getElementById('shock-list').insertAdjacentHTML('beforeend', html);
  }

  function removeShock(id) {
    const card = document.getElementById(`shock-card-${id}`);
    if (card) card.remove();
  }

  function onSeedPolicyChange() {
    const policy = document.getElementById('seed-policy').value;
    document.getElementById('fixed-seed-group').style.display =
      policy === 'fixed' ? 'block' : 'none';
  }

  function _readShocks() {
    const shocks = [];
    document.querySelectorAll('.shock-card[id^="shock-card-"]').forEach(card => {
      const id = card.id.replace('shock-card-', '');
      const sectors = [...document.querySelectorAll(`input[name="shock-sectors-${id}"]:checked`)]
        .map(cb => parseInt(cb.value));
      if (sectors.length === 0) return;

      shocks.push({
        onset_tick: parseInt(document.getElementById(`shock-onset-${id}`).value) || 0,
        magnitude: parseFloat(document.getElementById(`shock-magnitude-${id}`).value) || 0.1,
        duration: parseInt(document.getElementById(`shock-duration-${id}`).value) || 20,
        affected_sectors: sectors,
        shock_type: document.getElementById(`shock-type-${id}`).value,
        channel: document.getElementById(`shock-channel-${id}`).value,
        reversion: document.getElementById(`shock-reversion-${id}`).value === 'true',
      });
    });
    return shocks;
  }

  function _readLambdas() {
    return SECTORS.map(s => {
      const val = parseFloat(document.getElementById(`lambda-${s}`)?.value);
      return isNaN(val) || val <= 0 ? 0.12 : val;
    });
  }

  function _readAgentOverrides() {
    const overrides = {};
    AGENT_TYPES.forEach(type => {
      const countEl = document.getElementById(`agent-count-${type}`);
      const aumEl   = document.getElementById(`agent-aum-${type}`);
      const count   = parseInt(countEl?.value);
      const aum     = parseFloat(aumEl?.value);
      const entry = {};
      if (!isNaN(count) && count >= 30) entry.agent_count = count;
      if (!isNaN(aum) && aum > 0) entry.aum_override = aum;
      if (Object.keys(entry).length > 0) overrides[type] = entry;
    });
    return overrides;
  }

  /** Build a CalibrationInput object from form values. */
  function buildCalibration() {
    const seedPolicy = document.getElementById('seed-policy').value;
    const fixedSeedEl = document.getElementById('fixed-seed');
    const overrides = _readAgentOverrides();

    return {
      scenario_name: document.getElementById('scenario-name').value || 'EMDOLA Run',
      scenario_description: document.getElementById('scenario-desc').value || '',
      market_context: document.getElementById('market-context').value || '',
      market: {
        lambdas: _readLambdas(),
        non_market_signal_intensity: parseFloat(document.getElementById('nm-intensity').value) || 0.4,
        narrative_half_life: parseInt(document.getElementById('narrative-half-life').value) || 10,
        pre_shock_ticks: 50,
      },
      shocks: _readShocks(),
      agents: { overrides },
      run: {
        total_ticks: parseInt(document.getElementById('total-ticks').value) || 250,
        num_runs: parseInt(document.getElementById('num-runs').value) || 10,
        data_granularity: document.getElementById('data-granularity').value,
        storage_mode: document.getElementById('storage-mode').value,
        seed_policy: seedPolicy,
        fixed_seed: seedPolicy === 'fixed' ? parseInt(fixedSeedEl?.value) || 42 : null,
      },
    };
  }

  // ── Preset scenarios ────────────────────────────────────────────────────────

  const PRESETS = {
    iran_energy: {
      scenario_name: 'Iran Conflict — Strait of Hormuz Closure',
      scenario_description:
        'Military conflict erupts between Israel and Iran, drawing in US forces within weeks. ' +
        'Iranian forces mine the Strait of Hormuz, temporarily closing the world\'s most critical oil chokepoint. ' +
        'Roughly 20% of global seaborne oil supply is disrupted. Brent crude spikes above $140/bbl in the first two weeks ' +
        'before partial re-routing and strategic reserve releases stabilise prices at around $110–120/bbl. ' +
        'The conflict remains contained but unresolved, sustaining elevated risk premia across energy markets. ' +
        'Inflation expectations re-accelerate; central banks face a stagflationary dilemma. Regional contagion is limited ' +
        'but Middle East sovereign credit spreads widen sharply.',
      lambdas: [1.50, 2.00, 1.20, 1.20, 1.00, 1.20, 0.80, 0.80, 1.20, 1.80, 2.50],
      nm_intensity: 0.60,
      narrative_half_life: 7,
      total_ticks: 125,
      num_runs: 50,
      shocks: [
        { onset: 40, duration: 25, magnitude:  0.15, type: 'chronic', channel: 'market', reversion: false, sectors: [0, 1] },
        { onset: 40, duration: 25, magnitude: -0.12, type: 'acute',   channel: 'both',   reversion: true,  sectors: [2, 4, 6, 7, 8, 10] },
      ],
    },

    russia_ukraine: {
      scenario_name: 'Russia–Ukraine Invasion — Multi-Commodity Shock (2022 Backtest)',
      scenario_description:
        'Russia\'s invasion of Ukraine on 24 February 2022 triggered the largest European land war since 1945 ' +
        'and a simultaneous multi-commodity supply shock not seen since the 1970s. Russia — supplying ~40% of European ' +
        'piped gas and ~12% of global oil — began weaponising energy exports within weeks of the invasion. Ukraine — ' +
        'the world\'s fourth-largest wheat exporter and a major producer of sunflower oil and ammonia-based fertilisers — ' +
        'had its agricultural exports and Black Sea port operations disrupted almost immediately. ' +
        'Brent crude rose from $95/bbl to $130/bbl before partial stabilisation. European TTF gas reached 10× its 5-year average. ' +
        'Wheat futures rose ~50% at peak; fertiliser prices rose ~70%. ' +
        'The consensus reaction was broad risk-off: European recession, global stagflation, sell everything cyclical. ' +
        'In practice, outcomes diverged sharply in ways markets were slow to price: US energy producers and LNG exporters ' +
        'became strategically indispensable; agricultural input companies saw margin expansion not contraction; ' +
        'and consumer-facing sectors absorbed input cost inflation in ways their "defensive" label did not predict. ' +
        'This backtest asks: does the model generate a similarly non-consensus sector repricing signal?',
      lambdas: [2.00, 2.00, 1.20, 1.20, 1.00, 1.20, 0.80, 0.80, 1.20, 1.80, 2.50],
      nm_intensity: 0.70,
      narrative_half_life: 15,
      total_ticks: 125,
      num_runs: 50,
      shocks: [
        // Energy: sustained supply disruption repricing. Brent held ~$100–110 for most of 2022 (+15–20% vs pre-war).
        { onset: 40, duration: 40, magnitude:  0.22, type: 'chronic', channel: 'market', reversion: false, sectors: [0] },
        // Materials: wheat/fertiliser/metals repricing. Agricultural exports disrupted for duration of conflict.
        { onset: 40, duration: 35, magnitude:  0.18, type: 'chronic', channel: 'market', reversion: false, sectors: [1] },
        // Industrials + Consumer Discretionary + Consumer Staples: European supply chain shock + consumer cost squeeze.
        // Acute with partial reversion as markets found alternative suppliers over Q2–Q3 2022.
        { onset: 40, duration: 20, magnitude: -0.12, type: 'acute',   channel: 'both',   reversion: true,  sectors: [2, 3, 4] },
        // Financials: European bank exposure, sanctions complexity, credit spread widening.
        // Reversion true — US financials decoupled within weeks; European banks adapted.
        { onset: 40, duration: 15, magnitude: -0.12, type: 'acute',   channel: 'both',   reversion: true,  sectors: [6] },
      ],
    },

    carbon_price: {
      scenario_name: 'Carbon Price Acceleration — G7 Climate Policy Shock',
      scenario_description:
        'Following an unexpected G7 summit agreement, a coordinated binding carbon price floor of $100/tonne is announced ' +
        'with full implementation in 24 months and immediate activation of a Carbon Border Adjustment Mechanism on EU imports. ' +
        'The announcement significantly exceeds all prior policy expectations. The EU ETS had been trading around $60–70/tonne; ' +
        'a $100/tonne floor represents a ~50% step-change in the marginal cost signal for carbon-intensive industries. ' +
        'The standard ESG trade — long clean energy and utilities, short fossil fuels — is immediately initiated by institutional allocators. ' +
        'But the actual repricing is likely more complex: energy majors\' stranded asset discount is partly offset by near-term ' +
        'commodity price increases as carbon costs raise production costs and constrict supply; heavy industry (steel, cement, chemicals) ' +
        'faces structural margin compression; utilities pivoting to renewables see structural re-rating while legacy coal operators face write-downs; ' +
        'and the green capital markets boom — green bond issuance, carbon trading desks, sustainability-linked lending — ' +
        'creates an unexpected tailwind for capital markets-exposed financial institutions. ' +
        'Consumer sectors face a persistent energy input cost squeeze from higher production energy costs, agricultural energy use, and logistics. ' +
        'The question for the model: beyond the obvious ESG trade, where does behavioural repricing diverge from consensus allocation?',
      lambdas: [1.50, 2.00, 1.20, 1.20, 1.00, 1.20, 0.80, 0.80, 1.20, 2.00, 2.50],
      nm_intensity: 0.75,
      narrative_half_life: 20,
      total_ticks: 250,
      num_runs: 50,
      shocks: [
        // Energy + Materials + Industrials: chronic structural repricing of carbon-intensive sectors.
        // Energy stranded asset discount (~20%), heavy industry margin compression (~15%).
        // Long duration (60 ticks) reflects 24-month policy horizon priced in immediately.
        { onset: 40, duration: 60, magnitude: -0.20, type: 'chronic', channel: 'market', reversion: false, sectors: [0, 1, 2] },
        // Utilities: clean power fundamental value structurally higher as carbon price makes renewables
        // competitively dominant vs gas/coal. Both channels — market repricing + strong ESG narrative.
        { onset: 40, duration: 55, magnitude:  0.20, type: 'chronic', channel: 'both',   reversion: false, sectors: [9] },
        // IT/Tech: clean tech enabler sentiment wave. Shorter duration — narrative burns hot then normalises.
        { onset: 40, duration: 20, magnitude:  0.10, type: 'acute',   channel: 'both',   reversion: true,  sectors: [7] },
        // Consumer Staples + Consumer Discretionary: persistent energy input cost inflation.
        // No reversion — structural cost increase, not transitory.
        { onset: 40, duration: 30, magnitude: -0.10, type: 'acute',   channel: 'both',   reversion: false, sectors: [3, 4] },
      ],
    },

    oil_crash_2016: {
      scenario_name: 'Oil Price Collapse — Oversupply Shock (2015–16 Backtest)',
      scenario_description:
        'Between June 2014 and January 2016, Brent crude collapsed from $115/bbl to $27/bbl — a 76% decline driven by ' +
        'Saudi Arabia\'s decision to defend market share rather than cut production, a US shale revolution that had added ' +
        '4 million barrels/day of new supply since 2010, and a sharp slowdown in Chinese industrial demand. ' +
        'OPEC refused to blink at its November 2014 meeting, cementing the view that oversupply would persist. ' +
        'The consensus view entering 2015 was straightforwardly positive for the broader economy: lower oil prices ' +
        'mean lower consumer costs, lower input prices across industry, and a net transfer from producers to consumers. ' +
        '"Cheap oil is a tax cut" was the consensus framing. In practice, the outcomes diverged sharply from this narrative. ' +
        'Energy sector equities fell 50–60%, destroying more market cap than the consumer sector gained. ' +
        'High-yield credit markets — with heavy energy sector loan and bond exposure — experienced near-systemic stress: ' +
        'spreads on HY energy bonds widened to 1,700bps by early 2016, dragging down financial sector sentiment. ' +
        'The expected consumer discretionary windfall from cheap petrol never materialised in spending data. ' +
        'Materials companies exposed to industrial metals (copper, iron ore) also fell as the China slowdown that triggered ' +
        'the oil glut hit commodity demand broadly. ' +
        'This backtest asks: does the model reproduce the counterintuitive finding that a large negative energy shock ' +
        'damages financials and fails to lift consumer sectors, rather than delivering the consensus "tax cut" narrative?',
      lambdas: [2.00, 2.00, 1.20, 1.20, 1.00, 1.20, 1.00, 0.80, 1.20, 1.80, 2.50],
      nm_intensity: 0.55,
      narrative_half_life: 12,
      total_ticks: 125,
      num_runs: 50,
      shocks: [
        // Energy: chronic structural oversupply repricing. XLE fell ~50% over 18 months — not a spike then recovery.
        { onset: 40, duration: 55, magnitude: -0.35, type: 'chronic', channel: 'market', reversion: false, sectors: [0] },
        // Materials: industrial metals correlation with China slowdown; mining and chemicals hurt.
        { onset: 40, duration: 40, magnitude: -0.15, type: 'chronic', channel: 'market', reversion: false, sectors: [1] },
        // Financials: HY energy credit exposure + contagion to broader credit markets.
        // Both channels — market repricing of loan book risk + strong bearish narrative.
        // Partial reversion: systemic risk fears proved excessive; most banks survived intact.
        { onset: 40, duration: 25, magnitude: -0.14, type: 'acute',   channel: 'both',   reversion: true,  sectors: [6] },
        // Industrials: oilfield services, pipeline capex cuts, industrial demand slowdown.
        { onset: 40, duration: 30, magnitude: -0.10, type: 'acute',   channel: 'both',   reversion: true,  sectors: [2] },
        // Utilities: mild defensive rotation + lower fuel costs — modest positive influence only.
        { onset: 50, duration: 20, magnitude:  0.06, type: 'acute',   channel: 'influence', reversion: true, sectors: [9] },
      ],
    },

    asia_heatwave: {
      scenario_name: 'Manufacturing Heatwave — Supply Chain Shock',
      scenario_description:
        'A prolonged extreme heat event triggers mandatory power rationing across a major semiconductor and agricultural ' +
        'producing region. Fab operations curtail by 30–40% for eight weeks; automotive parts and apparel manufacturing ' +
        'suspends operations. ' +
        'This is not a demand shock. Consumer spending is unaffected. It is a physical supply-side disruption — ' +
        'the kind that classical economics and most portfolio models treat as a temporary, mean-reverting disruption to be ignored. ' +
        'The consensus reaction is calm: supply chains are resilient, inventories will buffer, this will be forgotten in a quarter. ' +
        'The non-consensus case is more interesting. Semiconductor lead times extend from 12 weeks to 40+ weeks, ' +
        'creating a downstream ripple across every industry that embeds chips — automotive, industrial equipment, medical devices, ' +
        'consumer electronics — that takes 12–18 months to clear. The heat event itself lasts 8 weeks; its supply chain consequences ' +
        'persist for several quarters. Agricultural heat stress hits consumer staples supply chains in ways that are not priced ' +
        'because physical crop disruption is not a standard institutional risk scenario. ' +
        'Energy demand for cooling spikes sharply but briefly. ' +
        'The model is asked to identify whether the downstream supply chain effects from a physical climate event are ' +
        'systematically underpriced relative to the "temporary disruption" consensus framing, ' +
        'and whether Consumer Staples defensive status holds when agricultural supply is the transmission mechanism.',
      lambdas: [1.50, 2.00, 1.20, 1.20, 1.00, 1.20, 0.80, 1.20, 1.20, 1.80, 2.50],
      nm_intensity: 0.60,
      narrative_half_life: 10,
      total_ticks: 250,
      num_runs: 50,
      shocks: [
        // IT: chronic supply constraint — fab curtailment has a long tail (lead times, restocking).
        // Market channel only: it is a fundamental supply shift, not a narrative/sentiment story initially.
        { onset: 40, duration: 40, magnitude: -0.16, type: 'chronic', channel: 'market', reversion: false, sectors: [7] },
        // Consumer Discretionary: acute shock — automotive, electronics supply disruption.
        // Reversion true: demand deferred, not destroyed.
        { onset: 40, duration: 25, magnitude: -0.12, type: 'acute',   channel: 'both',   reversion: true,  sectors: [3] },
        // Consumer Staples + Industrials: agricultural heat stress + parts shortage.
        // Both channels: supply disruption narrative builds; partial reversion as alternative sourcing develops.
        { onset: 40, duration: 30, magnitude: -0.10, type: 'acute',   channel: 'both',   reversion: true,  sectors: [4, 2] },
        // Energy: acute positive — cooling demand spike drives short-term power and fuel consumption surge.
        // Influence channel only: narrative/commodity price driven, not a fundamental equity repricing.
        { onset: 40, duration: 12, magnitude:  0.10, type: 'acute',   channel: 'influence', reversion: true, sectors: [0] },
      ],
    },
  };

  let _activePreset = null;

  function _setActivePreset(key) {
    _activePreset = key;
    document.querySelectorAll('.btn-preset').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('onclick').includes(`'${key}'`));
    });
  }

  function clearPreset() {
    _activePreset = null;
    document.querySelectorAll('.btn-preset').forEach(btn => btn.classList.remove('active'));

    // Scenario tab
    const nameEl = document.getElementById('scenario-name');
    const descEl = document.getElementById('scenario-desc');
    const ctxEl  = document.getElementById('market-context');
    if (nameEl) nameEl.value = '';
    if (descEl) descEl.value = '';
    if (ctxEl)  ctxEl.value  = '';

    // Market tab — reset to defaults
    SECTORS.forEach((s, i) => {
      const el = document.getElementById(`lambda-${s}`);
      if (el) { el.value = LAMBDA_DEFAULTS[i]; _updateLambdaCtx(s); }
    });
    const nmEl = document.getElementById('nm-intensity');
    if (nmEl) nmEl.value = 0.4;
    const halfLifeEl = document.getElementById('narrative-half-life');
    if (halfLifeEl) halfLifeEl.value = 10;

    // Shocks tab — clear all shocks, hide banner, reset text
    document.querySelectorAll('.shock-card[id^="shock-card-"]').forEach(c => c.remove());
    shockCount = 0;
    const shockBanner = document.getElementById('preset-shock-banner');
    if (shockBanner) { shockBanner.style.display = 'none'; shockBanner.innerHTML = ''; }
    const shocksDefaultsText = document.getElementById('shocks-defaults-text');
    if (shocksDefaultsText) shocksDefaultsText.textContent = 'No shocks configured — select a preset or add shocks manually.';

    // Run tab — reset to defaults
    const ticksEl = document.getElementById('total-ticks');
    const runsEl  = document.getElementById('num-runs');
    if (ticksEl) ticksEl.value = 250;
    if (runsEl)  runsEl.value  = 10;
  }

  function loadPreset(key) {
    const p = PRESETS[key];
    if (!p) return;
    _setActivePreset(key);

    // Scenario tab
    const nameEl = document.getElementById('scenario-name');
    const descEl = document.getElementById('scenario-desc');
    if (nameEl) nameEl.value = p.scenario_name;
    if (descEl) descEl.value = p.scenario_description;

    // Market tab — lambdas
    SECTORS.forEach((s, i) => {
      const el = document.getElementById(`lambda-${s}`);
      if (el) { el.value = p.lambdas[i]; _updateLambdaCtx(s, p.lambdas[i]); }
    });
    const nmEl = document.getElementById('nm-intensity');
    if (nmEl) nmEl.value = p.nm_intensity;
    const halfLifeEl = document.getElementById('narrative-half-life');
    if (halfLifeEl) halfLifeEl.value = p.narrative_half_life ?? 10;

    // Shocks tab — show preset banner, update defaults bar, auto-open panel
    const shockBanner = document.getElementById('preset-shock-banner');
    if (shockBanner) {
      shockBanner.innerHTML = `<strong style="color:var(--text-1)">Preloaded shock calibration for: ${p.scenario_name}</strong> — these shocks reflect the scenario design. You can adjust individual parameters or add further shocks below.`;
      shockBanner.style.display = 'block';
    }
    const shocksDefaultsText = document.getElementById('shocks-defaults-text');
    if (shocksDefaultsText) shocksDefaultsText.textContent = `${p.shocks.length} shock${p.shocks.length !== 1 ? 's' : ''} configured for this scenario.`;
    // Shocks tab — clear existing, add preset shocks
    document.querySelectorAll('.shock-card[id^="shock-card-"]').forEach(c => c.remove());
    shockCount = 0;
    p.shocks.forEach(shock => {
      addShock();
      const id = shockCount;
      const onsetEl    = document.getElementById(`shock-onset-${id}`);
      const durEl      = document.getElementById(`shock-duration-${id}`);
      const magEl      = document.getElementById(`shock-magnitude-${id}`);
      const typeEl     = document.getElementById(`shock-type-${id}`);
      const chanEl     = document.getElementById(`shock-channel-${id}`);
      const revEl      = document.getElementById(`shock-reversion-${id}`);
      if (onsetEl) onsetEl.value = shock.onset;
      if (durEl)   durEl.value   = shock.duration;
      if (magEl)   magEl.value   = shock.magnitude;
      if (typeEl)  typeEl.value  = shock.type;
      if (chanEl)  chanEl.value  = shock.channel;
      if (revEl)   revEl.value   = String(shock.reversion);
      // Tick the correct sector checkboxes
      shock.sectors.forEach(idx => {
        const cb = document.querySelector(`input[name="shock-sectors-${id}"][value="${idx}"]`);
        if (cb) cb.checked = true;
      });
    });

    // Run tab
    const ticksEl = document.getElementById('total-ticks');
    const runsEl  = document.getElementById('num-runs');
    if (ticksEl) ticksEl.value = p.total_ticks;
    if (runsEl)  runsEl.value  = p.num_runs;
  }

  return { init, addShock, removeShock, onSeedPolicyChange, buildCalibration, loadPreset, clearPreset, updateLambdaCtx: _updateLambdaCtx };
})();

// Wire up the "Add shock" button.
document.getElementById('add-shock-btn')?.addEventListener('click', () => setup.addShock());
