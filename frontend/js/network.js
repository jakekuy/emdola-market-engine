/**
 * network.js — D3 SVG network visualisation (centrepiece of the runtime view).
 *
 * 8 agent type cards (retail left, institutional right) connected by flow lines
 * to a central MARKET node.
 *
 * Particles travel agent → market (buying) or market → agent (selling).
 * Particle colour = the agent type's own colour.
 * Buy/sell bars on each card show live demand composition.
 * Direction carries meaning; colour identifies the trading type.
 */
const network = (() => {
  const TYPE_COLORS = {
    R1:'#6a1422', R2:'#b83030', R3:'#e04040', R4:'#d86020',
    I1:'#7a5808', I2:'#b88010', I3:'#d4a80e', I4:'#ecc820',
  };

  // Short sector names for floating labels (index matches SECTORS list in sector_config.py)
  const SECTOR_SHORT = [
    'Energy','Materials','Industrials','Con. Discr.','Con. Staples',
    'Health Care','Financials','Info Tech','Comms','Utilities','Real Estate',
  ];
  const AGENT_TYPES = ['R1','R2','R3','R4','I1','I2','I3','I4'];
  const TYPE_LABELS = {
    R1:'Passive',     R2:'Self-directed', R3:'HNW',     R4:'Speculative',
    I1:'Asset mgr',   I2:'Hedge fund',    I3:'Pension',  I4:'Sovereign',
  };

  // Full type labels for the merged card display.
  const TYPE_FULL = {
    R1:'Passive retail',     R2:'Self-directed',  R3:'HNW retail',    R4:'Speculative',
    I1:'Asset manager',      I2:'Hedge fund',     I3:'Pension fund',   I4:'Sovereign WF',
  };

  // ── Layout constants ───────────────────────────────────────────────────────
  const CW = 200, CH = 160, CR = 5;  // card width, height, corner radius
  const XCOL_OUT = 240;               // x-offset for outer (top/bottom) cards
  const XCOL_IN  = 320;               // x-offset for inner (middle) cards — pushed out for arc
  const YSPC = 172;                   // vertical spacing: MUST stay ≥ CH to guarantee zero card overlap
  const MARKET_R = 55;               // market node radius

  // Card centres (relative to g-group centre).
  // YSPC >= CH so adjacent vertical neighbours never overlap.
  // Middle nodes pushed out further (XCOL_IN > XCOL_OUT) for arc.
  // Bottom nodes 1.65× to fill dead space below.
  const NODE_POSITIONS = {
    R1: [-XCOL_OUT, -YSPC * 1.5],  R2: [-XCOL_IN, -YSPC * 0.5],
    R3: [-XCOL_IN,   YSPC * 0.5],  R4: [-XCOL_OUT,  YSPC * 1.65],
    I1: [ XCOL_OUT, -YSPC * 1.5],  I2: [ XCOL_IN, -YSPC * 0.5],
    I3: [ XCOL_IN,   YSPC * 0.5],  I4: [ XCOL_OUT,  YSPC * 1.65],
  };
  const MARKET_POS = [0, 0];

  // ── State ──────────────────────────────────────────────────────────────────
  let svg, g, linesG, marketG, cardsG, particlesG;
  const width = 880, height = 780;
  let shockPulseActive = false;

  const typeState = {};
  AGENT_TYPES.forEach(t => {
    typeState[t] = { activityRatio: 0, buy: 0, sell: 0, topSector: -1 };
  });

  // ── init ───────────────────────────────────────────────────────────────────

  function init(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    svg = d3.select('#network-svg');
    svg.selectAll('*').remove();   // clean slate on each run

    svg.attr('viewBox', `0 0 ${width} ${height}`)
       .attr('preserveAspectRatio', 'xMidYMid meet');

    const defs = svg.append('defs');

    // Glow — market node
    const glow = defs.append('filter').attr('id', 'mkt-glow')
      .attr('x', '-60%').attr('y', '-60%').attr('width', '220%').attr('height', '220%');
    glow.append('feGaussianBlur').attr('stdDeviation', 6).attr('result', 'blur');
    const fm1 = glow.append('feMerge');
    fm1.append('feMergeNode').attr('in', 'blur');
    fm1.append('feMergeNode').attr('in', 'SourceGraphic');

    // Main group centred.
    g = svg.append('g');

    linesG     = g.append('g').attr('class', 'net-lines');
    marketG    = g.append('g').attr('class', 'net-market');
    cardsG     = g.append('g').attr('class', 'net-cards');
    particlesG = g.append('g').attr('class', 'net-particles');

    // Pinch-to-zoom and pan — desktop only.
    // On mobile the SVG is non-interactive so single-finger scroll reaches the page.
    const isMobile = window.matchMedia('(max-width: 768px)').matches;
    if (isMobile) {
      g.attr('transform', `translate(${width / 2}, ${height / 2})`);
      svg.node().style.pointerEvents = 'none';
    } else {
      const zoom = d3.zoom()
        .scaleExtent([0.25, 4])
        .on('zoom', (event) => { g.attr('transform', event.transform); });
      svg.call(zoom);
      svg.call(zoom.transform, d3.zoomIdentity.translate(width / 2, height / 2));
    }

    _drawClusterLabels();
    _drawLines();
    _drawMarket();
    _drawCards();
  }

  // ── Draw helpers ───────────────────────────────────────────────────────────

  function _drawClusterLabels() {
    const y = -YSPC * 1.5 - CH / 2 - 14;
    [['RETAIL', -XCOL_OUT], ['INSTITUTIONAL', XCOL_OUT]].forEach(([text, cx]) => {
      g.append('text')
        .attr('x', cx).attr('y', y)
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.18)')
        .attr('font-family', "'SF Mono','Fira Code',monospace")
        .attr('font-size', 8)
        .attr('letter-spacing', '0.12em')
        .text(text);
    });
  }

  function _drawLines() {
    AGENT_TYPES.forEach(type => {
      const [sx, sy] = NODE_POSITIONS[type];
      const [tx, ty] = MARKET_POS;
      linesG.append('line')
        .attr('class', `flow-line flow-line-${type}`)
        .attr('x1', sx).attr('y1', sy)
        .attr('x2', tx).attr('y2', ty)
        .attr('stroke', 'rgba(255,255,255,0.05)')
        .attr('stroke-width', 0.8);
    });
  }

  function _drawMarket() {
    const [cx, cy] = MARKET_POS;

    // Outer pulse ring
    marketG.append('circle')
      .attr('class', 'market-ring')
      .attr('cx', cx).attr('cy', cy)
      .attr('r', MARKET_R + 10)
      .attr('fill', 'none')
      .attr('stroke', 'rgba(212,168,14,0.12)')
      .attr('stroke-width', 1);

    // Main node
    marketG.append('circle')
      .attr('class', 'market-circle')
      .attr('cx', cx).attr('cy', cy)
      .attr('r', MARKET_R)
      .attr('fill', 'rgba(212,168,14,0.07)')
      .attr('stroke', 'rgba(212,168,14,0.45)')
      .attr('stroke-width', 1.5)
      .attr('filter', 'url(#mkt-glow)');

    marketG.append('text')
      .attr('x', cx).attr('y', cy - 6)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(212,168,14,0.80)')
      .attr('font-family', "'SF Mono','Fira Code',monospace")
      .attr('font-size', 13).attr('font-weight', '700')
      .attr('letter-spacing', '0.12em')
      .text('MARKET');

    marketG.append('text')
      .attr('class', 'market-sub')
      .attr('x', cx).attr('y', cy + 12)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.30)')
      .attr('font-family', "'SF Mono','Fira Code',monospace")
      .attr('font-size', 9)
      .text('');
  }

  function _drawCards() {
    const BAR_W = CW - 16;
    const BAR_H = 4;
    const HCH = CH / 2;
    // Vertical layout (y relative to card centre), CH=160 → HCH=80:
    const Y_HEADER    = -HCH + 18;  // type code + "view →" hint
    const Y_SEP       = -HCH + 29;  // thin separator line
    const Y_LABEL     = -HCH + 46;  // full type label
    const Y_SECTOR    = -HCH + 67;  // GICS sector indicator (updated each tick)
    const Y_PCT       = -HCH + 88;  // "X% engaging this tick"
    const Y_BS        = -HCH + 108; // "↑ N buying · ↓ N selling"
    const BAR_Y       =  HCH - 13;  // buy/sell bar

    AGENT_TYPES.forEach(type => {
      const [cx, cy] = NODE_POSITIONS[type];
      const color = TYPE_COLORS[type];

      const cg = cardsG.append('g')
        .attr('class', `card-g card-g-${type}`)
        .attr('transform', `translate(${cx},${cy})`)
        .attr('cursor', 'pointer')
        .on('click', () => runtime.highlightType(type))
        .on('mouseenter', function() {
          d3.select(this).raise();
          d3.select(this).transition().duration(150)
            .attr('transform', `translate(${cx},${cy}) scale(1.7)`);
        })
        .on('mouseleave', function() {
          d3.select(this).transition().duration(200)
            .attr('transform', `translate(${cx},${cy})`);
        });

      // Card background
      cg.append('rect')
        .attr('class', `card-bg card-bg-${type}`)
        .attr('x', -CW / 2).attr('y', -CH / 2)
        .attr('width', CW).attr('height', CH)
        .attr('rx', CR)
        .attr('fill', 'rgba(16,16,20,0.92)')
        .attr('stroke', color)
        .attr('stroke-width', 1)
        .attr('stroke-opacity', 0.35);

      // Header: type code (left) + "view →" hint (right)
      cg.append('text')
        .attr('x', -CW / 2 + 10).attr('y', Y_HEADER)
        .attr('text-anchor', 'start')
        .attr('fill', color)
        .attr('font-family', "'SF Mono','Fira Code',monospace")
        .attr('font-size', 15).attr('font-weight', '700')
        .text(type);

      cg.append('text')
        .attr('x', CW / 2 - 8).attr('y', Y_HEADER)
        .attr('text-anchor', 'end')
        .attr('fill', 'rgba(255,255,255,0.55)')
        .attr('font-family', "'SF Mono','Fira Code',monospace")
        .attr('font-size', 8)
        .text('view archetypes →');

      // Separator line below header
      cg.append('line')
        .attr('x1', -CW / 2 + 5).attr('x2', CW / 2 - 5)
        .attr('y1', Y_SEP).attr('y2', Y_SEP)
        .attr('stroke', 'rgba(255,255,255,0.08)')
        .attr('stroke-width', 0.5);

      // Full type label
      cg.append('text')
        .attr('x', -CW / 2 + 10).attr('y', Y_LABEL)
        .attr('text-anchor', 'start')
        .attr('fill', 'rgba(255,255,255,0.30)')
        .attr('font-family', "'SF Mono','Fira Code',monospace")
        .attr('font-size', 9.5)
        .text(TYPE_FULL[type]);

      // GICS sector indicator — updated each tick (shows top-traded sector)
      cg.append('text')
        .attr('class', `card-sector card-sector-${type}`)
        .attr('x', -CW / 2 + 10).attr('y', Y_SECTOR)
        .attr('text-anchor', 'start')
        .attr('fill', 'rgba(255,255,255,0.20)')
        .attr('font-family', "'SF Mono','Fira Code',monospace")
        .attr('font-size', 10)
        .text('');

      // Activity % stat — updated each tick
      cg.append('text')
        .attr('class', `card-pct card-pct-${type}`)
        .attr('x', -CW / 2 + 10).attr('y', Y_PCT)
        .attr('text-anchor', 'start')
        .attr('fill', 'rgba(255,255,255,0.22)')
        .attr('font-family', "'SF Mono','Fira Code',monospace")
        .attr('font-size', 10)
        .text('—');

      // Buy / sell count — updated each tick
      cg.append('text')
        .attr('class', `card-bs card-bs-${type}`)
        .attr('x', -CW / 2 + 10).attr('y', Y_BS)
        .attr('text-anchor', 'start')
        .attr('fill', 'rgba(255,255,255,0.22)')
        .attr('font-family', "'SF Mono','Fira Code',monospace")
        .attr('font-size', 9.5)
        .text('');

      // Bar track (background)
      cg.append('rect')
        .attr('x', -BAR_W / 2).attr('y', BAR_Y - BAR_H / 2)
        .attr('width', BAR_W).attr('height', BAR_H)
        .attr('rx', 1)
        .attr('fill', 'rgba(255,255,255,0.04)');

      // Buy segment — grows rightward from left
      cg.append('rect')
        .attr('class', `bar-buy bar-buy-${type}`)
        .attr('x', -BAR_W / 2).attr('y', BAR_Y - BAR_H / 2)
        .attr('width', 0).attr('height', BAR_H)
        .attr('rx', 1)
        .attr('fill', 'rgba(212,168,14,0.75)');

      // Sell segment — grows leftward from right
      cg.append('rect')
        .attr('class', `bar-sell bar-sell-${type}`)
        .attr('x', BAR_W / 2).attr('y', BAR_Y - BAR_H / 2)
        .attr('width', 0).attr('height', BAR_H)
        .attr('rx', 1)
        .attr('fill', 'rgba(148,48,64,0.8)');
    });
  }

  // ── Update (called each tick from runtime.js) ──────────────────────────────

  function update(data) {
    const activity = data.type_activity || {};
    const shockActive = data.shock_active || false;

    AGENT_TYPES.forEach(t => {
      const a = activity[t];
      if (!a) return;
      typeState[t].activityRatio = a.activity_ratio || 0;
      typeState[t].buy       = a.buy  || 0;
      typeState[t].sell      = a.sell || 0;
      typeState[t].topSector = (a.top_sector !== undefined) ? a.top_sector : -1;
    });

    _updateCards();
    _updateLines();
    _updateMarket();
    _spawnParticles(activity);

    if (shockActive && !shockPulseActive) _triggerShockPulse();
    shockPulseActive = shockActive;
  }

  function _updateCards() {
    const BAR_W = CW - 14;

    AGENT_TYPES.forEach(type => {
      const s = typeState[type];
      const total = s.buy + s.sell;
      const active = s.activityRatio > 0.04;
      const scaledRatio = Math.min(s.activityRatio * 2.5, 1);

      // Card border brightness
      d3.select(`.card-bg-${type}`)
        .transition().duration(180)
        .attr('stroke-opacity', active ? 0.85 : 0.35)
        .attr('stroke-width',   active ? 1.5  : 1);

      // GICS sector indicator — shows top-traded sector this tick
      const sectorLabel = (active && s.topSector >= 0)
        ? `\u25b8 ${SECTOR_SHORT[s.topSector]}`
        : '';
      d3.select(`.card-sector-${type}`)
        .text(sectorLabel)
        .attr('fill', active ? 'rgba(212,168,14,0.80)' : 'rgba(255,255,255,0.20)');

      // Activity % line
      const pct = (s.activityRatio * 100).toFixed(0);
      d3.select(`.card-pct-${type}`)
        .text(active ? `${pct}% active this tick` : '—')
        .attr('fill', active ? 'rgba(255,255,255,0.78)' : 'rgba(255,255,255,0.22)');

      // Buy / sell count line
      d3.select(`.card-bs-${type}`)
        .text(active ? `\u2191 ${s.buy} buying  \u00b7  \u2193 ${s.sell} selling` : '');

      // Buy/sell bars
      const buyFrac  = total > 0 ? s.buy  / total : 0;
      const sellFrac = total > 0 ? s.sell / total : 0;
      const buyW  = buyFrac  * BAR_W * scaledRatio;
      const sellW = sellFrac * BAR_W * scaledRatio;

      d3.select(`.bar-buy-${type}`)
        .transition().duration(180)
        .attr('width', Math.max(0, buyW));

      d3.select(`.bar-sell-${type}`)
        .transition().duration(180)
        .attr('x', BAR_W / 2 - Math.max(0, sellW))
        .attr('width', Math.max(0, sellW));
    });
  }

  function _updateLines() {
    AGENT_TYPES.forEach(type => {
      const s = typeState[type];
      const active = s.activityRatio > 0.04;
      const netBuying = s.buy >= s.sell;
      d3.select(`.flow-line-${type}`)
        .transition().duration(250)
        .attr('stroke', active
          ? (netBuying
              ? `rgba(212,168,14,${0.12 + s.activityRatio * 0.25})`
              : `rgba(148,48,64,${0.12 + s.activityRatio * 0.25})`)
          : 'rgba(255,255,255,0.04)')
        .attr('stroke-width', active ? 1 + s.activityRatio * 1.5 : 0.8);
    });
  }

  function _updateMarket() {
    const totalAR   = AGENT_TYPES.reduce((s, t) => s + typeState[t].activityRatio, 0);
    const avgAR     = totalAR / AGENT_TYPES.length;
    const netBuyAll = AGENT_TYPES.reduce((s, t) => s + typeState[t].buy, 0);
    const netSelAll = AGENT_TYPES.reduce((s, t) => s + typeState[t].sell, 0);

    d3.select('.market-circle')
      .transition().duration(280)
      .attr('fill', `rgba(212,168,14,${0.05 + avgAR * 0.18})`);

    d3.select('.market-ring')
      .transition().duration(280)
      .attr('r', MARKET_R + 10 + avgAR * 6)
      .attr('stroke-opacity', 0.08 + avgAR * 0.45);

    const sub = avgAR > 0.04
      ? (netBuyAll > netSelAll ? '▲ net demand' : '▼ net supply')
      : '';
    d3.select('.market-sub').text(sub);
  }

  function _spawnParticles(activity) {
    const [mx, my] = MARKET_POS;

    AGENT_TYPES.forEach(type => {
      const a = activity[type];
      if (!a || (a.activity_ratio || 0) < 0.08) return;  // only on actual trades

      const [ax, ay] = NODE_POSITIONS[type];
      const netBuying = (a.buy || 0) >= (a.sell || 0);
      const color = TYPE_COLORS[type];

      // Direction encodes buy vs sell:
      //   buying  → agent sends demand INTO the market (agent → market)
      //   selling → agent draws supply OUT of the market (market → agent)
      const [x1, y1] = netBuying ? [ax, ay] : [mx, my];
      const [x2, y2] = netBuying ? [mx, my] : [ax, ay];

      const count = Math.max(1, Math.round(a.activity_ratio * 2.5));

      for (let i = 0; i < count; i++) {
        particlesG.append('circle')
          .attr('r', 2.5)
          .attr('cx', x1).attr('cy', y1)
          .attr('fill', color)
          .attr('opacity', 0.8)
          .attr('pointer-events', 'none')
          .transition()
          .duration(440)
          .delay(i * 55)
          .ease(d3.easeCubicIn)
          .attr('cx', x2).attr('cy', y2)
          .attr('r', 1.5)
          .attr('opacity', 0)
          .remove();
      }

    });
  }

  function _triggerShockPulse() {
    const [cx, cy] = MARKET_POS;
    particlesG.append('circle')
      .attr('cx', cx).attr('cy', cy)
      .attr('r', MARKET_R + 4)
      .attr('fill', 'none')
      .attr('stroke', '#943040')
      .attr('stroke-width', 2)
      .attr('opacity', 0.7)
      .transition()
      .duration(850)
      .ease(d3.easeCubicOut)
      .attr('r', 230)
      .attr('opacity', 0)
      .remove();
  }

  return { init, update };
})();
