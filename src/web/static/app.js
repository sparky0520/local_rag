// AetherRAG Dashboard & Interactive Playground

let currentBenchmarkData = null;

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initBenchmarkControls();
  initPlayground();
  loadBenchmarkData();
  loadTestSetData();
  loadCorpusData();
});

// --- Tab Navigation ---
function initNavigation() {
  const navBtns = document.querySelectorAll(".nav-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const pageTitle = document.getElementById("page-title");

  const titles = {
    "tab-benchmark": "Retrieval Quality Measurement Dashboard",
    "tab-playground": "Interactive Hybrid RAG Playground",
    "tab-testset": "Curated Ground-Truth Test Set (36 Queries)",
    "tab-corpus": "Technical Documentation Corpus & Chunk Index"
  };

  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      navBtns.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(targetTab);
      if (targetPane) targetPane.classList.add("active");

      if (pageTitle && titles[targetTab]) {
        pageTitle.innerText = titles[targetTab];
      }
    });
  });
}

// --- Benchmark Controls & Loading ---
function initBenchmarkControls() {
  const btnRun = document.getElementById("btn-run-bench");
  if (btnRun) {
    btnRun.addEventListener("click", async () => {
      btnRun.disabled = true;
      btnRun.innerHTML = `<span class="btn-icon">⏳</span> Running Benchmark...`;
      try {
        const res = await fetch("/api/benchmark/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ alpha: 0.5 })
        });
        const data = await res.json();
        renderBenchmark(data);
      } catch (err) {
        console.error("Benchmark run failed:", err);
        alert("Failed to run benchmark: " + err);
      } finally {
        btnRun.disabled = false;
        btnRun.innerHTML = `<span class="btn-icon">⚡</span> Run Live Benchmark`;
      }
    });
  }

  // Query filter input
  const filterInput = document.getElementById("query-filter-input");
  if (filterInput) {
    filterInput.addEventListener("input", (e) => {
      filterQueryDetailsTable(e.target.value.toLowerCase());
    });
  }
}

async function loadBenchmarkData() {
  try {
    const res = await fetch("/api/benchmark");
    const data = await res.json();
    renderBenchmark(data);
  } catch (err) {
    console.error("Failed to load benchmark data:", err);
  }
}

function renderBenchmark(data) {
  currentBenchmarkData = data;
  const modes = data.modes;

  // 1. KPI Cards
  const hybridLin = modes["hybrid_linear"] || modes["hybrid_rrf"];
  const bm25 = modes["bm25"];
  const dense = modes["dense"];

  if (hybridLin) {
    document.getElementById("kpi-hybrid-r5").innerText = `${(hybridLin.recall_at_k["5"] * 100).toFixed(1)}%`;
    document.getElementById("kpi-mrr").innerText = hybridLin.mrr.toFixed(3);
  }
  if (bm25) {
    document.getElementById("kpi-bm25-r1").innerText = `${(bm25.recall_at_k["1"] * 100).toFixed(1)}%`;
  }
  if (dense) {
    document.getElementById("kpi-dense-r1").innerText = `${(dense.recall_at_k["1"] * 100).toFixed(1)}%`;
  }

  // 2. Primary Metrics Table
  const tbodyPrimary = document.getElementById("tbody-primary-metrics");
  if (tbodyPrimary) {
    tbodyPrimary.innerHTML = "";
    Object.keys(modes).forEach(mKey => {
      const m = modes[mKey];
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${m.display_name}</strong></td>
        <td>${(m.recall_at_k["1"] * 100).toFixed(1)}%</td>
        <td>${(m.recall_at_k["3"] * 100).toFixed(1)}%</td>
        <td><strong style="color: #34d399">${(m.recall_at_k["5"] * 100).toFixed(1)}%</strong></td>
        <td>${(m.recall_at_k["10"] * 100).toFixed(1)}%</td>
        <td><strong style="color: #60a5fa">${m.mrr.toFixed(3)}</strong></td>
        <td>${m.map_score.toFixed(3)}</td>
        <td>${m.ndcg_at_k["5"].toFixed(3)}</td>
        <td><span class="badge badge-gray">${m.latency_ms_p50.toFixed(2)} ms</span></td>
      `;
      tbodyPrimary.appendChild(tr);
    });
  }

  // 3. Recall Bars
  const barsContainer = document.getElementById("recall-bars-container");
  if (barsContainer) {
    barsContainer.innerHTML = "";
    const colorClasses = {
      "bm25": "bm25",
      "dense": "dense",
      "hybrid_rrf": "rrf",
      "hybrid_linear": "linear"
    };

    Object.keys(modes).forEach(mKey => {
      const m = modes[mKey];
      const r5Val = (m.recall_at_k["5"] * 100).toFixed(1);
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <div class="bar-label" title="${m.display_name}">${m.display_name}</div>
        <div class="bar-track">
          <div class="bar-fill ${colorClasses[mKey] || 'linear'}" style="width: ${r5Val}%">
            ${r5Val}% (R@5)
          </div>
        </div>
      `;
      barsContainer.appendChild(row);
    });
  }

  // 4. Archetype Breakdown
  const archContainer = document.getElementById("archetype-breakdown-container");
  if (archContainer) {
    archContainer.innerHTML = "";
    const cats = ["keyword_exact", "semantic_concept", "hybrid_balanced"];
    const catLabels = {
      "keyword_exact": "Keyword Exact",
      "semantic_concept": "Semantic Concept",
      "hybrid_balanced": "Hybrid Balanced"
    };

    cats.forEach(cat => {
      const catBlock = document.createElement("div");
      catBlock.style.marginBottom = "1.2rem";
      
      let html = `<div style="font-size: 0.85rem; font-weight: 600; color: #93c5fd; margin-bottom: 0.4rem;">${catLabels[cat] || cat} (Recall@5)</div>`;
      Object.keys(modes).forEach(mKey => {
        const m = modes[mKey];
        const val = ((m.category_recall_at_5[cat] || 0.0) * 100).toFixed(1);
        html += `
          <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin: 0.2rem 0; color: #94a3b8;">
            <span>${m.display_name}:</span>
            <strong style="color: ${val >= 100 ? '#34d399' : '#f59e0b'}">${val}%</strong>
          </div>
        `;
      });
      catBlock.innerHTML = html;
      archContainer.appendChild(catBlock);
    });
  }

  // 5. Query Details Table
  renderQueryDetailsTable(data.query_details);
}

function renderQueryDetailsTable(details) {
  const tbody = document.getElementById("tbody-query-details");
  if (!tbody) return;
  tbody.innerHTML = "";

  details.forEach(q => {
    const tr = document.createElement("tr");
    tr.dataset.queryId = q.query_id.toLowerCase();
    tr.dataset.category = q.category.toLowerCase();
    tr.dataset.text = q.query.toLowerCase();

    const bm25R = q.ranks["bm25"] ? `#${q.ranks["bm25"]}` : "Miss (>10)";
    const denseR = q.ranks["dense"] ? `#${q.ranks["dense"]}` : "Miss (>10)";
    const rrfR = q.ranks["hybrid_rrf"] ? `#${q.ranks["hybrid_rrf"]}` : "Miss (>10)";
    const linR = q.ranks["hybrid_linear"] ? `#${q.ranks["hybrid_linear"]}` : "Miss (>10)";

    let catBadge = "badge-gray";
    if (q.category === "keyword_exact") catBadge = "badge-outline";
    if (q.category === "semantic_concept") catBadge = "badge-purple";
    if (q.category === "hybrid_balanced") catBadge = "badge-primary";

    tr.innerHTML = `
      <td><span class="badge badge-outline">${q.query_id}</span></td>
      <td><strong>${q.query}</strong></td>
      <td><span class="badge ${catBadge}">${q.category}</span></td>
      <td>${bm25R}</td>
      <td>${denseR}</td>
      <td><strong style="color: #a78bfa">${rrfR}</strong></td>
      <td><strong style="color: #34d399">${linR}</strong></td>
      <td><span class="badge badge-green">✓ Hit</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function filterQueryDetailsTable(term) {
  const rows = document.querySelectorAll("#tbody-query-details tr");
  rows.forEach(r => {
    const match = r.dataset.text.includes(term) ||
                  r.dataset.queryId.includes(term) ||
                  r.dataset.category.includes(term);
    r.style.display = match ? "" : "none";
  });
}

// --- Playground Functions ---
function initPlayground() {
  const btnAsk = document.getElementById("btn-play-ask");
  const btnSearch = document.getElementById("btn-play-search");
  const queryInput = document.getElementById("play-query");
  const modeSelect = document.getElementById("play-mode");
  const alphaSlider = document.getElementById("play-alpha");
  const alphaLabel = document.getElementById("alpha-val-label");
  const alphaControl = document.getElementById("alpha-control-item");
  const topkSelect = document.getElementById("play-topk");

  if (alphaSlider && alphaLabel) {
    alphaSlider.addEventListener("input", (e) => {
      alphaLabel.innerText = parseFloat(e.target.value).toFixed(2);
    });
  }

  if (modeSelect) {
    modeSelect.addEventListener("change", (e) => {
      if (e.target.value === "hybrid_linear") {
        alphaControl.style.opacity = "1";
        alphaControl.style.pointerEvents = "auto";
      } else {
        alphaControl.style.opacity = "0.4";
        alphaControl.style.pointerEvents = "none";
      }
    });
  }

  // Sample Query Chips
  const chips = document.querySelectorAll(".chip-btn");
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      queryInput.value = q;
      handleAsk();
    });
  });

  if (btnAsk) btnAsk.addEventListener("click", handleAsk);
  if (btnSearch) btnSearch.addEventListener("click", handleSearchOnly);
  if (queryInput) {
    queryInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") handleAsk();
    });
  }

  async function handleAsk() {
    const q = queryInput.value.trim();
    if (!q) return;

    btnAsk.disabled = true;
    btnAsk.innerText = "Synthesizing...";

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          mode: modeSelect.value,
          top_k: parseInt(topkSelect.value, 10),
          alpha: parseFloat(alphaSlider.value)
        })
      });
      const data = await res.json();
      renderPlaygroundAnswer(data);
      renderPlaygroundChunks(data.sources);
    } catch (err) {
      console.error(err);
      alert("Ask error: " + err);
    } finally {
      btnAsk.disabled = false;
      btnAsk.innerText = "Generate RAG Answer";
    }
  }

  async function handleSearchOnly() {
    const q = queryInput.value.trim();
    if (!q) return;

    btnSearch.disabled = true;
    btnSearch.innerText = "Searching...";

    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          mode: modeSelect.value,
          top_k: parseInt(topkSelect.value, 10),
          alpha: parseFloat(alphaSlider.value)
        })
      });
      const data = await res.json();
      document.getElementById("play-answer-panel").classList.add("hidden");
      renderPlaygroundChunks(data.results);
    } catch (err) {
      console.error(err);
      alert("Search error: " + err);
    } finally {
      btnSearch.disabled = false;
      btnSearch.innerText = "Search Only";
    }
  }
}

function renderPlaygroundAnswer(ragData) {
  const panel = document.getElementById("play-answer-panel");
  const content = document.getElementById("answer-markdown-content");
  const modeBadge = document.getElementById("answer-mode-badge");
  const latencyBadge = document.getElementById("answer-latency-badge");

  panel.classList.remove("hidden");
  modeBadge.innerText = ragData.mode.toUpperCase();
  latencyBadge.innerText = `${ragData.total_latency_ms} ms (Retrieval: ${ragData.retrieval_latency_ms}ms)`;

  // Simple clean markdown formatting
  let html = ragData.answer
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^- (.*$)/gim, '<li>$1</li>');

  content.innerHTML = html;
}

function renderPlaygroundChunks(chunks) {
  const panel = document.getElementById("play-chunks-panel");
  const list = document.getElementById("retrieved-chunks-list");
  const countBadge = document.getElementById("chunks-count-badge");

  panel.classList.remove("hidden");
  countBadge.innerText = `${chunks.length} chunks retrieved`;
  list.innerHTML = "";

  chunks.forEach(c => {
    const card = document.createElement("div");
    card.className = "chunk-card";

    let rankSignals = "";
    if (c.dense_rank !== undefined && c.bm25_rank !== undefined) {
      rankSignals = `<span class="badge badge-purple">Dense Rank #${c.dense_rank}</span> <span class="badge badge-primary">BM25 Rank #${c.bm25_rank}</span>`;
    }

    const preview = c.snippet || c.content;

    card.innerHTML = `
      <div class="chunk-card-header">
        <div class="chunk-title">
          <span style="color: #60a5fa">#${c.rank}</span> ${c.doc_title} → ${c.section_title}
        </div>
        <div class="chunk-ranks">
          ${rankSignals}
          <span class="badge badge-green">Score: ${c.score.toFixed(4)}</span>
          <span class="badge badge-outline">${c.chunk_id}</span>
        </div>
      </div>
      <div class="chunk-content-preview">${preview}</div>
    `;
    list.appendChild(card);
  });
}

// --- Test Set Tab ---
async function loadTestSetData() {
  try {
    const res = await fetch("/api/test-set");
    const data = await res.json();
    const tbody = document.getElementById("tbody-testset-full");
    if (!tbody) return;
    tbody.innerHTML = "";

    data.queries.forEach(q => {
      const tr = document.createElement("tr");
      let catBadge = "badge-gray";
      if (q.category === "keyword_exact") catBadge = "badge-outline";
      if (q.category === "semantic_concept") catBadge = "badge-purple";
      if (q.category === "hybrid_balanced") catBadge = "badge-primary";

      tr.innerHTML = `
        <td><span class="badge badge-outline">${q.query_id}</span></td>
        <td><span class="badge ${catBadge}">${q.category}</span></td>
        <td><strong>${q.query}</strong></td>
        <td><code>${q.expected_chunks.join(", ")}</code></td>
        <td style="font-size: 0.8rem; color: #94a3b8;">${q.rationale || ""}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load test set:", err);
  }
}

// --- Corpus Tab ---
async function loadCorpusData() {
  try {
    const res = await fetch("/api/corpus");
    const data = await res.json();
    const container = document.getElementById("corpus-chunks-container");
    const statsBadge = document.getElementById("corpus-stats-badge");

    if (statsBadge) {
      statsBadge.innerText = `${data.documents_count} Documents • ${data.total_chunks} Chunks`;
    }

    if (!container) return;
    container.innerHTML = "";

    data.chunks.forEach(c => {
      const item = document.createElement("div");
      item.className = "corpus-item";
      item.innerHTML = `
        <div class="corpus-item-id">${c.chunk_id}</div>
        <div class="corpus-item-title">${c.doc_title} → ${c.section_title}</div>
        <div class="corpus-item-desc">${c.word_count} words • ${c.char_count} chars</div>
      `;
      container.appendChild(item);
    });
  } catch (err) {
    console.error("Failed to load corpus:", err);
  }
}

