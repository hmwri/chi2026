const state = {
  papers: [],
  coords: null,
  projection: null,
  chart: null,
  worker: null,
  assetVersion: "",
  requestSeq: 0,
  currentResults: [],
  currentCenter: null,
  currentMode: "query"
};

const els = {
  form: document.querySelector("#searchForm"),
  query: document.querySelector("#query"),
  button: document.querySelector("#submit"),
  status: document.querySelector("#status"),
  detail: document.querySelector("#detail"),
  results: document.querySelector("#results"),
  chart: document.querySelector("#chart")
};

function assetUrl(name, version = state.assetVersion) {
  const url = new URL(`assets/${name}`, window.location.href);
  if (version) url.searchParams.set("v", version);
  return url.toString();
}

function apiBasePath() {
  return window.location.pathname.replace(/\/$/, "");
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function fetchFloat32(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(res.statusText);
  return new Float32Array(await res.arrayBuffer());
}

function escapeHtml(value) {
  return String(value || "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#039;"
  }[ch]));
}

function snippet(value, maxLen = 240) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).replace(/\s+\S*$/, "") + "...";
}

function paperAt(index, score) {
  const paper = state.papers[index];
  return {
    ...paper,
    index,
    score,
    x: state.coords[index * 2],
    y: state.coords[index * 2 + 1],
    snippet: snippet(paper.abstract),
    snippet_ja: snippet(paper.abstract_ja)
  };
}

function queryPointFromEmbedding(vector) {
  const mean = state.projection.mean;
  const components = state.projection.components;
  let x = 0;
  let y = 0;
  for (let i = 0; i < vector.length; i++) {
    const centered = vector[i] - mean[i];
    x += centered * components[0][i];
    y += centered * components[1][i];
  }
  return { x, y };
}

function workerRequest(type, payload) {
  const requestId = ++state.requestSeq;
  return new Promise(resolve => {
    const handler = event => {
      if (event.data.type === "results" && event.data.requestId === requestId) {
        state.worker.removeEventListener("message", handler);
        resolve(event.data.results);
      }
    };
    state.worker.addEventListener("message", handler);
    state.worker.postMessage({ type, requestId, ...payload });
  });
}

async function initWorker(embeddings) {
  const worker = new Worker(assetUrl("visual_worker.js"));
  state.worker = worker;
  await new Promise(resolve => {
    worker.addEventListener("message", event => {
      if (event.data.type === "ready") resolve();
    }, { once: true });
    worker.postMessage({
      type: "init",
      buffer: embeddings.buffer,
      count: state.papers.length,
      dims: state.projection.dimensions
    }, [embeddings.buffer]);
  });
}

function chartOption(center, results) {
  const values = [center, ...results];
  const xs = values.map(item => item.x);
  const ys = values.map(item => item.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const padX = Math.max((maxX - minX) * 0.24, 0.01);
  const padY = Math.max((maxY - minY) * 0.24, 0.01);

  return {
    animationDuration: 350,
    grid: { left: 28, right: 28, top: 28, bottom: 28 },
    xAxis: { min: minX - padX, max: maxX + padX, show: false },
    yAxis: { min: minY - padY, max: maxY + padY, show: false },
    tooltip: {
      trigger: "item",
      confine: true,
      formatter: params => {
        if (params.seriesName === "center") return "query / selected paper";
        const item = params.data.paper;
        return `<strong>${escapeHtml(item.title_ja || item.title)}</strong><br>${escapeHtml(item.content_type || "")}<br>score ${Number(item.score).toFixed(3)}`;
      }
    },
    series: [
      {
        name: "links",
        type: "lines",
        coordinateSystem: "cartesian2d",
        silent: true,
        lineStyle: { color: "#9ea7b3", width: 1.5, type: "dashed", opacity: 0.58 },
        data: results.map(item => ({ coords: [[center.x, center.y], [item.x, item.y]] }))
      },
      {
        name: "center",
        type: "scatter",
        symbolSize: 24,
        itemStyle: { color: "#d79a00", borderColor: "#ffffff", borderWidth: 2 },
        data: [[center.x, center.y]]
      },
      {
        name: "results",
        type: "scatter",
        symbolSize: value => Math.max(13, Math.min(28, 13 + value[2] * 22)),
        itemStyle: { color: "#c13f63", borderColor: "#ffffff", borderWidth: 2 },
        label: {
          show: true,
          formatter: params => String(params.data.rank),
          color: "#17191f",
          fontSize: 13,
          backgroundColor: "#ffffff",
          borderRadius: 4,
          padding: [2, 4]
        },
        data: results.map((item, index) => ({
          value: [item.x, item.y, item.score],
          rank: index + 1,
          paper: item
        }))
      }
    ]
  };
}

function renderChart(center, results) {
  state.currentCenter = center;
  state.currentResults = results;
  state.chart.setOption(chartOption(center, results), true);
}

function renderResults(results) {
  els.results.innerHTML = "";
  for (const [index, result] of results.entries()) {
    const li = document.createElement("li");
    li.dataset.index = result.index;
    li.innerHTML = `
      <div class="rank">#${index + 1} score ${Number(result.score).toFixed(3)}</div>
      <div class="title">${escapeHtml(result.title_ja || result.title)}</div>
      ${result.title_ja ? `<div class="title-en">${escapeHtml(result.title)}</div>` : ""}
      <div class="meta">${escapeHtml([result.content_type, result.session_name, result.session_room].filter(Boolean).join(" / "))}</div>
      <div class="snippet">${escapeHtml(result.snippet_ja || result.snippet)}</div>
    `;
    li.addEventListener("click", () => selectResult(result, false));
    els.results.appendChild(li);
  }
}

function markActive(index) {
  document.querySelectorAll("li").forEach(li => {
    li.classList.toggle("active", Number(li.dataset.index) === index);
  });
}

function renderDetail(result) {
  if (!result) {
    els.detail.className = "empty";
    els.detail.textContent = "マップ上の点、または検索結果を選択してください。";
    return;
  }
  els.detail.className = "";
  els.detail.innerHTML = `
    <div class="rank">selected / score ${Number(result.score).toFixed(3)}</div>
    <div class="title">${escapeHtml(result.title_ja || result.title)}</div>
    ${result.title_ja ? `<div class="title-en">${escapeHtml(result.title)}</div>` : ""}
    <div class="meta">${escapeHtml(result.authors || "")}</div>
    <div class="snippet">${escapeHtml(result.snippet_ja || result.snippet)}</div>
    <div class="actions">
      <button class="link-button" type="button" id="similarBtn">この論文に近いものを探す</button>
      <a href="${result.url}" target="_blank" rel="noreferrer">SIGCHI program</a>
    </div>
  `;
  document.querySelector("#similarBtn").addEventListener("click", () => runSimilar(result.index));
}

function selectResult(result, followSimilar) {
  markActive(result.index);
  renderDetail(result);
  if (followSimilar) runSimilar(result.index);
}

async function runQuery(query) {
  els.button.disabled = true;
  els.status.textContent = "OpenAI APIでクエリベクトルを計算中";
  try {
    const data = await fetchJson(`${apiBasePath()}/api/embed?q=${encodeURIComponent(query)}`);
    const vector = new Float32Array(data.embedding);
    els.status.textContent = "ブラウザ側で類似度を計算中";
    const top = await workerRequest("query", { vector, topK: 10 });
    const results = top.map(([index, score]) => paperAt(index, score));
    const center = queryPointFromEmbedding(vector);
    state.currentMode = "query";
    renderChart(center, results);
    renderResults(results);
    renderDetail(null);
    els.status.textContent = `${results.length}件を表示 / 類似度計算はクライアント側`;
  } catch (error) {
    els.status.textContent = `エラー: ${error.message}`;
  } finally {
    els.button.disabled = false;
  }
}

async function runSimilar(index) {
  els.status.textContent = "ブラウザ側で近傍論文を計算中";
  const top = await workerRequest("similar", { index, topK: 10 });
  const results = top.map(([paperIndex, score]) => paperAt(paperIndex, score));
  const source = paperAt(index, 1);
  state.currentMode = "similar";
  renderChart({ x: source.x, y: source.y }, results);
  renderResults(results);
  renderDetail(source);
  els.status.textContent = `「${source.title_ja || source.title}」に近い${results.length}件`;
}

async function init() {
  if (!window.echarts) {
    els.status.textContent = "EChartsの読み込みに失敗しました。";
    return;
  }
  state.chart = echarts.init(els.chart, null, { renderer: "canvas" });
  state.chart.on("click", params => {
    if (params.seriesName === "results" && params.data && params.data.paper) {
      selectResult(params.data.paper, true);
    }
  });
  els.status.textContent = "静的ベクトルデータを読み込み中";
  const manifest = await fetchJson(assetUrl("manifest.json", ""), { cache: "no-store" });
  state.assetVersion = manifest.version || Date.now().toString();
  const [papers, projection, embeddings, coords] = await Promise.all([
    fetchJson(assetUrl("papers.json"), { cache: "no-store" }),
    fetchJson(assetUrl("projection.json"), { cache: "no-store" }),
    fetchFloat32(assetUrl("embeddings.f32"), { cache: "no-store" }),
    fetchFloat32(assetUrl("coords.f32"), { cache: "no-store" })
  ]);
  state.papers = papers;
  state.projection = projection;
  state.coords = coords;
  await initWorker(embeddings);
  els.status.textContent = `${papers.length}件を読み込み済み`;
  await runQuery(els.query.value.trim());
}

els.form.addEventListener("submit", event => {
  event.preventDefault();
  const query = els.query.value.trim();
  if (query) runQuery(query);
});

window.addEventListener("resize", () => state.chart?.resize());

init();
