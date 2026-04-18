const state = {
  papers: [],
  coords: null,
  projection: null,
  chart: null,
  worker: null,
  assetVersion: "",
  requestSeq: 0,
  currentResults: [],
  graph: null,
  selectedIndex: null,
  currentMode: "query"
};

const els = {
  form: document.querySelector("#searchForm"),
  query: document.querySelector("#query"),
  button: document.querySelector("#submit"),
  status: document.querySelector("#status"),
  detail: document.querySelector("#detail"),
  results: document.querySelector("#results"),
  chart: document.querySelector("#chart"),
  zoomIn: document.querySelector("#zoomIn"),
  zoomOut: document.querySelector("#zoomOut"),
  fitMap: document.querySelector("#fitMap")
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

function shortTitle(value, maxLen = 34) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 1) + "...";
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

function queryPointFromEmbedding(vector, results) {
  if (state.projection.method === "umap") {
    let total = 0;
    let x = 0;
    let y = 0;
    for (const result of results) {
      const weight = Math.max(result.score, 0.000001);
      total += weight;
      x += result.x * weight;
      y += result.y * weight;
    }
    return total > 0 ? { x: x / total, y: y / total } : { x: 0, y: 0 };
  }
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

function createGraph() {
  return {
    query: null,
    nodes: new Map(),
    edges: [],
    edgeKeys: new Set()
  };
}

function addGraphNode(result) {
  const existing = state.graph.nodes.get(result.index);
  if (!existing || result.score > existing.score) {
    state.graph.nodes.set(result.index, result);
    return result;
  }
  return existing;
}

function addGraphEdge(from, to, score, kind) {
  const key = `${from}->${to}`;
  if (state.graph.edgeKeys.has(key)) return;
  state.graph.edgeKeys.add(key);
  state.graph.edges.push({ from, to, score, kind });
}

function pointForRef(ref) {
  if (ref === "query") return state.graph.query;
  const node = state.graph.nodes.get(Number(ref)) || paperAt(Number(ref), 1);
  return { x: node.x, y: node.y };
}

function resetGraph(center, results) {
  state.graph = createGraph();
  state.graph.query = { x: center.x, y: center.y, label: "query" };
  state.selectedIndex = null;
  for (const result of results) {
    addGraphNode(result);
    addGraphEdge("query", result.index, result.score, "query");
  }
}

function mergeNeighborhood(source, results) {
  addGraphNode(source);
  state.selectedIndex = source.index;
  for (const result of results) {
    addGraphNode(result);
    addGraphEdge(source.index, result.index, result.score, "similar");
  }
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

function chartOption() {
  const nodes = Array.from(state.graph.nodes.values());
  const values = [state.graph.query, ...nodes].filter(Boolean);
  const xs = values.map(item => item.x);
  const ys = values.map(item => item.y);
  let minX = Math.min(...xs);
  let maxX = Math.max(...xs);
  let minY = Math.min(...ys);
  let maxY = Math.max(...ys);
  let xSpan = Math.max(maxX - minX, 0.0001);
  let ySpan = Math.max(maxY - minY, 0.0001);
  const padRatio = 0.24;
  minX -= xSpan * padRatio;
  maxX += xSpan * padRatio;
  minY -= ySpan * padRatio;
  maxY += ySpan * padRatio;

  // Keep x/y data units visually comparable. Otherwise UMAP clusters can look
  // vertically stretched just because the chart viewport is tall.
  xSpan = maxX - minX;
  ySpan = maxY - minY;
  const plotWidth = Math.max(state.chart.getWidth() - 56, 1);
  const plotHeight = Math.max(state.chart.getHeight() - 56, 1);
  const targetYSpan = xSpan * (plotHeight / plotWidth);
  if (targetYSpan > ySpan) {
    const extra = (targetYSpan - ySpan) / 2;
    minY -= extra;
    maxY += extra;
  } else {
    const targetXSpan = ySpan * (plotWidth / plotHeight);
    const extra = (targetXSpan - xSpan) / 2;
    minX -= extra;
    maxX += extra;
  }

  return {
    animationDuration: 350,
    grid: { left: 28, right: 28, top: 28, bottom: 28 },
    xAxis: { min: minX, max: maxX, show: false },
    yAxis: { min: minY, max: maxY, show: false },
    dataZoom: [
      {
        type: "inside",
        xAxisIndex: 0,
        filterMode: "none",
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
        preventDefaultMouseMove: true,
        start: 0,
        end: 100
      },
      {
        type: "inside",
        yAxisIndex: 0,
        filterMode: "none",
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
        preventDefaultMouseMove: true,
        start: 0,
        end: 100
      }
    ],
    tooltip: {
      trigger: "item",
      confine: true,
      formatter: params => {
        if (params.seriesName === "query") return "query";
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
        data: state.graph.edges.map(edge => {
          const from = pointForRef(edge.from);
          const to = pointForRef(edge.to);
          return {
            coords: [[from.x, from.y], [to.x, to.y]],
            lineStyle: {
              color: edge.kind === "query" ? "#9ea7b3" : "#007c73",
              width: edge.kind === "query" ? 1.4 : 1.8,
              type: edge.kind === "query" ? "dashed" : "solid",
              opacity: edge.kind === "query" ? 0.46 : 0.62
            }
          };
        })
      },
      {
        name: "query",
        type: "scatter",
        symbol: "diamond",
        symbolSize: 22,
        itemStyle: { color: "#d79a00", borderColor: "#ffffff", borderWidth: 2 },
        data: state.graph.query ? [[state.graph.query.x, state.graph.query.y]] : []
      },
      {
        name: "papers",
        type: "scatter",
        symbolSize: value => Math.max(13, Math.min(28, 13 + value[2] * 22)),
        label: {
          show: true,
          position: "right",
          distance: 8,
          formatter: params => params.data.label || "",
          color: "#17191f",
          fontSize: 12,
          lineHeight: 16,
          backgroundColor: "rgba(255,255,255,.92)",
          borderRadius: 4,
          padding: [2, 5]
        },
        labelLayout: {
          hideOverlap: true,
          moveOverlap: "shiftY"
        },
        data: nodes
          .filter(item => item.index !== state.selectedIndex)
          .map(item => ({
            value: [item.x, item.y, item.score],
            rank: state.currentResults.findIndex(result => result.index === item.index) + 1 || "",
            paper: item,
            label: state.currentResults.some(result => result.index === item.index)
              ? shortTitle(item.title_ja || item.title)
              : "",
            itemStyle: { color: "#c13f63", borderColor: "#ffffff", borderWidth: 2 }
          }))
      },
      {
        name: "selected",
        type: "scatter",
        symbolSize: 30,
        z: 5,
        itemStyle: { color: "#007c73", borderColor: "#ffffff", borderWidth: 3 },
        label: {
          show: true,
          position: "right",
          distance: 10,
          formatter: params => shortTitle(params.data.paper.title_ja || params.data.paper.title, 42),
          color: "#17191f",
          fontSize: 12,
          lineHeight: 16,
          backgroundColor: "rgba(255,255,255,.95)",
          borderRadius: 4,
          padding: [3, 6]
        },
        labelLayout: {
          hideOverlap: true,
          moveOverlap: "shiftY"
        },
        data: state.selectedIndex === null ? [] : (() => {
          const item = state.graph.nodes.get(state.selectedIndex);
          return item ? [{ value: [item.x, item.y, item.score], paper: item }] : [];
        })()
      }
    ]
  };
}

function renderGraph() {
  if (!state.graph) return;
  state.chart.setOption(chartOption(), true);
}

function currentZoom(index) {
  const option = state.chart.getOption();
  const zoom = option.dataZoom?.[index] || {};
  return {
    start: Number.isFinite(zoom.start) ? zoom.start : 0,
    end: Number.isFinite(zoom.end) ? zoom.end : 100
  };
}

function zoomBy(factor) {
  if (!state.graph) return;
  for (const dataZoomIndex of [0, 1]) {
    const zoom = currentZoom(dataZoomIndex);
    const center = (zoom.start + zoom.end) / 2;
    const span = Math.min(100, Math.max(4, (zoom.end - zoom.start) * factor));
    const start = Math.max(0, Math.min(100 - span, center - span / 2));
    state.chart.dispatchAction({
      type: "dataZoom",
      dataZoomIndex,
      start,
      end: start + span
    });
  }
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
    li.addEventListener("click", () => expandFromPaper(result));
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
      <button class="link-button" type="button" id="similarBtn">近い論文を追加</button>
      <a href="${result.url}" target="_blank" rel="noreferrer">SIGCHI program</a>
    </div>
  `;
  document.querySelector("#similarBtn").addEventListener("click", () => expandFromPaper(result));
}

function selectResult(result) {
  markActive(result.index);
  renderDetail(result);
  state.selectedIndex = result.index;
  renderGraph();
}

async function expandFromPaper(result) {
  selectResult(result);
  await runSimilar(result.index);
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
    const center = queryPointFromEmbedding(vector, results);
    state.currentMode = "query";
    state.currentResults = results;
    resetGraph(center, results);
    renderGraph();
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
  state.currentResults = results;
  mergeNeighborhood(source, results);
  renderGraph();
  renderResults(results);
  renderDetail(source);
  els.status.textContent = `「${source.title_ja || source.title}」に近い${results.length}件をマップに追加`;
}

async function init() {
  if (!window.echarts) {
    els.status.textContent = "EChartsの読み込みに失敗しました。";
    return;
  }
  state.chart = echarts.init(els.chart, null, { renderer: "canvas" });
  state.chart.on("click", params => {
    if ((params.seriesName === "papers" || params.seriesName === "selected") && params.data && params.data.paper) {
      expandFromPaper(params.data.paper);
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

els.zoomIn.addEventListener("click", () => zoomBy(0.7));
els.zoomOut.addEventListener("click", () => zoomBy(1.35));
els.fitMap.addEventListener("click", () => renderGraph());

window.addEventListener("resize", () => {
  state.chart?.resize();
  renderGraph();
});

init();
