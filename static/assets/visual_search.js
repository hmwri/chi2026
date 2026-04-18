const DEFAULT_LANGUAGE = "ja";
const UI_VERSION = "20260418-i18n-ui";

const state = {
  papers: [],
  embeddingDims: 0,
  chart: null,
  worker: null,
  assetVersion: "",
  requestSeq: 0,
  currentResults: [],
  graph: null,
  layers: [],
  activeLayerId: null,
  dialogPaper: null,
  selectedIndex: null,
  currentMode: "query",
  labelMode: "tagline",
  language: DEFAULT_LANGUAGE
};

const els = {
  form: document.querySelector("#searchForm"),
  subtitle: document.querySelector("#subtitle"),
  query: document.querySelector("#query"),
  topK: document.querySelector("#topK"),
  topKLabel: document.querySelector("#topKLabel"),
  labelMode: document.querySelector("#labelMode"),
  labelModeLabel: document.querySelector("#labelModeLabel"),
  labelModeTagline: document.querySelector("#labelModeTagline"),
  labelModeTitle: document.querySelector("#labelModeTitle"),
  languageMode: document.querySelector("#languageMode"),
  languageModeLabel: document.querySelector("#languageModeLabel"),
  button: document.querySelector("#submit"),
  status: document.querySelector("#status"),
  layers: document.querySelector("#layers"),
  detail: document.querySelector("#detail"),
  results: document.querySelector("#results"),
  chart: document.querySelector("#chart"),
  backLayer: document.querySelector("#backLayer"),
  zoomIn: document.querySelector("#zoomIn"),
  zoomOut: document.querySelector("#zoomOut"),
  fitMap: document.querySelector("#fitMap"),
  overlay: document.querySelector("#paperOverlay"),
  overlayBody: document.querySelector("#overlayBody"),
  overlayClose: document.querySelector("#overlayClose"),
  overlaySimilar: document.querySelector("#overlaySimilar"),
  overlayProgram: document.querySelector("#overlayProgram")
};

const I18N = {
  ja: {
    subtitle: "中心に近い論文ほど内側へ、隣り合う論文ほど互いに近くなるように配置します。",
    topK: "表示件数",
    labelMode: "グラフ表示",
    tagline: "キャッチコピー",
    title: "タイトル",
    language: "言語",
    search: "検索",
    back: "戻る",
    fit: "全体",
    selected: "selected",
    detailInitial: "マップ上の点、または検索結果を選択してください。",
    detailEmpty: "マップ上の点、または検索結果を選択すると、対応する項目をハイライトします。",
    detailHint: "詳細はマップ上のオーバーレイで確認できます。",
    similarButton: "近い論文を探す",
    loadingEcharts: "EChartsの読み込みに失敗しました。",
    loadingAssets: "静的ベクトルデータを読み込み中",
    loaded: count => `${count}件を読み込み済み`,
    embedding: "OpenAI APIでクエリベクトルを計算中",
    scoring: "ブラウザ側で類似度を計算中",
    similarScoring: "ブラウザ側で近傍論文を計算中",
    queryResults: count => `${count}件を、queryとの類似度を距離にして表示`,
    paperResults: (title, count) => `「${title}」との類似度を距離にして${count}件を表示`,
    error: message => `エラー: ${message}`
  },
  en: {
    subtitle: "Papers closer to the center are more similar; neighboring papers are arranged by mutual similarity.",
    topK: "Results",
    labelMode: "Map label",
    tagline: "Summary",
    title: "Title",
    language: "Language",
    search: "Search",
    back: "Back",
    fit: "Fit",
    selected: "selected",
    detailInitial: "Select a node or result to inspect it.",
    detailEmpty: "Select a node or result to highlight the corresponding item.",
    detailHint: "Open the map overlay for details.",
    similarButton: "Find similar papers",
    loadingEcharts: "Failed to load ECharts.",
    loadingAssets: "Loading static vector data",
    loaded: count => `${count} papers loaded`,
    embedding: "Computing the query embedding with the OpenAI API",
    scoring: "Computing similarities in the browser",
    similarScoring: "Computing nearby papers in the browser",
    queryResults: count => `Showing ${count} papers by distance from the query`,
    paperResults: (title, count) => `Showing ${count} papers by distance from "${title}"`,
    error: message => `Error: ${message}`
  }
};

function tr(key, ...args) {
  const value = I18N[state.language][key];
  return typeof value === "function" ? value(...args) : value;
}

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

function graphTitle(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function isJapaneseMode() {
  return state.language === "ja";
}

function applyUiLanguage() {
  document.documentElement.lang = state.language;
  els.subtitle.textContent = tr("subtitle");
  els.topKLabel.textContent = tr("topK");
  els.labelModeLabel.textContent = tr("labelMode");
  els.labelModeTagline.textContent = tr("tagline");
  els.labelModeTitle.textContent = tr("title");
  els.languageModeLabel.textContent = tr("language");
  els.button.textContent = tr("search");
  els.backLayer.textContent = tr("back");
  els.fitMap.textContent = tr("fit");
  els.overlayClose.setAttribute("aria-label", state.language === "ja" ? "閉じる" : "Close");
  els.overlaySimilar.textContent = tr("similarButton");
  els.query.setAttribute("aria-label", state.language === "ja" ? "検索語" : "Search query");
  els.topK.setAttribute("aria-label", tr("topK"));
  els.labelMode.setAttribute("aria-label", tr("labelMode"));
  els.languageMode.setAttribute("aria-label", tr("language"));
}

function paperTitle(paper) {
  return isJapaneseMode() ? (paper.title_ja || paper.title) : paper.title;
}

function secondaryTitle(paper) {
  return isJapaneseMode() && paper.title_ja && paper.title_ja !== paper.title ? paper.title : "";
}

function paperAbstract(paper) {
  return isJapaneseMode() ? (paper.abstract_ja || paper.abstract || "") : (paper.abstract || "");
}

function paperSnippet(paper) {
  return snippet(paperAbstract(paper));
}

function paperTagline(paper) {
  return isJapaneseMode() ? paper.tagline_ja : "";
}

function graphLabel(paper) {
  if (state.labelMode === "title") return paperTitle(paper);
  return paperTagline(paper) || paperTitle(paper);
}

function paperAt(index, score) {
  const paper = state.papers[index];
  return {
    ...paper,
    index,
    score,
    x: 0,
    y: 0,
    snippet: snippet(paper.abstract),
    snippet_ja: snippet(paper.abstract_ja)
  };
}

function rankedResults(results) {
  return results
    .slice()
    .sort((a, b) => b.score - a.score)
    .map((result, index) => ({
      ...result,
      rank: index + 1
    }));
}

function displayResults(results) {
  return rankedResults(results).map(result => {
    const existing = results.find(item => item.index === result.index) || {};
    return {
      ...existing,
      ...result
    };
  });
}

function topKValue() {
  const value = Number.parseInt(els.topK.value, 10);
  if (!Number.isFinite(value)) return 10;
  return Math.max(3, Math.min(30, value));
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

async function orderResultsByPairSimilarity(results) {
  const order = await workerRequest("order", { indices: results.map(result => result.index) });
  const byIndex = new Map(results.map(result => [result.index, result]));
  return order.map(index => byIndex.get(index)).filter(Boolean);
}

function circularLayout(center, similarityRankedResults, placementOrderedResults) {
  const orderedResults = placementOrderedResults || similarityRankedResults;
  const countScale = Math.min(Math.max(orderedResults.length - 10, 0) * 0.025, 0.35);
  const minRadius = center.kind === "paper" ? 0.34 : 0.44;
  const maxRadius = (center.kind === "paper" ? 1.08 : 1.52) + countScale;
  const scores = orderedResults.map(result => result.score);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const scoreSpan = maxScore - minScore;
  const startAngle = -Math.PI / 2;
  const step = (Math.PI * 2) / Math.max(orderedResults.length, 1);
  const laidOut = orderedResults.map((result, index) => {
    const angle = startAngle + step * index;
    const rankNorm = orderedResults.length > 1 ? index / (orderedResults.length - 1) : 0;
    const scoreNorm = scoreSpan > 0.000001 ? (result.score - minScore) / scoreSpan : 1 - rankNorm;
    const radius = minRadius + (1 - scoreNorm) * (maxRadius - minRadius);
    return {
      ...result,
      x: center.x + Math.cos(angle) * radius,
      y: center.y + Math.sin(angle) * radius,
      angle,
      radius
    };
  });
  return {
    center,
    results: similarityRankedResults.map(result => laidOut.find(item => item.index === result.index) || result)
  };
}

function createLayer(mode, center, results, parentId) {
  const id = `${mode}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return {
    id,
    parentId,
    mode,
    center,
    results,
    label: mode === "query"
      ? `query: ${shortTitle(els.query.value.trim(), 28)}`
      : `paper: ${shortTitle(paperTitle(center), 28)}`
  };
}

function createGraph() {
  return {
    center: null,
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
  if (ref === "center") return state.graph.center;
  const node = state.graph.nodes.get(Number(ref)) || paperAt(Number(ref), 1);
  return { x: node.x, y: node.y };
}

function positionedPaper(index) {
  if (state.graph?.center?.index === index) return state.graph.center;
  const node = state.graph?.nodes.get(index);
  return node || paperAt(index, 1);
}

function resetGraph(center, results) {
  state.graph = createGraph();
  state.graph.center = center;
  state.selectedIndex = center.index ?? null;
  for (const [index, result] of results.entries()) {
    addGraphNode(result);
    addGraphEdge("center", result.index, result.score, center.kind === "paper" ? "similar" : "query");
    const next = results[(index + 1) % results.length];
    if (next && next.index !== result.index) {
      addGraphEdge(result.index, next.index, Math.min(result.score, next.score), "peer");
    }
  }
}

function mergeNeighborhood(source, results) {
  resetGraph(source, results);
}

function labelPlacement(item) {
  return {
    position: item.x < 0 ? "left" : "right",
    align: item.x < 0 ? "right" : "left"
  };
}

function paperSeriesData() {
  return Array.from(state.graph.nodes.values())
    .filter(item => item.index !== state.selectedIndex)
    .map(item => {
      const rank = item.rank || "";
      const placement = labelPlacement(item);
      return {
        value: [item.x, item.y, item.score],
        rank,
        paper: item,
        labelText: graphTitle(graphLabel(item)),
        labelShiftX: placement.position === "left" ? -8 : 8,
        labelShiftY: rank ? ((rank % 3) - 1) * 7 : 0,
        label: {
          ...placement
        },
        itemStyle: { color: "#c13f63", borderColor: "#ffffff", borderWidth: 2 }
      };
    });
}

function selectedSeriesData() {
  if (state.selectedIndex === null) return [];
  const item = state.graph.nodes.get(state.selectedIndex);
  return item ? [{
    value: [item.x, item.y, item.score],
    paper: item,
    label: {
      ...labelPlacement(item)
    }
  }] : [];
}

function pointFromSnapshot(snapshot, ref) {
  if (ref === "center") return snapshot.center;
  return snapshot.nodes.find(node => node.index === Number(ref));
}

function layerById(id) {
  return state.layers.find(layer => layer.id === id);
}

function ancestorLayers(id) {
  const ancestors = [];
  let current = layerById(id);
  while (current?.parentId) {
    const parent = layerById(current.parentId);
    if (!parent) break;
    ancestors.unshift(parent);
    current = parent;
  }
  return ancestors;
}

function inactiveLayerSeries() {
  const series = [];
  for (const layer of ancestorLayers(state.activeLayerId)) {
    if (!layer.graph) continue;
    const snapshot = layer.graph;
    series.push({
      name: `background-links-${layer.id}`,
      type: "lines",
      coordinateSystem: "cartesian2d",
      silent: true,
      z: 0,
      data: snapshot.edges.map(edge => {
        const from = pointFromSnapshot(snapshot, edge.from);
        const to = pointFromSnapshot(snapshot, edge.to);
        if (!from || !to) return null;
        return {
          coords: [[from.x, from.y], [to.x, to.y]],
          lineStyle: { color: "#8f98a5", width: 1, opacity: 0.14 }
        };
      }).filter(Boolean)
    });
    series.push({
      name: `background-center-${layer.id}`,
      type: "scatter",
      silent: true,
      z: 1,
      symbol: "diamond",
      symbolSize: 17,
      itemStyle: { color: "#d79a00", opacity: 0.26, borderWidth: 0 },
      data: snapshot.center ? [[snapshot.center.x, snapshot.center.y]] : []
    });
    series.push({
      name: `background-papers-${layer.id}`,
      type: "scatter",
      silent: true,
      z: 1,
      symbolSize: 11,
      itemStyle: { color: "#c13f63", opacity: 0.20, borderWidth: 0 },
      data: snapshot.nodes.map(node => [node.x, node.y, node.score])
    });
  }
  return series;
}

function snapshotGraph() {
  return {
    center: state.graph.center,
    nodes: Array.from(state.graph.nodes.values()),
    edges: state.graph.edges.map(edge => ({ ...edge }))
  };
}

function restoreGraph(snapshot) {
  state.graph = createGraph();
  state.graph.center = snapshot.center;
  for (const node of snapshot.nodes) {
    state.graph.nodes.set(node.index, node);
  }
  for (const edge of snapshot.edges) {
    state.graph.edges.push({ ...edge });
    state.graph.edgeKeys.add(`${edge.from}->${edge.to}`);
  }
}

function activeLayerIndex() {
  return state.layers.findIndex(item => item.id === state.activeLayerId);
}

function updateLayerControls() {
  const layer = layerById(state.activeLayerId);
  els.backLayer.disabled = !layer?.parentId;
}

function renderLayers() {
  els.layers.innerHTML = "";
  const pathIds = new Set([
    ...ancestorLayers(state.activeLayerId).map(layer => layer.id),
    state.activeLayerId
  ]);
  for (const [index, layer] of state.layers.entries()) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = [
      "layer-button",
      layer.id === state.activeLayerId ? "active" : "",
      pathIds.has(layer.id) ? "in-path" : "off-path"
    ].filter(Boolean).join(" ");
    button.textContent = `${index + 1}. ${layer.label}`;
    button.title = layer.label;
    button.addEventListener("click", () => activateLayer(layer.id));
    els.layers.appendChild(button);
  }
  updateLayerControls();
}

function activateLayer(id) {
  const layer = state.layers.find(item => item.id === id);
  if (!layer) return;
  state.activeLayerId = id;
  state.currentMode = layer.mode;
  state.currentResults = layer.results;
  restoreGraph(layer.graph);
  state.selectedIndex = layer.center.index ?? null;
  renderGraph();
  renderResults(layer.results);
  renderDetail(layer.mode === "query" ? null : layer.center);
  renderLayers();
  els.status.textContent = layer.mode === "query"
    ? tr("queryResults", layer.results.length)
    : tr("paperResults", paperTitle(layer.center), layer.results.length);
}

function backLayer() {
  const layer = layerById(state.activeLayerId);
  if (layer?.parentId) activateLayer(layer.parentId);
}

function pushLayer(mode, center, results) {
  const parentId = mode === "query" ? null : state.activeLayerId;
  const layer = createLayer(mode, center, results, parentId);
  layer.graph = snapshotGraph();
  if (mode === "query") {
    state.layers = [layer];
  } else {
    state.layers.push(layer);
  }
  state.activeLayerId = layer.id;
  renderLayers();
}

async function initWorker(embeddings) {
  const worker = new Worker(assetUrl("visual_worker.js", UI_VERSION));
  state.worker = worker;
  await new Promise(resolve => {
    worker.addEventListener("message", event => {
      if (event.data.type === "ready") resolve();
    }, { once: true });
    worker.postMessage({
      type: "init",
      buffer: embeddings.buffer,
      count: state.papers.length,
      dims: state.embeddingDims
    }, [embeddings.buffer]);
  });
}

function chartOption() {
  const nodes = Array.from(state.graph.nodes.values());
  const inactiveValues = state.layers
    .filter(layer => layer.id !== state.activeLayerId && layer.graph)
    .flatMap(layer => [layer.graph.center, ...layer.graph.nodes]);
  const values = [state.graph.center, ...nodes, ...inactiveValues].filter(Boolean);
  const xs = values.map(item => item.x);
  const ys = values.map(item => item.y);
  let minX = Math.min(...xs);
  let maxX = Math.max(...xs);
  let minY = Math.min(...ys);
  let maxY = Math.max(...ys);
  let xSpan = Math.max(maxX - minX, 0.0001);
  let ySpan = Math.max(maxY - minY, 0.0001);
  const padRatio = 0.36;
  minX -= xSpan * padRatio;
  maxX += xSpan * padRatio;
  minY -= ySpan * padRatio;
  maxY += ySpan * padRatio;

  // Keep x/y data units visually comparable so the radial distance keeps its
  // meaning even when the chart viewport is not square.
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
    animation: false,
    animationDuration: 0,
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
        if (params.seriesName === "center") {
          return params.data.paper
            ? `<strong>${escapeHtml(paperTitle(params.data.paper))}</strong>`
            : "query";
        }
        const item = params.data.paper;
        return `<strong>${escapeHtml(paperTitle(item))}</strong><br>${escapeHtml(item.content_type || "")}<br>score ${Number(item.score).toFixed(3)}`;
      }
    },
    series: [
      ...inactiveLayerSeries(),
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
              color: edge.kind === "peer" ? "#b8c0ca" : edge.kind === "query" ? "#9ea7b3" : "#007c73",
              width: edge.kind === "peer" ? 1 : edge.kind === "query" ? 1.4 : 1.8,
              type: edge.kind === "peer" ? "dotted" : edge.kind === "query" ? "dashed" : "solid",
              opacity: edge.kind === "peer" ? 0.5 : edge.kind === "query" ? 0.46 : 0.62
            }
          };
        })
      },
      {
        name: "center",
        type: "scatter",
        symbol: "diamond",
        symbolSize: 22,
        itemStyle: { color: "#d79a00", borderColor: "#ffffff", borderWidth: 2 },
        label: {
          show: true,
          position: "top",
          distance: 10,
          formatter: params => params.data.label || "query",
          color: "#17191f",
          fontSize: 12,
          lineHeight: 16,
          backgroundColor: "rgba(255,255,255,.95)",
          borderRadius: 4,
          padding: [3, 6]
        },
        data: state.graph.center ? [{
          value: [state.graph.center.x, state.graph.center.y],
          paper: state.graph.center.index === undefined ? null : state.graph.center,
          label: state.graph.center.index === undefined ? "query" : shortTitle(graphLabel(state.graph.center), 42)
        }] : []
      },
      {
        id: "papers",
        name: "papers",
        type: "scatter",
        symbolSize: value => Math.max(13, Math.min(28, 13 + value[2] * 22)),
        label: {
          show: true,
          position: "right",
          distance: 14,
          formatter: params => params.data.labelText || "",
          color: "#17191f",
          fontSize: 12,
          lineHeight: 16,
          backgroundColor: "rgba(255,255,255,.92)",
          borderRadius: 4,
          padding: [2, 5]
        },
        labelLayout: params => ({
          hideOverlap: true,
          moveOverlap: "shiftY",
          draggable: false,
          dx: params.data?.labelShiftX || 0,
          dy: params.data?.labelShiftY || 0
        }),
        data: paperSeriesData()
      },
      {
        id: "selected",
        name: "selected",
        type: "scatter",
        symbolSize: 30,
        z: 5,
        itemStyle: { color: "#007c73", borderColor: "#ffffff", borderWidth: 3 },
        label: {
          show: true,
          position: "right",
          distance: 16,
          formatter: params => graphTitle(graphLabel(params.data.paper)),
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
        data: selectedSeriesData()
      }
    ]
  };
}

function renderGraph() {
  if (!state.graph) return;
  state.chart.setOption(chartOption(), true);
}

function refreshSelection() {
  if (!state.graph) return;
  state.chart.setOption({
    series: [
      { id: "papers", data: paperSeriesData() },
      { id: "selected", data: selectedSeriesData() }
    ]
  });
}

function refreshLanguage() {
  applyUiLanguage();
  renderGraph();
  renderResults(state.currentResults);
  const selected = state.selectedIndex === null ? null : positionedPaper(state.selectedIndex);
  renderDetail(selected?.index === undefined ? null : selected);
  if (!els.overlay.hidden && state.dialogPaper) {
    openPaperOverlay(positionedPaper(state.dialogPaper.index));
  }
  renderLayers();
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
  for (const result of displayResults(results)) {
    const li = document.createElement("li");
    li.dataset.index = result.index;
    li.innerHTML = `
      <div class="rank">#${result.rank} score ${Number(result.score).toFixed(3)}</div>
      ${paperTagline(result) ? `<div class="tagline">${escapeHtml(paperTagline(result))}</div>` : ""}
      <div class="title">${escapeHtml(paperTitle(result))}</div>
      ${secondaryTitle(result) ? `<div class="title-en">${escapeHtml(secondaryTitle(result))}</div>` : ""}
      <div class="meta">${escapeHtml([result.content_type, result.session_name, result.session_room].filter(Boolean).join(" / "))}</div>
      <div class="snippet">${escapeHtml(paperSnippet(result))}</div>
      <div class="actions">
        <a href="${result.url}" target="_blank" rel="noreferrer">CHI program</a>
      </div>
    `;
    li.addEventListener("click", () => selectResult(result, true));
    li.querySelector("a").addEventListener("click", event => event.stopPropagation());
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
    els.detail.textContent = tr("detailEmpty");
    return;
  }
  els.detail.className = "empty";
  els.detail.innerHTML = `
    <div class="rank">${escapeHtml(tr("selected"))}</div>
    ${paperTagline(result) ? `<div class="tagline">${escapeHtml(paperTagline(result))}</div>` : ""}
    <div class="title">${escapeHtml(paperTitle(result))}</div>
    <div class="meta">${escapeHtml(tr("detailHint"))}</div>
  `;
}

function overlayPositionFor(result) {
  const point = state.chart.convertToPixel({ xAxisIndex: 0, yAxisIndex: 0 }, [result.x, result.y]);
  const chartRect = els.chart.getBoundingClientRect();
  const width = Math.min(430, window.innerWidth - 28);
  const height = Math.min(560, window.innerHeight - 28);
  let left = chartRect.left + point[0] + 18;
  let top = chartRect.top + point[1] - 26;
  if (left + width > window.innerWidth - 14) left = chartRect.left + point[0] - width - 18;
  if (left < 14) left = 14;
  if (top + height > window.innerHeight - 14) top = window.innerHeight - height - 14;
  if (top < 14) top = 14;
  return { left, top };
}

function openPaperOverlay(result) {
  const abstract = paperAbstract(result);
  state.dialogPaper = result;
  els.overlayBody.innerHTML = `
    <div class="overlayMeta">
      <div class="rank">selected / score ${Number(result.score).toFixed(3)}</div>
      ${paperTagline(result) ? `<div class="tagline">${escapeHtml(paperTagline(result))}</div>` : ""}
      <div class="title">${escapeHtml(paperTitle(result))}</div>
      ${secondaryTitle(result) ? `<div class="title-en">${escapeHtml(secondaryTitle(result))}</div>` : ""}
      <div class="meta">${escapeHtml(result.authors || "")}</div>
      <div class="meta">${escapeHtml([result.content_type, result.session_name, result.session_room].filter(Boolean).join(" / "))}</div>
    </div>
    <div class="snippet">${escapeHtml(abstract)}</div>
  `;
  els.overlayProgram.href = result.url || "#";
  const { left, top } = overlayPositionFor(result);
  els.overlay.style.left = `${left}px`;
  els.overlay.style.top = `${top}px`;
  els.overlay.hidden = false;
}

function closePaperOverlay() {
  els.overlay.hidden = true;
}

function selectResult(result, showDialog = false) {
  markActive(result.index);
  renderDetail(result);
  state.selectedIndex = result.index;
  refreshSelection();
  const activeItem = document.querySelector(`li[data-index="${result.index}"]`);
  activeItem?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  if (showDialog) openPaperOverlay(result);
}

async function expandFromPaper(result) {
  closePaperOverlay();
  selectResult(result, false);
  await runSimilar(result.index);
}

async function runQuery(query) {
  els.button.disabled = true;
  els.status.textContent = tr("embedding");
  try {
    const data = await fetchJson(`${apiBasePath()}/api/embed?q=${encodeURIComponent(query)}`);
    const vector = new Float32Array(data.embedding);
    els.status.textContent = tr("scoring");
    const topK = topKValue();
    els.topK.value = String(topK);
    const top = await workerRequest("query", { vector, topK });
    const results = rankedResults(top.map(([index, score]) => paperAt(index, score)));
    const placement = await orderResultsByPairSimilarity(results);
    const layout = circularLayout({ x: 0, y: 0, label: "query", kind: "query" }, results, placement);
    state.currentMode = "query";
    state.currentResults = layout.results;
    resetGraph(layout.center, layout.results);
    pushLayer("query", layout.center, layout.results);
    renderGraph();
    renderResults(layout.results);
    renderDetail(null);
    els.status.textContent = tr("queryResults", layout.results.length);
  } catch (error) {
    els.status.textContent = tr("error", error.message);
  } finally {
    els.button.disabled = false;
  }
}

async function runSimilar(index) {
  els.status.textContent = tr("similarScoring");
  const topK = topKValue();
  els.topK.value = String(topK);
  const top = await workerRequest("similar", { index, topK });
  const results = rankedResults(top.map(([paperIndex, score]) => paperAt(paperIndex, score)));
  const placement = await orderResultsByPairSimilarity(results);
  const sourcePaper = positionedPaper(index);
  const source = {
    ...sourcePaper,
    label: shortTitle(graphLabel(sourcePaper), 42),
    kind: "paper"
  };
  const layout = circularLayout(source, results, placement);
  state.currentMode = "similar";
  state.currentResults = layout.results;
  mergeNeighborhood(layout.center, layout.results);
  pushLayer("paper", layout.center, layout.results);
  renderGraph();
  renderResults(layout.results);
  renderDetail(source);
  els.status.textContent = tr("paperResults", paperTitle(source), layout.results.length);
}

async function init() {
  if (!window.echarts) {
    els.status.textContent = tr("loadingEcharts");
    return;
  }
  state.chart = echarts.init(els.chart, null, { renderer: "canvas" });
  state.chart.on("click", params => {
    if ((params.seriesName === "papers" || params.seriesName === "selected" || params.seriesName === "center") && params.data && params.data.paper) {
      selectResult(params.data.paper, true);
    }
  });
  applyUiLanguage();
  renderDetail(null);
  els.status.textContent = tr("loadingAssets");
  const manifest = await fetchJson(assetUrl("manifest.json", ""), { cache: "no-store" });
  state.assetVersion = manifest.version || Date.now().toString();
  const [papers, embeddings] = await Promise.all([
    fetchJson(assetUrl("papers.json"), { cache: "no-store" }),
    fetchFloat32(assetUrl("embeddings.f32"), { cache: "no-store" })
  ]);
  state.papers = papers;
  state.embeddingDims = embeddings.length / papers.length;
  await initWorker(embeddings);
  els.status.textContent = tr("loaded", papers.length);
  await runQuery(els.query.value.trim());
}

els.form.addEventListener("submit", event => {
  event.preventDefault();
  const query = els.query.value.trim();
  if (query) runQuery(query);
});

els.labelMode.addEventListener("change", () => {
  state.labelMode = els.labelMode.value === "title" ? "title" : "tagline";
  renderGraph();
});

els.languageMode.addEventListener("change", () => {
  state.language = els.languageMode.value === "en" ? "en" : "ja";
  refreshLanguage();
});

els.backLayer.addEventListener("click", backLayer);
els.zoomIn.addEventListener("click", () => zoomBy(0.7));
els.zoomOut.addEventListener("click", () => zoomBy(1.35));
els.fitMap.addEventListener("click", () => renderGraph());
els.overlayClose.addEventListener("click", closePaperOverlay);
document.addEventListener("click", event => {
  if (els.overlay.hidden) return;
  const clickedOverlay = els.overlay.contains(event.target);
  const clickedChart = els.chart.contains(event.target);
  const clickedResult = els.results.contains(event.target);
  if (!clickedOverlay && !clickedChart && !clickedResult) closePaperOverlay();
});
els.overlaySimilar.addEventListener("click", () => {
  if (state.dialogPaper) expandFromPaper(state.dialogPaper);
});

window.addEventListener("resize", () => {
  state.chart?.resize();
  closePaperOverlay();
  renderGraph();
});

init();
