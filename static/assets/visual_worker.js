let embeddings = null;
let count = 0;
let dims = 0;

function topKFromVector(vector, topK, excludeIndex) {
  const results = [];
  for (let index = 0; index < count; index++) {
    if (index === excludeIndex) continue;
    let score = 0;
    const offset = index * dims;
    for (let dim = 0; dim < dims; dim++) {
      score += embeddings[offset + dim] * vector[dim];
    }
    if (results.length < topK) {
      results.push([index, score]);
      results.sort((a, b) => a[1] - b[1]);
    } else if (score > results[0][1]) {
      results[0] = [index, score];
      results.sort((a, b) => a[1] - b[1]);
    }
  }
  results.sort((a, b) => b[1] - a[1]);
  return results;
}

function similarity(a, b) {
  let score = 0;
  const aOffset = a * dims;
  const bOffset = b * dims;
  for (let dim = 0; dim < dims; dim++) {
    score += embeddings[aOffset + dim] * embeddings[bOffset + dim];
  }
  return score;
}

function orderByPairSimilarity(indices) {
  if (indices.length <= 2) return indices;
  const remaining = new Set(indices);
  let bestStart = indices[0];
  let bestMean = -Infinity;
  for (const index of indices) {
    let total = 0;
    for (const other of indices) {
      if (other !== index) total += similarity(index, other);
    }
    const mean = total / Math.max(indices.length - 1, 1);
    if (mean > bestMean) {
      bestMean = mean;
      bestStart = index;
    }
  }

  const ordered = [bestStart];
  remaining.delete(bestStart);
  while (remaining.size) {
    const last = ordered[ordered.length - 1];
    let bestNext = null;
    let bestScore = -Infinity;
    for (const candidate of remaining) {
      const score = similarity(last, candidate);
      if (score > bestScore) {
        bestScore = score;
        bestNext = candidate;
      }
    }
    ordered.push(bestNext);
    remaining.delete(bestNext);
  }
  return ordered;
}

self.onmessage = event => {
  const message = event.data;
  if (message.type === "init") {
    embeddings = new Float32Array(message.buffer);
    count = message.count;
    dims = message.dims;
    self.postMessage({ type: "ready" });
    return;
  }
  if (message.type === "query") {
    const vector = new Float32Array(message.vector);
    self.postMessage({
      type: "results",
      requestId: message.requestId,
      results: topKFromVector(vector, message.topK, -1)
    });
    return;
  }
  if (message.type === "similar") {
    const offset = message.index * dims;
    const vector = embeddings.subarray(offset, offset + dims);
    self.postMessage({
      type: "results",
      requestId: message.requestId,
      results: topKFromVector(vector, message.topK, message.index)
    });
    return;
  }
  if (message.type === "order") {
    self.postMessage({
      type: "results",
      requestId: message.requestId,
      results: orderByPairSimilarity(message.indices)
    });
  }
};
