const SVG_NS = "http://www.w3.org/2000/svg";
const palette = ["#24735a", "#e56842", "#4c72b0", "#8b6aa8", "#c49a24"];
const numberFormat = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 1 });
const currencyFormat = new Intl.NumberFormat("en-GB", {
  style: "currency",
  currency: "GBP",
  maximumFractionDigits: 0,
});

const navButtons = [...document.querySelectorAll("[data-view]")];
const panels = [...document.querySelectorAll(".view-panel")];

function activateView(viewName, updateAddress = false) {
  const target = document.getElementById(`view-${viewName}`);
  if (!target) return;

  for (const button of navButtons) {
    if (button.dataset.view === viewName) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
  for (const panel of panels) {
    const active = panel === target;
    panel.classList.toggle("active", active);
    panel.setAttribute("aria-hidden", String(!active));
  }
  if (updateAddress && window.location.hash !== `#${target.id}`) {
    window.history.pushState(null, "", `#${target.id}`);
  }
}

for (const button of navButtons) {
  button.addEventListener("click", () => activateView(button.dataset.view, true));
}


function activateAddressView() {
  const requestedId = window.location.hash.slice(1);
  const requestedPanel = panels.find((panel) => panel.id === requestedId);
  activateView(requestedPanel ? requestedPanel.id.replace("view-", "") : "overview");
}

window.addEventListener("popstate", activateAddressView);
window.addEventListener("hashchange", activateAddressView);
activateAddressView();
function element(name, attributes = {}, text = "") {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== "") node.textContent = text;
  return node;
}

function svgRoot(container, height = 300, width = 900) {
  container.replaceChildren();
  const svg = element("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width,
    height,
    role: "presentation",
    preserveAspectRatio: "xMidYMid meet",
  });
  container.append(svg);
  return svg;
}

function tooltip(node, text) {
  node.append(element("title", {}, text));
}

function lineChart(container, labels, values, formatter, color = palette[0], xAxisTitle = "Month", yAxisTitle = "Value") {
  const svg = svgRoot(container);
  const width = 900;
  const height = 300;
  const margin = { top: 18, right: 25, bottom: 48, left: 88 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const maxValue = Math.max(...values, 1) * 1.08;
  const x = (index) => margin.left + (labels.length < 2 ? 0 : index * plotWidth / (labels.length - 1));
  const y = (value) => margin.top + plotHeight - value / maxValue * plotHeight;

  for (let tick = 0; tick <= 4; tick += 1) {
    const value = maxValue * tick / 4;
    const tickY = y(value);
    svg.append(element("line", { x1: margin.left, x2: width - margin.right, y1: tickY, y2: tickY, class: "grid-line" }));
    svg.append(element("text", { x: margin.left - 10, y: tickY + 4, "text-anchor": "end", class: "axis-text" }, formatter(value)));
  }

  const path = values.map((value, index) => `${index === 0 ? "M" : "L"}${x(index)},${y(value)}`).join(" ");
  svg.append(element("path", { d: path, fill: "none", stroke: color, "stroke-width": 3, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  values.forEach((value, index) => {
    const circle = element("circle", { cx: x(index), cy: y(value), r: 4.5, fill: color, class: "chart-point" });
    tooltip(circle, `${labels[index]}: ${formatter(value)}`);
    svg.append(circle);
  });

  const step = Math.max(1, Math.ceil(labels.length / 8));
  labels.forEach((label, index) => {
    if (index % step === 0 || index === labels.length - 1) {
      svg.append(element("text", { x: x(index), y: height - 15, "text-anchor": "middle", class: "axis-text" }, label));
    }
  });
  svg.append(element("text", { x: margin.left + plotWidth / 2, y: height - 1, "text-anchor": "middle", class: "axis-title" }, xAxisTitle));
  svg.append(element("text", { x: 17, y: height / 2, transform: `rotate(-90 17 ${height / 2})`, "text-anchor": "middle", class: "axis-title" }, yAxisTitle));
}

function horizontalBars(container, labels, values, formatter, colorFor = () => palette[0], axisTitle = "Value") {
  const rowHeight = 31;
  const height = Math.max(160, labels.length * rowHeight + 55);
  const svg = svgRoot(container, height);
  const width = 900;
  const left = Math.min(285, Math.max(160, Math.max(...labels.map((label) => label.length)) * 8));
  const plotWidth = width - left - 100;
  const maxValue = Math.max(...values, 1);

  labels.forEach((label, index) => {
    const y = 24 + index * rowHeight;
    const barWidth = values[index] / maxValue * plotWidth;
    svg.append(element("text", { x: left - 12, y: y + 16, "text-anchor": "end", class: "axis-text" }, label.length > 34 ? `${label.slice(0, 31)}...` : label));
    const bar = element("rect", { x: left, y, width: Math.max(1, barWidth), height: 19, rx: 2, fill: colorFor(index), class: "chart-bar" });
    tooltip(bar, `${label}: ${formatter(values[index])}`);
    svg.append(bar);
    svg.append(element("text", { x: left + barWidth + 8, y: y + 15, class: "value-text" }, formatter(values[index])));
  });
  svg.append(element("text", { x: left + plotWidth / 2, y: height - 8, "text-anchor": "middle", class: "axis-title" }, axisTitle));
}

function histogram(container, labels, counts) {
  const svg = svgRoot(container);
  const margin = { top: 18, right: 24, bottom: 48, left: 66 };
  const width = 900;
  const height = 300;
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const maxCount = Math.max(...counts, 1);
  const barWidth = plotWidth / counts.length;

  for (let tick = 0; tick <= 4; tick += 1) {
    const count = maxCount * tick / 4;
    const y = margin.top + plotHeight - count / maxCount * plotHeight;
    svg.append(element("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: "grid-line" }));
    svg.append(element("text", { x: margin.left - 9, y: y + 4, "text-anchor": "end", class: "axis-text" }, numberFormat.format(count)));
  }
  counts.forEach((count, index) => {
    const bar = element("rect", {
      x: margin.left + index * barWidth + 1,
      y: margin.top + plotHeight - count / maxCount * plotHeight,
      width: Math.max(1, barWidth - 2),
      height: count / maxCount * plotHeight,
      fill: palette[0],
      class: "chart-bar",
    });
    tooltip(bar, `Customer value ${currencyFormat.format(labels[index])}: ${numberFormat.format(count)} customers`);
    svg.append(bar);
  });
  svg.append(element("text", { x: width / 2, y: height - 6, "text-anchor": "middle", class: "axis-title" }, "Customer monetary value (£)"));
  svg.append(element("text", { x: 15, y: height / 2, transform: `rotate(-90 15 ${height / 2})`, "text-anchor": "middle", class: "axis-title" }, "Number of customers"));
}

function scatterChart(container, points, selectedCluster) {
  const visible = selectedCluster === "all" ? points : points.filter((point) => String(point.cluster) === selectedCluster);
  const svg = svgRoot(container);
  const width = 900;
  const height = 360;
  const margin = { top: 20, right: 30, bottom: 52, left: 70 };
  const xs = visible.map((point) => point.x);
  const ys = visible.map((point) => point.y);
  const xMin = Math.min(...xs, -1);
  const xMax = Math.max(...xs, 1);
  const yMin = Math.min(...ys, -1);
  const yMax = Math.max(...ys, 1);
  const px = (value) => margin.left + (value - xMin) / (xMax - xMin || 1) * (width - margin.left - margin.right);
  const py = (value) => height - margin.bottom - (value - yMin) / (yMax - yMin || 1) * (height - margin.top - margin.bottom);

  for (let tick = 0; tick <= 4; tick += 1) {
    const vx = xMin + (xMax - xMin) * tick / 4;
    const vy = yMin + (yMax - yMin) * tick / 4;
    const gx = px(vx);
    const gy = py(vy);
    svg.append(element("line", { x1: gx, x2: gx, y1: margin.top, y2: height - margin.bottom, class: "grid-line" }));
    svg.append(element("line", { x1: margin.left, x2: width - margin.right, y1: gy, y2: gy, class: "grid-line" }));
    svg.append(element("text", { x: gx, y: height - margin.bottom + 18, "text-anchor": "middle", class: "axis-text" }, numberFormat.format(vx)));
    svg.append(element("text", { x: margin.left - 10, y: gy + 4, "text-anchor": "end", class: "axis-text" }, numberFormat.format(vy)));
  }
  for (const point of visible) {
    const circle = element("circle", { cx: px(point.x), cy: py(point.y), r: 3.4, fill: palette[Number(point.cluster) % palette.length], opacity: 0.68, class: "chart-point" });
    tooltip(circle, `Cluster ${point.cluster} · Recency z-score ${point.x.toFixed(2)} · Frequency z-score ${point.y.toFixed(2)}`);
    svg.append(circle);
  }
  svg.append(element("text", { x: width / 2, y: height - 5, "text-anchor": "middle", class: "axis-title" }, "Standardized Recency (z-score)"));
  svg.append(element("text", { x: 16, y: height / 2, transform: `rotate(-90 16 ${height / 2})`, "text-anchor": "middle", class: "axis-title" }, "Standardized Frequency (z-score)"));
}

function profileBars(container, segments, metric) {
  const rows = segments.map((segment) => ({
    label: `Cluster ${segment.Cluster} · ${segment.Segment}`,
    value: Number(segment[metric]),
  }));
  const min = Math.min(0, ...rows.map((row) => row.value));
  const max = Math.max(1, ...rows.map((row) => row.value));
  const rowHeight = 54;
  const height = Math.max(170, rows.length * rowHeight + 60);
  const svg = svgRoot(container, height);
  const left = 260;
  const plotWidth = 560;
  const span = max - min || 1;
  const zeroX = left + (0 - min) / span * plotWidth;
  svg.append(element("line", { x1: zeroX, x2: zeroX, y1: 10, y2: height - 35, class: "zero-line" }));
  rows.forEach((row, index) => {
    const y = 20 + index * rowHeight;
    const valueX = left + (row.value - min) / span * plotWidth;
    svg.append(element("text", { x: left - 12, y: y + 18, "text-anchor": "end", class: "axis-text" }, row.label));
    const rect = element("rect", { x: Math.min(zeroX, valueX), y, width: Math.max(1, Math.abs(valueX - zeroX)), height: 25, rx: 2, fill: palette[index % palette.length], class: "chart-bar" });
    tooltip(rect, `${row.label}: ${metric === "Monetary" ? currencyFormat.format(row.value) : numberFormat.format(row.value)}`);
    svg.append(rect);
    svg.append(element("text", { x: valueX + (row.value >= 0 ? 8 : -8), y: y + 17, "text-anchor": row.value >= 0 ? "start" : "end", class: "value-text" }, metric === "Monetary" ? currencyFormat.format(row.value) : numberFormat.format(row.value)));
  });
  const metricUnit = { Recency: "Mean recency (days)", Frequency: "Mean frequency (invoices)", Monetary: "Mean monetary value (£)" }[metric];
  svg.append(element("text", { x: left + plotWidth / 2, y: height - 7, "text-anchor": "middle", class: "axis-title" }, metricUnit));
}

function heatmap(container, labels, matrix, isCorrelation = false) {
  const size = Math.min(88, 440 / labels.length);
  const left = Math.max(145, labels.reduce((longest, label) => Math.max(longest, label.length * 8), 0));
  const top = 40;
  const width = left + labels.length * size + 35;
  const height = top + labels.length * size + 40;
  const svg = svgRoot(container, height, width);
  labels.forEach((label, index) => {
    svg.append(element("text", { x: left - 8, y: top + index * size + size / 2 + 4, "text-anchor": "end", class: "axis-text" }, label));
    svg.append(element("text", { x: left + index * size + size / 2, y: top - 12, "text-anchor": "middle", class: "axis-text" }, label));
    labels.forEach((columnLabel, column) => {
      const value = Number(matrix[index][column]);
      const cell = element("rect", {
        x: left + column * size,
        y: top + index * size,
        width: size - 2,
        height: size - 2,
        rx: 2,
        fill: isCorrelation ? correlationColor(value) : countColor(value, Math.max(...matrix.flat())),
      });
      tooltip(cell, `${label} / ${columnLabel}: ${isCorrelation ? value.toFixed(3) : numberFormat.format(value)}`);
      svg.append(cell);
      svg.append(element("text", { x: left + column * size + size / 2, y: top + index * size + size / 2 + 4, "text-anchor": "middle", class: "heat-value" }, isCorrelation ? value.toFixed(2) : numberFormat.format(value)));
    });
  });
  svg.append(element("text", { x: left + labels.length * size / 2, y: height - 5, "text-anchor": "middle", class: "axis-title" }, isCorrelation ? "Pearson correlation (unitless)" : "Predicted label · customer count"));
}

function correlationColor(value) {
  const t = (value + 1) / 2;
  const red = Math.round(238 - t * 150);
  const green = Math.round(113 + t * 95);
  const blue = Math.round(100 + t * 55);
  return `rgb(${red}, ${green}, ${blue})`;
}

function countColor(value, max) {
  const strength = Math.round(242 - value / (max || 1) * 145);
  return `rgb(${strength - 15}, ${strength}, 250)`;
}

function renderDashboard(data) {
  const { signature } = data;
  const snapshotDate = new Date(signature.snapshot_date);
  document.getElementById("snapshot-chip").textContent = `Snapshot · ${new Intl.DateTimeFormat("en-GB", { dateStyle: "medium" }).format(snapshotDate)}`;
  document.getElementById("kpi-customers").textContent = numberFormat.format(signature.n_customers);
  document.getElementById("kpi-clean-rows").textContent = numberFormat.format(signature.clean_rows);
  document.getElementById("kpi-clusters").textContent = numberFormat.format(signature.n_clusters);
  document.getElementById("selected-k").textContent = numberFormat.format(signature.n_clusters);
  document.getElementById("selected-k-caption").textContent = numberFormat.format(signature.n_clusters);
  document.getElementById("mix-customers").textContent = `${numberFormat.format(signature.n_customers)} customer records`;
  document.getElementById("kpi-silhouette").textContent = Number(signature.silhouette).toFixed(4);
  document.getElementById("kpi-accuracy").textContent = `${(signature.classifier_accuracy * 100).toFixed(2)}%`;
  document.getElementById("notice-accuracy").textContent = `${(signature.classifier_accuracy * 100).toFixed(2)}% accuracy`;
  document.getElementById("footer-sklearn").textContent = signature.sklearn_version;
  document.getElementById("top-feature").textContent = signature.top_feature;

  const mix = document.getElementById("cluster-mix");
  mix.replaceChildren();
  signature.cluster_sizes.forEach((size, cluster) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("span");
    label.textContent = `Cluster ${cluster}`;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = `bar-fill${cluster % 2 ? " alt" : ""}`;
    fill.style.width = `${size / signature.n_customers * 100}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = `${numberFormat.format(size)} · ${numberFormat.format(size / signature.n_customers * 100)}%`;
    row.append(label, track, value);
    mix.append(row);
  });

  const drawSales = () => {
    const metric = document.getElementById("sales-metric").value;
    const values = data.monthly[metric];
    lineChart(document.getElementById("sales-chart"), data.monthly.labels, values,
      metric === "revenue" ? currencyFormat.format : numberFormat.format,
      metric === "revenue" ? palette[0] : palette[2], "Month",
      metric === "revenue" ? "Sales revenue (£)" : "Unique invoices (count)");
  };
  const drawCountries = () => {
    const metric = document.getElementById("country-metric").value;
    horizontalBars(document.getElementById("country-chart"), data.countries.labels,
      data.countries[metric], metric === "revenue" ? currencyFormat.format : numberFormat.format,
      (index) => palette[index % palette.length], metric === "revenue" ? "Revenue (£)" : "Transactions (count)");
  };
  const drawClusters = () => scatterChart(document.getElementById("cluster-chart"), data.cluster_points, document.getElementById("cluster-filter").value);
  const drawProfile = () => profileBars(document.getElementById("profile-chart"), data.segments, document.getElementById("profile-metric").value);

  drawSales();
  histogram(document.getElementById("spending-chart"), data.spending.labels, data.spending.counts);
  drawCountries();
  scatterChart(document.getElementById("cluster-chart"), data.cluster_points, "all");
  drawProfile();
  heatmap(document.getElementById("correlation-chart"), ["Recency", "Frequency", "Monetary"], [
    [data.correlations.Recency.Recency, data.correlations.Frequency.Recency, data.correlations.Monetary.Recency],
    [data.correlations.Recency.Frequency, data.correlations.Frequency.Frequency, data.correlations.Monetary.Frequency],
    [data.correlations.Recency.Monetary, data.correlations.Frequency.Monetary, data.correlations.Monetary.Monetary],
  ], true);
  horizontalBars(document.getElementById("products-chart"), data.products.labels, data.products.revenue, currencyFormat.format, () => palette[2], "Sales revenue (£)");
  lineChart(document.getElementById("inertia-chart"), data.k_selection.map((row) => `k=${row.k}`), data.k_selection.map((row) => row.inertia), numberFormat.format, palette[2], "Candidate cluster count (k)", "KMeans inertia (squared distance)");
  lineChart(document.getElementById("silhouette-chart"), data.k_selection.map((row) => `k=${row.k}`), data.k_selection.map((row) => row.silhouette), (value) => value.toFixed(2), palette[3], "Candidate cluster count (k)", "Silhouette score (unitless)");
  heatmap(document.getElementById("confusion-chart"), data.confusion.labels, data.confusion.matrix, false);

  const clusterFilter = document.getElementById("cluster-filter");
  clusterFilter.querySelectorAll('option:not([value="all"])').forEach((option) => option.remove());
  [...new Set(data.cluster_points.map((point) => point.cluster))].sort((a, b) => a - b).forEach((cluster) => {
    const option = document.createElement("option");
    option.value = String(cluster);
    option.textContent = `Cluster ${cluster}`;
    clusterFilter.append(option);
  });
  document.getElementById("sales-metric").onchange = drawSales;
  document.getElementById("country-metric").onchange = drawCountries;
  clusterFilter.onchange = drawClusters;
  document.getElementById("profile-metric").onchange = drawProfile;
}

async function loadDashboard() {
  const status = document.getElementById("run-status");
  status.textContent = "Loading latest run...";
  status.classList.remove("status-error");
  try {
    const response = await fetch(`../outputs/dashboard_data.json?updated=${Date.now()}`);
    if (!response.ok) throw new Error(`Dashboard data request failed (${response.status})`);
    const data = await response.json();
    renderDashboard(data);
    status.textContent = `Run loaded · ${data.signature.n_customers.toLocaleString("en-GB")} customers`;
  } catch (error) {
    status.textContent = "Dashboard data unavailable";
    status.classList.add("status-error");
    document.querySelectorAll(".interactive-chart").forEach((container) => {
      container.textContent = "Run the project pipeline and open this page through a local web server to load current charts.";
    });
    console.error(error);
  }
}

document.getElementById("refresh-data").addEventListener("click", loadDashboard);
loadDashboard();