// static/js/scripts.js
document.addEventListener("DOMContentLoaded", () => {
  const header = document.getElementById("header");
  const toggleHeader = document.getElementById("toggle-header");
  const tree = document.getElementById("tree");
  const nodes = document.querySelectorAll(".node");
  const layerInfo = document.getElementById("layer-info");
  const searchInput = document.getElementById("search");
  const searchResults = document.getElementById("search-results");
  const tensorStats = window.tensorStats; // Injected via backend
  const anomalies = window.anomalies; // Injected via backend

  // Toggle header expansion
  toggleHeader.addEventListener("click", () => {
    header.classList.toggle("expanded");
    updateTreeMargin();
  });

  // Update tree margin
  function updateTreeMargin() {
    const headerHeight = header.offsetHeight;
    tree.style.marginTop = `${headerHeight + 20}px`;
  }

  // Initial margin update
  updateTreeMargin();

  // Update tree margin on window resize
  window.addEventListener("resize", updateTreeMargin);

  // Function to generate SVG for tensor shapes
  function generateShapeSVG(shape) {
    if (!shape) {
      return "<p>No shape information available.</p>";
    }
    // Match shapes in the format "N × C × H × W"
    const torchMatch = shape.match(/(\d+)\s*×\s*(\d+)\s*×\s*(\d+)\s*×\s*(\d+)/);
    if (torchMatch) {
      const [_, n, c, h, w] = torchMatch;
      return `
                <svg width="100" height="100" viewBox="0 0 100 100">
                    <rect x="10" y="10" width="80" height="80" fill="#f0f0f0" stroke="#333"/>
                    <text x="50" y="30" text-anchor="middle" font-size="12">N: ${n}</text>
                    <text x="50" y="50" text-anchor="middle" font-size="12">C: ${c}</text>
                    <text x="50" y="70" text-anchor="middle" font-size="12">H×W: ${h}×${w}</text>
                </svg>
            `;
    }

    // Match general shapes with any number of dimensions
    const generalMatch = shape.match(/(\d+(?:\s*×\s*\d+)*)/);
    if (generalMatch) {
      const dims = generalMatch[1].split("×").map((d) => d.trim());
      const dimsText = dims.join(" × ");
      return `
                <svg width="200" height="50" viewBox="0 0 200 50">
                    <rect x="5" y="5" width="190" height="40" fill="#f0f0f0" stroke="#333"/>
                    <text x="100" y="30" text-anchor="middle" font-size="14">${dimsText}</text>
                </svg>
            `;
    }

    // If shape does not match expected formats
    return "<p>No shape information available.</p>";
  }

  // Function to generate histogram HTML
  function generateHistogram(histogramData) {
    if (!histogramData) {
      return "<p>No histogram available.</p>";
    }
    const [counts, bins] = histogramData;
    const maxCount = Math.max(...counts);
    const histogramHTML = counts
      .map((count) => {
        const height = (count / maxCount) * 100;
        return `<div class="histogram-bar" style="height: ${height}px;"></div>`;
      })
      .join("");
    return `<div style="display: flex; align-items: flex-end; height: 100px;">${histogramHTML}</div>`;
  }

  // Set initial top margin for tree
  tree.style.marginTop = `${header.offsetHeight + 20}px`;

  // Update tree margin on window resize
  window.addEventListener("resize", () => {
    tree.style.marginTop = `${header.offsetHeight + 20}px`;
  });

  // Node click event
  tree.addEventListener("click", function (e) {
    if (
      e.target.classList.contains("caret") ||
      e.target.classList.contains("node")
    ) {
      const nodeElement = e.target.classList.contains("caret")
        ? e.target.parentElement
        : e.target;
      const nestedUl = nodeElement.nextElementSibling;

      // Toggle expand/collapse
      if (nestedUl) {
        nestedUl.classList.toggle("active");
        const caret = nodeElement.querySelector(".caret");
        if (caret) caret.classList.toggle("caret-down");
      }

      // Show node info
      nodes.forEach((n) => n.classList.remove("selected"));
      nodeElement.classList.add("selected");
      const params = nodeElement.dataset.params;
      const shape = nodeElement.dataset.shape;
      const name = nodeElement.dataset.name;
      const key = nodeElement.dataset.fullname;
      let infoHTML = `<h3>${name}</h3><p>Parameters: ${params}</p>`;
      if (shape) {
        // Use SVG for shape representation
        const shapeSVG = generateShapeSVG(shape);
        infoHTML += `<div class="shape-svg">${shapeSVG}</div>`;
      }
      if (tensorStats[key]) {
        const stats = tensorStats[key];
        if (stats.mean !== null) {
          infoHTML += `<h4>Statistics:</h4><p>Mean: ${stats.mean.toFixed(
            4
          )}, Std: ${stats.std.toFixed(4)}, Min: ${stats.min.toFixed(
            4
          )}, Max: ${stats.max.toFixed(4)}, Zeros: ${stats.num_zeros}</p>`;
          if (stats.histogram) {
            infoHTML += `<div id="histogram-container">${generateHistogram(
              stats.histogram
            )}</div>`;
          }
        } else {
          infoHTML += `<p>Statistics: Unable to calculate.</p>`;
        }
      }
      if (anomalies[key]) {
        infoHTML += `<div id="anomaly-info">Anomaly Detected: ${anomalies[key]}</div>`;
      }
      layerInfo.innerHTML = infoHTML;
      layerInfo.style.display = "block";

      e.stopPropagation();
    }
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".node") && !e.target.closest("#layer-info")) {
      layerInfo.style.display = "none";
      nodes.forEach((n) => n.classList.remove("selected"));
    }
  });

  // Search functionality
  searchInput.addEventListener("input", function () {
    const searchTerm = this.value.toLowerCase();
    let matchCount = 0;
    let firstMatch = null;

    nodes.forEach((node) => {
      node.classList.remove("highlight", "current-highlight");
      const nodeText = node.textContent.toLowerCase();

      if (nodeText.includes(searchTerm)) {
        matchCount++;
        node.classList.add("highlight");
        if (!firstMatch) {
          firstMatch = node;
          node.classList.add("current-highlight");
        }
        expandParents(node);
      } else {
        node.classList.remove("highlight", "current-highlight");
      }
    });

    searchResults.textContent = searchTerm
      ? `${matchCount} result${matchCount !== 1 ? "s" : ""} found`
      : "";

    if (firstMatch) {
      firstMatch.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });

  function expandParents(node) {
    let parent = node.parentElement.parentElement;
    while (parent && parent.id !== "tree") {
      if (parent.tagName.toLowerCase() === "ul") {
        parent.classList.add("active");
        const parentNode = parent.previousElementSibling;
        if (parentNode && parentNode.classList.contains("node")) {
          const caret = parentNode.querySelector(".caret");
          if (caret) caret.classList.add("caret-down");
        }
      }
      parent = parent.parentElement;
    }
  }
});
