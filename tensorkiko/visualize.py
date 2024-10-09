import argparse
import os
import sys
import logging
import webbrowser
import http.server
import socketserver
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import torch
from anytree import Node
from safetensors import safe_open
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm  # For progress bars
import re

@dataclass
class ModelVisualizer:
    file_paths: List[str]
    debug: bool = False
    no_tree: bool = False
    port: int = 8000
    output_dir: str = ''
    include_layers: Optional[str] = None
    exclude_layers: Optional[str] = None
    state_dicts: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    model_trees: Dict[str, Any] = field(default_factory=dict)
    tree_data: Dict[str, Any] = field(default_factory=dict)
    layer_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    model_info: Dict[str, Any] = field(default_factory=lambda: {
        'total_params': 0,
        'memory_usage': 0,
        'precisions': set(),
        'estimated_flops': 0
    })
    tensor_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    anomalies: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_and_process_safetensors(self) -> None:
        for file_path in self.file_paths:
            if not os.path.exists(file_path) or not file_path.lower().endswith('.safetensors'):
                self.logger.error(f"Invalid file: {file_path}")
                continue
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    state_dict = {key: f.get_tensor(key) for key in f.keys()}
                self.logger.debug(f"Loaded {len(state_dict)} tensors from {file_path}")
                self.state_dicts[file_path] = state_dict
            except Exception as e:
                self.logger.error(f"Failed to load the safetensors file {file_path}: {e}")
                continue

            root = Node("Model", type="root", full_name="Model")
            for key, tensor in state_dict.items():
                if self.include_layers and not re.search(self.include_layers, key):
                    continue
                if self.exclude_layers and re.search(self.exclude_layers, key):
                    continue

                parent = root
                parts = key.split('.')
                full_name = ""
                for i, part in enumerate(parts):
                    full_name = f"{full_name}.{part}" if full_name else part
                    child = next((c for c in parent.children if c.name == part), None)
                    if not child:
                        node_type = "layer" if i == len(parts) - 1 else "module"
                        node_attrs = {
                            "type": node_type,
                            "params": tensor.numel() if node_type == "layer" else 0,
                            "shape": str(tensor.shape) if node_type == "layer" else "",
                            "full_name": full_name,
                        }
                        child = Node(part, parent=parent, **node_attrs)
                    parent = child
            self.model_trees[file_path] = root

            # Update model_info
            self.update_model_info(state_dict)

            # Process tensors
            self.process_tensors(state_dict)

    def update_model_info(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model_info['total_params'] += sum(t.numel() for t in state_dict.values())
        self.model_info['memory_usage'] += sum(t.numel() * t.element_size() for t in state_dict.values())
        self.model_info['precisions'].update(str(t.dtype) for t in state_dict.values())

    def process_tensors(self, state_dict: Dict[str, torch.Tensor]) -> None:
        def process_tensor(key_tensor):
            key, tensor = key_tensor
            if self.include_layers and not re.search(self.include_layers, key):
                return
            if self.exclude_layers and re.search(self.exclude_layers, key):
                return

            parts = key.split('.')
            layer_type = parts[-2] if parts[-1] in {'weight', 'bias', 'running_mean', 'running_var'} else parts[-1]
            self.layer_types[layer_type] += 1

            # Enhanced FLOPs estimation
            if 'weight' in key:
                if tensor.dim() == 4:
                    n, c, h, w = tensor.shape
                    input_size = 224  # Placeholder; consider making this configurable
                    self.model_info['estimated_flops'] += 2 * n * c * h * w * input_size * input_size
                elif tensor.dim() == 2:
                    m, n = tensor.shape
                    self.model_info['estimated_flops'] += 2 * m * n

            # Calculate tensor statistics
            tensor_data = tensor.cpu().numpy()
            stats = self.calculate_tensor_stats(tensor_data)
            self.tensor_stats[key] = stats

            # Advanced anomaly detection
            anomaly = self.detect_anomalies(tensor_data, stats)
            if anomaly:
                self.anomalies[key] = anomaly

        # Use ThreadPoolExecutor with progress bar
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            key_tensor_pairs = list(state_dict.items())
            futures = {executor.submit(process_tensor, kt): kt for kt in key_tensor_pairs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tensors"):
                try:
                    future.result()
                except Exception as e:
                    kt = futures[future]
                    self.logger.error(f"Error processing tensor {kt[0]}: {e}")
                    self.anomalies[kt[0]] = f"Error during processing: {e}"
                    self.tensor_stats[kt[0]] = {
                        'mean': None,
                        'std': None,
                        'min': None,
                        'max': None,
                        'num_zeros': None,
                        'num_elements': kt[1].numel(),
                        'histogram': None
                    }

        self.model_info['memory_usage'] /= (1024 * 1024)  # Convert to MB

    def calculate_tensor_stats(self, tensor_data: np.ndarray) -> Dict[str, Any]:
        stats = {
            'mean': float(np.mean(tensor_data)),
            'std': float(np.std(tensor_data)),
            'min': float(np.min(tensor_data)),
            'max': float(np.max(tensor_data)),
            'num_zeros': int(np.sum(tensor_data == 0)),
            'num_elements': tensor_data.size,
            'histogram': None  # Histograms are optional for large tensors
        }
        # Only compute histogram if tensor is small to avoid performance issues
        if tensor_data.size <= 1e6:
            hist_counts, bin_edges = np.histogram(tensor_data, bins=50)
            stats['histogram'] = [hist_counts.tolist(), bin_edges.tolist()]
        return stats

    def detect_anomalies(self, tensor_data: np.ndarray, stats: Dict[str, Any]) -> Optional[str]:
        if np.isnan(tensor_data).any() or np.isinf(tensor_data).any():
            return 'Contains NaN or Inf values'
        elif stats['std'] == 0:
            return 'Zero variance'
        else:
            # Advanced anomaly detection using z-scores
            if stats['std'] != 0:
                z_scores = np.abs((tensor_data - stats['mean']) / stats['std'])
                outliers = np.where(z_scores > 6)[0]  # Threshold of 6 standard deviations
                if len(outliers) > 0:
                    return f'Outliers detected: {len(outliers)} values beyond 6 std dev'
            else:
                return 'Standard deviation is zero; cannot compute z-scores'
        return None

    def tree_to_dict(self, node: Node) -> Dict[str, Any]:
        return {
            'name': node.name,
            'type': getattr(node, 'type', 'unknown'),
            'params': getattr(node, 'params', 0),
            'shape': getattr(node, 'shape', ''),
            'full_name': getattr(node, 'full_name', ''),
            'children': [self.tree_to_dict(child) for child in node.children] if node.children else []
        }

    def generate_html(self, tree_data: Dict[str, Any], model_name: str) -> str:
        
        def format_shape(s):
            m = re.match(r'torch\.Size\(\[(.*)\]\)', s)
            return ' × '.join(m.group(1).split(', ')) if m else s

        def calculate_totals(node):
            if not node.get('children'):
                node['total_params'] = node.get('params', 0)
                return node['total_params']
            node['total_params'] = sum(calculate_totals(child) for child in node['children'])
            return node['total_params']

        def generate_tree_html(node):
            if not isinstance(node, dict):
                return ""
            name, children = node.get('name', 'Unknown'), node.get('children', [])
            params, total_params, shape = node.get('params', 0), node.get('total_params', 0), format_shape(node.get('shape', ''))
            param_display = f"{params:,}" if params > 0 else f"Total: {total_params:,}"
            caret = '<span class="caret"></span>' if children else ''
            anomaly = self.anomalies.get(node.get('full_name', ''), '')
            anomaly_class = 'anomaly' if anomaly else ''
            html = f'<li><div class="node {anomaly_class}" data-params="{param_display}" data-shape="{shape}" data-name="{name}" data-fullname="{node.get("full_name", "")}">{caret}{name}</div>'
            if children:
                html += f'<ul class="nested">{"".join(generate_tree_html(child) for child in children)}</ul>'
            return html + '</li>'

        calculate_totals(tree_data)
        tree_html = generate_tree_html(tree_data) if not self.no_tree else "<p>Tree visualization is disabled.</p>"
        layer_type_html = ''.join(f'<li>{layer}: {count}</li>' for layer, count in sorted(self.layer_types.items(), key=lambda x: x[1], reverse=True))

        # Prepare tensor stats and anomalies for JavaScript
        tensor_stats_json = json.dumps(self.tensor_stats)
        anomalies_json = json.dumps(self.anomalies)

        # Aggregate precisions
        precisions = ', '.join(self.model_info['precisions'])

        # HTML content with linked CSS and embedded JavaScript
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Model Visualizer - {model_name}</title>
            <!-- Link to external CSS -->
            <link rel="stylesheet" type="text/css" href="styles.css">
        </head>
        <body>
            <div id="header">
                <h1>{model_name}</h1>
                <div id="model-info">
                    <div id="model-details">
                        <h3>Model Details</h3>
                        <p>Total Parameters: {self.model_info['total_params']:,}</p>
                        <p>Memory Usage: {self.model_info['memory_usage']:.2f} MB</p>
                        <p>Precisions: {precisions}</p>
                        <p>Estimated FLOPs: {self.model_info['estimated_flops']:,}</p>
                    </div>
                    <div id="layer-types-container">
                        <h3>Layer Types</h3>
                        <ul id="layer-types">{layer_type_html}</ul>
                    </div>
                </div>
                <div id="search-container">
                    <input type="text" id="search" placeholder="Search for layers...">
                    <span id="search-results"></span>
                </div>
            </div>
            <div id="tree"><ul class="tree">{tree_html}</ul></div>
            <div id="layer-info"></div>
            <script>
                document.addEventListener('DOMContentLoaded', () => {{
                    const header = document.getElementById('header');
                    const tree = document.getElementById('tree');
                    const nodes = document.querySelectorAll('.node');
                    const layerInfo = document.getElementById('layer-info');
                    const searchInput = document.getElementById('search');
                    const searchResults = document.getElementById('search-results');
                    const tensorStats = {tensor_stats_json};
                    const anomalies = {anomalies_json};

                    // Function to generate SVG for tensor shapes
                    function generateShapeSVG(shape) {{
                        if (!shape) {{
                            return '<p>No shape information available.</p>';
                        }}
                        // Match shapes in the format "N × C × H × W"
                        const torchMatch = shape.match(/(\\d+)\\s*×\\s*(\\d+)\\s*×\\s*(\\d+)\\s*×\\s*(\\d+)/);
                        if (torchMatch) {{
                            const [_, n, c, h, w] = torchMatch;
                            return `
                                <svg width="100" height="100" viewBox="0 0 100 100">
                                    <rect x="10" y="10" width="80" height="80" fill="#f0f0f0" stroke="#333"/>
                                    <text x="50" y="30" text-anchor="middle" font-size="12">N: ${{n}}</text>
                                    <text x="50" y="50" text-anchor="middle" font-size="12">C: ${{c}}</text>
                                    <text x="50" y="70" text-anchor="middle" font-size="12">H×W: ${{h}}×${{w}}</text>
                                </svg>
                            `;
                        }}

                        // Match general shapes with any number of dimensions
                        const generalMatch = shape.match(/(\\d+(?:\\s*×\\s*\\d+)*)/);
                        if (generalMatch) {{
                            const dims = generalMatch[1].split('×').map(d => d.trim());
                            const dimsText = dims.join(' × ');
                            return `
                                <svg width="200" height="50" viewBox="0 0 200 50">
                                    <rect x="5" y="5" width="190" height="40" fill="#f0f0f0" stroke="#333"/>
                                    <text x="100" y="30" text-anchor="middle" font-size="14">${{dimsText}}</text>
                                </svg>
                            `;
                        }}

                        // If shape does not match expected formats
                        return '<p>No shape information available.</p>';
                    }}

                    // Function to generate histogram HTML
                    function generateHistogram(histogramData) {{
                        if (!histogramData) {{
                            return '<p>No histogram available.</p>';
                        }}
                        const [counts, bins] = histogramData;
                        const maxCount = Math.max(...counts);
                        const histogramHTML = counts.map(count => {{
                            const height = (count / maxCount) * 100;
                            return `<div class="histogram-bar" style="height: ${{height}}px;"></div>`;
                        }}).join('');
                        return `<div style="display: flex; align-items: flex-end; height: 100px;">${{histogramHTML}}</div>`;
                    }}

                    // Set initial top margin for tree
                    tree.style.marginTop = `${{header.offsetHeight + 20}}px`;

                    // Update tree margin on window resize
                    window.addEventListener('resize', () => {{
                        tree.style.marginTop = `${{header.offsetHeight + 20}}px`;
                    }});

                    // Node click event
                    tree.addEventListener('click', function(e) {{
                        if (e.target.classList.contains('caret') || e.target.classList.contains('node')) {{
                            const nodeElement = e.target.classList.contains('caret') ? e.target.parentElement : e.target;
                            const nestedUl = nodeElement.nextElementSibling;

                            // Toggle expand/collapse
                            if (nestedUl) {{
                                nestedUl.classList.toggle('active');
                                const caret = nodeElement.querySelector('.caret');
                                if (caret) caret.classList.toggle('caret-down');
                            }}

                            // Show node info
                            nodes.forEach(n => n.classList.remove('selected'));
                            nodeElement.classList.add('selected');
                            const params = nodeElement.dataset.params;
                            const shape = nodeElement.dataset.shape;
                            const name = nodeElement.dataset.name;
                            const key = nodeElement.dataset.fullname;
                            let infoHTML = `<h3>${{name}}</h3><p>Parameters: ${{params}}</p>`;
                            if (shape) {{
                                // Use SVG for shape representation
                                const shapeSVG = generateShapeSVG(shape);
                                infoHTML += `<div class="shape-svg">${{shapeSVG}}</div>`;
                            }}
                            if (tensorStats[key]) {{
                                const stats = tensorStats[key];
                                if (stats.mean !== null) {{
                                    infoHTML += `<h4>Statistics:</h4><p>Mean: ${{stats.mean.toFixed(4)}}, Std: ${{stats.std.toFixed(4)}}, Min: ${{stats.min.toFixed(4)}}, Max: ${{stats.max.toFixed(4)}}, Zeros: ${{stats.num_zeros}}</p>`;
                                    if (stats.histogram) {{
                                        infoHTML += `<div id="histogram-container">${{generateHistogram(stats.histogram)}}</div>`;
                                    }}
                                }} else {{
                                    infoHTML += `<p>Statistics: Unable to calculate.</p>`;
                                }}
                            }}
                            if (anomalies[key]) {{
                                infoHTML += `<div id="anomaly-info">Anomaly Detected: ${{anomalies[key]}}</div>`;
                            }}
                            layerInfo.innerHTML = infoHTML;
                            layerInfo.style.display = 'block';

                            e.stopPropagation();
                        }}
                    }});

                    document.addEventListener('click', (e) => {{
                        if (!e.target.closest('.node') && !e.target.closest('#layer-info')) {{
                            layerInfo.style.display = 'none';
                            nodes.forEach(n => n.classList.remove('selected'));
                        }}
                    }});

                    // Search functionality
                    searchInput.addEventListener('input', function() {{
                        const searchTerm = this.value.toLowerCase();
                        let matchCount = 0;
                        let firstMatch = null;

                        nodes.forEach(node => {{
                            node.classList.remove('highlight', 'current-highlight');
                            const nodeText = node.textContent.toLowerCase();

                            if (nodeText.includes(searchTerm)) {{
                                matchCount++;
                                node.classList.add('highlight');
                                if (!firstMatch) {{
                                    firstMatch = node;
                                    node.classList.add('current-highlight');
                                }}
                                expandParents(node);
                            }} else {{
                                node.classList.remove('highlight', 'current-highlight');
                            }}
                        }});

                        searchResults.textContent = searchTerm ? `${{matchCount}} result${{matchCount !== 1 ? 's' : ''}} found` : '';

                        if (firstMatch) {{
                            firstMatch.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                        }}
                    }});

                    function expandParents(node) {{
                        let parent = node.parentElement.parentElement;
                        while (parent && parent.id !== 'tree') {{
                            if (parent.tagName.toLowerCase() === 'ul') {{
                                parent.classList.add('active');
                                const parentNode = parent.previousElementSibling;
                                if (parentNode && parentNode.classList.contains('node')) {{
                                    const caret = parentNode.querySelector('.caret');
                                    if (caret) caret.classList.add('caret-down');
                                }}
                            }}
                            parent = parent.parentElement;
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """

        return html_content


    def save_html(self, html_content: str, model_name: str) -> str:
        output_dir = self.output_dir or os.path.dirname(os.path.abspath(self.file_paths[0]))
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(model_name))[0]
        html_file = os.path.join(output_dir, f"{base_name}_visualization.html")
        
        # Path to the CSS file
        css_file = os.path.join(output_dir, "styles.css")
        
        # Write the HTML file
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        self.logger.info(f"HTML visualization generated: {html_file}")
        
        # Write the CSS file if it doesn't exist
        if not os.path.exists(css_file):
            with open(css_file, 'w', encoding='utf-8') as f:
                f.write("""/* Basic styles for the Model Visualizer */

body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
}

#header {
    position: fixed;
    top: 0;
    width: 100%;
    background-color: #333;
    color: white;
    padding: 10px;
    z-index: 1000;
}

#header h1 {
    margin: 0;
    font-size: 24px;
}

#model-info {
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
}

#model-details, #layer-types-container {
    background-color: #444;
    padding: 10px;
    border-radius: 5px;
}

#model-details h3, #layer-types-container h3 {
    margin-top: 0;
}

#search-container {
    margin-top: 10px;
}

#search {
    width: 100%;
    padding: 8px;
    border-radius: 4px;
    border: none;
}

.tree {
    list-style-type: none;
    padding-left: 20px;
}

.node {
    cursor: pointer;
    position: relative;
    padding: 5px 0;
}

.caret {
    user-select: none;
    margin-right: 6px;
    border-top: 5px solid;
    border-right: 5px solid transparent;
    border-left: 5px solid transparent;
    display: inline-block;
    transform: rotate(0deg);
    transition: transform 0.3s;
}

.caret-down {
    transform: rotate(90deg);
}

.nested {
    display: none;
    list-style-type: none;
    padding-left: 20px;
}

.nested.active {
    display: block;
}

.selected {
    background-color: #555;
    border-radius: 4px;
}

.highlight {
    background-color: yellow;
}

.current-highlight {
    background-color: orange;
}

.shape-svg {
    margin-top: 10px;
}

.histogram-bar {
    width: 3px;
    background-color: #4CAF50;
    margin-right: 1px;
}

.anomaly {
    border-left: 4px solid red;
}

#layer-info {
    position: fixed;
    top: 100px;
    right: 20px;
    width: 300px;
    max-height: 80%;
    overflow-y: auto;
    background-color: #f9f9f9;
    border: 1px solid #ddd;
    padding: 15px;
    border-radius: 5px;
    display: none;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}

#anomaly-info {
    color: red;
    font-weight: bold;
}
""")
            self.logger.info(f"CSS file created: {css_file}")
        
        return html_file


    def serve_html(self, html_file: str) -> None:
        output_dir = self.output_dir or os.path.dirname(os.path.abspath(self.file_paths[0]))
        os.chdir(output_dir)
        try:
            handler = http.server.SimpleHTTPRequestHandler
            with socketserver.TCPServer(("", self.port), handler) as httpd:
                url = f"http://localhost:{self.port}/{os.path.basename(html_file)}"
                self.logger.info(f"Serving visualization at {url}")
                webbrowser.open(url)
                self.logger.info("Press Ctrl+C to stop the server and exit.")
                httpd.serve_forever()
        except KeyboardInterrupt:
            self.logger.info("\nServer stopped by user.")
        except Exception as e:
            self.logger.error(f"Error starting the server: {e}")
            self.logger.info(f"You can manually open the HTML file: {html_file}")

    def generate_visualization(self) -> None:
        try:
            self.logger.info("Loading models...")
            self.load_and_process_safetensors()
            for file_path, model_tree in self.model_trees.items():
                if not self.no_tree:
                    self.tree_data = self.tree_to_dict(model_tree)
                else:
                    self.tree_data = {}
                model_name = os.path.basename(file_path)
                html_content = self.generate_html(self.tree_data, model_name)
                html_file = self.save_html(html_content, model_name)
                self.serve_html(html_file)
        except Exception as e:
            self.logger.error(f"Error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze safetensors models.")
    parser.add_argument("file_paths", nargs='+', help="Paths to the safetensors files")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-tree", action="store_true", help="Disable tree visualization")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port (default: 8000)")
    parser.add_argument("--output-dir", default="", help="Directory to save the HTML")
    parser.add_argument("--include-layers", help="Regex to include specific layers")
    parser.add_argument("--exclude-layers", help="Regex to exclude specific layers")
    args = parser.parse_args()

    visualizer = ModelVisualizer(
        file_paths=args.file_paths,
        debug=args.debug,
        no_tree=args.no_tree,
        port=args.port,
        output_dir=args.output_dir,
        include_layers=args.include_layers,
        exclude_layers=args.exclude_layers
    )
    visualizer.generate_visualization()

if __name__ == "__main__":
    main()
