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
from anytree import Node, RenderTree
from safetensors import safe_open
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm  # For progress bars

@dataclass
class ModelVisualizer:
    file_path: str
    debug: bool = False
    no_tree: bool = False
    port: int = 8000
    output_dir: str = ''
    state_dict: Dict[str, torch.Tensor] = field(default_factory=dict)
    model_tree: Optional[Node] = None
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
        if not os.path.exists(self.file_path) or not self.file_path.lower().endswith('.safetensors'):
            raise ValueError(f"Invalid file: {self.file_path}")
        try:
            with safe_open(self.file_path, framework="pt", device="cpu") as f:
                self.state_dict = {key: f.get_tensor(key) for key in f.keys()}
            self.logger.debug(f"Loaded {len(self.state_dict)} tensors from {self.file_path}")
        except Exception as e:
            raise ValueError(f"Failed to load the safetensors file: {e}") from e

        root = Node("Model", type="root", full_name="Model")
        for key, tensor in self.state_dict.items():
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
                        "full_name": full_name
                    }
                    child = Node(part, parent=parent, **node_attrs)
                parent = child
        self.model_tree = root

        # Calculate total parameters and update model_info
        self.model_info['total_params'] = sum(t.numel() for t in self.state_dict.values())

        # Prepare for parallel processing
        def process_tensor(key_tensor):
            key, tensor = key_tensor
            parts = key.split('.')
            layer_type = parts[-2] if parts[-1] in {'weight', 'bias', 'running_mean', 'running_var'} else parts[-1]
            self.layer_types[layer_type] += 1
            self.model_info['precisions'].add(str(tensor.dtype))
            self.model_info['memory_usage'] += tensor.numel() * tensor.element_size()

            # Enhanced FLOPs estimation
            if 'weight' in key:
                if tensor.dim() == 4:
                    n, c, h, w = tensor.shape
                    input_size = 224  # Placeholder; consider making this configurable
                    self.model_info['estimated_flops'] += 2 * n * c * h * w * input_size * input_size
                elif tensor.dim() == 2:
                    m, n = tensor.shape
                    self.model_info['estimated_flops'] += 2 * m * n

            # Calculate tensor statistics with overflow handling
            try:
                # Convert to float64 to prevent overflow
                tensor_data = tensor.cpu().numpy().astype(np.float64)

                # Suppress overflow warnings temporarily
                with np.errstate(over='raise', invalid='raise'):
                    hist_counts, bin_edges = np.histogram(tensor_data, bins=50)
                    stats = {
                        'mean': float(np.mean(tensor_data)),
                        'std': float(np.std(tensor_data)),
                        'min': float(np.min(tensor_data)),
                        'max': float(np.max(tensor_data)),
                        'num_zeros': int(np.sum(tensor_data == 0)),
                        'num_elements': tensor.numel(),
                        'histogram': [hist_counts.tolist(), bin_edges.tolist()]
                    }

                # Anomaly detection
                if np.isnan(tensor_data).any() or np.isinf(tensor_data).any():
                    self.anomalies[key] = 'Contains NaN or Inf values'
                elif stats['std'] == 0:
                    self.anomalies[key] = 'Zero variance'
                elif stats['max'] > 1e6 or stats['min'] < -1e6:
                    self.anomalies[key] = 'Extreme values detected'

                self.tensor_stats[key] = stats

            except FloatingPointError:
                self.logger.warning(f"Overflow encountered while processing tensor: {key}")
                self.anomalies[key] = 'Overflow encountered during statistics calculation'
                # Optionally, set stats to None or a default value
                self.tensor_stats[key] = {
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'num_zeros': None,
                    'num_elements': tensor.numel(),
                    'histogram': None
                }

        # Use ThreadPoolExecutor with progress bar
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Create a list of key-tensor pairs
            key_tensor_pairs = list(self.state_dict.items())

            # Use tqdm for a progress bar
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
            return ' × '.join(s.strip('torch.Size([])').split(',')) if s.startswith('torch.Size') else s

        def calculate_totals(node: Dict[str, Any]) -> int:
            if not node.get('children'):
                node['total_params'] = node.get('params', 0)
                return node['total_params']
            node['total_params'] = sum(calculate_totals(child) for child in node['children'])
            return node['total_params']

        def generate_tree_html(node: Dict[str, Any]) -> str:
            name, children = node.get('name', 'Unknown'), node.get('children', [])
            params, total_params, shape = node.get('params', 0), node.get('total_params', 0), format_shape(node.get('shape', ''))
            param_display = f"{params:,}" if params > 0 else f"Total: {total_params:,}"
            caret = '<span class="caret"></span>' if children else ''
            anomaly = self.anomalies.get(node.get('full_name', ''), '')
            anomaly_class = 'anomaly' if anomaly else ''
            # Include full_name as a data attribute for accurate key mapping
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

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Model Visualizer - {model_name}</title>
            <style>
                /* CSS styles remain unchanged for brevity */
                body {{
                    font-family: 'Arial', sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f0f0f0;
                    color: #333;
                }}
                #header {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    background-color: rgba(255, 255, 255, 0.95);
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 10px 20px;
                    z-index: 1000;
                    display: flex;
                    flex-direction: column;
                }}
                #model-info {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                    align-items: flex-start;
                }}
                #model-details, #layer-types-container {{
                    flex: 1;
                    min-width: 200px;
                    margin-right: 20px;
                }}
                #layer-types {{
                    font-size: 0.9em;
                    max-height: 150px;
                    overflow-y: auto;
                    list-style-type: none;
                    padding: 0;
                    margin: 0;
                    display: flex;
                    flex-wrap: wrap;
                }}
                #layer-types li {{
                    margin-right: 10px;
                    margin-bottom: 5px;
                    background-color: #e0e0e0;
                    padding: 2px 5px;
                    border-radius: 3px;
                }}
                #search-container {{
                    margin-top: 10px;
                    position: relative;
                }}
                #search {{
                    width: 100%;
                    padding: 5px;
                }}
                #search-results {{
                    position: absolute;
                    right: 5px;
                    top: 50%;
                    transform: translateY(-50%);
                    font-size: 0.8em;
                    color: #666;
                }}
                #tree {{
                    padding: 20px;
                }}
                .tree {{
                    --spacing: 1.5rem;
                    --radius: 10px;
                }}
                .tree li {{
                    display: block;
                    position: relative;
                    padding-left: calc(2 * var(--spacing) - var(--radius) - 2px);
                }}
                .tree ul {{
                    margin-left: calc(var(--radius) - var(--spacing));
                    padding-left: 0;
                }}
                .tree ul li {{
                    border-left: 2px solid #ddd;
                }}
                .tree ul li:last-child {{
                    border-color: transparent;
                }}
                .tree ul li::before {{
                    content: '';
                    display: block;
                    position: absolute;
                    top: calc(var(--spacing) / -2);
                    left: -2px;
                    width: calc(var(--spacing) + 2px);
                    height: calc(var(--spacing) + 1px);
                    border: solid #ddd;
                    border-width: 0 0 2px 2px;
                }}
                .tree .node {{
                    display: inline-block;
                    cursor: pointer;
                    background-color: #fff;
                    border: 2px solid #ddd;
                    border-radius: var(--radius);
                    padding: 0.5rem 1rem;
                    margin: 0.5rem 0;
                    transition: all 0.3s;
                }}
                .tree .node:hover {{
                    background-color: #f0f0f0;
                    transform: translateY(-2px);
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .tree .node.selected {{
                    background-color: #e6f3ff;
                    border-color: #3498db;
                }}
                .caret {{
                    cursor: pointer;
                    user-select: none;
                    display: inline-block;
                    width: 0;
                    height: 0;
                    margin-right: 6px;
                    vertical-align: middle;
                    border-top: 4px solid transparent;
                    border-bottom: 4px solid transparent;
                    border-left: 6px solid #333;
                    transition: transform 0.2s;
                }}
                .caret-down {{
                    transform: rotate(90deg);
                }}
                .nested {{
                    display: none;
                }}
                .active {{
                    display: block;
                }}
                #layer-info {{
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    background-color: #fff;
                    border-radius: 10px;
                    padding: 15px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    display: none;
                    max-width: 300px;
                    overflow: auto;
                    max-height: 80vh;
                    z-index: 1001;
                }}
                .highlight {{
                    background-color: yellow;
                }}
                .current-highlight {{
                    background-color: orange;
                }}
                .anomaly {{
                    background-color: #ffe6e6;
                }}
                #histogram-container {{
                    margin-top: 20px;
                }}
                .histogram-bar {{
                    display: inline-block;
                    width: 2px;
                    background-color: #3498db;
                    vertical-align: bottom;
                    margin-right: 1px;
                }}
                #anomaly-info {{
                    color: red;
                    font-weight: bold;
                }}
            </style>
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

                // Set initial top margin for tree
                tree.style.marginTop = `${{header.offsetHeight + 20}}px`;

                // Update tree margin on window resize
                window.addEventListener('resize', () => {{
                    tree.style.marginTop = `${{header.offsetHeight + 20}}px`;
                }});

                // Combined functionality for expanding/collapsing and showing info
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
                            infoHTML += `<p>Shape: ${{shape}}</p><div id="shape-svg">${{generateShapeSVG(shape)}}</div>`;
                        }}
                        if (tensorStats[key]) {{
                            const stats = tensorStats[key];
                            if (stats.mean !== null) {{
                                infoHTML += `<h4>Statistics:</h4><p>Mean: ${{stats.mean.toFixed(4)}}, Std: ${{stats.std.toFixed(4)}}, Min: ${{stats.min.toFixed(4)}}, Max: ${{stats.max.toFixed(4)}}, Zeros: ${{stats.num_zeros}}</p>`;
                                infoHTML += `<div id="histogram-container">${{generateHistogram(stats.histogram)}}</div>`;
                            }} else {{
                                infoHTML += `<p>Statistics: Unable to calculate due to overflow.</p>`;
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

                // Enhanced search functionality
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

                function generateShapeSVG(shape) {{
                    if (!shape) {{
                        return '<p>No shape information available.</p>';
                    }}
                    const dims = shape.split(' × ').map(Number);
                    if (dims.length === 4) {{
                        const [n, c, h, w] = dims;
                        return `<svg width="100" height="100" viewBox="0 0 100 100">
                                    <rect x="10" y="10" width="80" height="80" fill="#f0f0f0" stroke="#333"/>
                                    <text x="50" y="30" text-anchor="middle" font-size="12">N: ${{n}}</text>
                                    <text x="50" y="50" text-anchor="middle" font-size="12">C: ${{c}}</text>
                                    <text x="50" y="70" text-anchor="middle" font-size="12">H×W: ${{h}}×${{w}}</text>
                                </svg>`;
                    }} else if (dims.length === 2) {{
                        const [m, n] = dims;
                        return `<svg width="100" height="50" viewBox="0 0 100 50">
                                    <rect x="5" y="5" width="90" height="40" fill="#f0f0f0" stroke="#333"/>
                                    <text x="50" y="25" text-anchor="middle" font-size="14">${{m}} × ${{n}}</text>
                                </svg>`;
                    }} else {{
                        return `<svg width="100" height="50" viewBox="0 0 100 50">
                                    <rect x="5" y="5" width="90" height="40" fill="#f0f0f0" stroke="#333"/>
                                    <text x="50" y="25" text-anchor="middle" font-size="12">${{shape}}</text>
                                </svg>`;
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """

    def save_html(self, html_content: str, model_name: str) -> str:
        output_dir = self.output_dir or os.path.dirname(os.path.abspath(self.file_path))
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        html_file = os.path.join(output_dir, f"{base_name}_visualization.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        self.logger.info(f"HTML visualization generated: {html_file}")
        return html_file

    def serve_html(self, html_file: str) -> None:
        output_dir = self.output_dir or os.path.dirname(os.path.abspath(self.file_path))
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
            self.logger.info(f"Loading model from: {self.file_path}")
            self.load_and_process_safetensors()
            if not self.no_tree:
                self.tree_data = self.tree_to_dict(self.model_tree)
            model_name = os.path.splitext(os.path.basename(self.file_path))[0]
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
    parser = argparse.ArgumentParser(description="Visualize and analyze a safetensors model.")
    parser.add_argument("file_path", help="Path to the safetensors file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-tree", action="store_true", help="Disable tree visualization")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port (default: 8000)")
    parser.add_argument("--output-dir", default="", help="Directory to save the HTML")
    args = parser.parse_args()

    visualizer = ModelVisualizer(**vars(args))
    visualizer.generate_visualization()

if __name__ == "__main__":
    main()
