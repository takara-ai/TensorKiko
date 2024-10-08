import argparse
import asyncio
import os
import re
import sys
import logging
import webbrowser
import http.server
import socketserver
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Any
import torch
from anytree import Node
from safetensors import safe_open
from collections import defaultdict

@dataclass
class ModelVisualizer:
    file_path: str
    debug: bool
    no_tree: bool
    port: int
    output_dir: str
    state_dict: Dict[str, torch.Tensor] = field(default_factory=dict)
    model_tree: Any = None
    tree_data: Dict[str, Any] = field(default_factory=dict)
    layer_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    model_info: Dict[str, Any] = field(default_factory=lambda: {
        'total_params': 0,
        'memory_usage': 0,
        'precision': None,
        'estimated_flops': 0
    })

    def __post_init__(self):
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

    async def load_and_process_safetensors(self) -> None:
        if not os.path.exists(self.file_path) or not self.file_path.lower().endswith('.safetensors'):
            raise ValueError(f"Invalid file: {self.file_path}")
        try:
            with safe_open(self.file_path, framework="pt", device="cpu") as f:
                self.state_dict = {key: f.get_tensor(key) for key in f.keys()}
        except Exception as e:
            raise ValueError(f"Failed to load the safetensors file: {e}") from e

        root = Node("Model", type="root")
        for key, tensor in self.state_dict.items():
            parent = root
            for i, part in enumerate(key.split('.')):
                child = next((c for c in parent.children if c.name == part), None)
                if not child:
                    node_type = "layer" if i == len(key.split('.')) - 1 else "module"
                    child = Node(
                        part,
                        parent=parent,
                        type=node_type,
                        params=tensor.numel() if node_type == "layer" else 0,
                        shape=str(tensor.shape) if node_type == "layer" else ""
                    )
                parent = child
        self.model_tree = root

        # Calculate total parameters and update model_info
        self.model_info['total_params'] = sum(t.numel() for t in self.state_dict.values())

        for key, tensor in self.state_dict.items():
            parts = key.split('.')
            layer_type = parts[-2] if parts[-1] in {'weight', 'bias'} else parts[-1]
            self.layer_types[layer_type] += 1
            if not self.model_info['precision']:
                self.model_info['precision'] = tensor.dtype
            self.model_info['memory_usage'] += tensor.numel() * tensor.element_size()
            if 'weight' in key:
                if tensor.dim() == 4:
                    n, c, h, w = tensor.shape
                    self.model_info['estimated_flops'] += 2 * n * c * h * w * 224 * 224
                elif tensor.dim() == 2:
                    m, n = tensor.shape
                    self.model_info['estimated_flops'] += 2 * m * n
        self.model_info['memory_usage'] /= (1024 * 1024)

    def tree_to_dict(self, node: Node) -> Dict[str, Any]:
        return {
            'name': node.name,
            'type': getattr(node, 'type', 'unknown'),
            'params': getattr(node, 'params', 0),
            'shape': getattr(node, 'shape', ''),
            'children': [self.tree_to_dict(child) for child in node.children] if node.children else None
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
            html = f'<li><div class="node" data-params="{param_display}" data-shape="{shape}">{caret}{name}</div>'
            if children:
                html += f'<ul class="nested">{"".join(generate_tree_html(child) for child in children)}</ul>'
            return html + '</li>'

        calculate_totals(tree_data)
        tree_html = generate_tree_html(tree_data)
        layer_type_html = ''.join(f'<li>{layer}: {count}</li>' for layer, count in sorted(self.layer_types.items(), key=lambda x: x[1], reverse=True))

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Model Visualizer - {model_name}</title>
            <style>
                body {{ font-family: 'Arial', sans-serif; margin: 0; padding: 0; background-color: #f0f0f0; color: #333; }}
                #header {{ position: fixed; top: 0; left: 0; right: 0; background-color: rgba(255, 255, 255, 0.9); box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 10px 20px; z-index: 1000; }}
                #model-info {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; }}
                #model-details {{ flex: 1; min-width: 200px; margin-right: 20px; }}
                #layer-types-container {{ flex: 1; min-width: 200px; }}
                #layer-types {{ font-size: 0.8em; max-height: 150px; overflow-y: auto; list-style-type: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; }}
                #layer-types li {{ margin-right: 10px; margin-bottom: 5px; background-color: #e0e0e0; padding: 2px 5px; border-radius: 3px; }}
                #tree {{ padding: 20px; }}
                .tree {{ --spacing: 1.5rem; --radius: 10px; }}
                .tree li {{ display: block; position: relative; padding-left: calc(2 * var(--spacing) - var(--radius) - 2px); }}
                .tree ul {{ margin-left: calc(var(--radius) - var(--spacing)); padding-left: 0; }}
                .tree ul li {{ border-left: 2px solid #ddd; }}
                .tree ul li:last-child {{ border-color: transparent; }}
                .tree ul li::before {{ content: ''; display: block; position: absolute; top: calc(var(--spacing) / -2); left: -2px; width: calc(var(--spacing) + 2px); height: calc(var(--spacing) + 1px); border: solid #ddd; border-width: 0 0 2px 2px; }}
                .tree .node {{ display: inline-block; cursor: pointer; background-color: #fff; border: 2px solid #ddd; border-radius: var(--radius); padding: 0.5rem 1rem; margin: 0.5rem 0; transition: all 0.3s; }}
                .tree .node:hover {{ background-color: #f0f0f0; transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .tree .node.selected {{ background-color: #e6f3ff; border-color: #3498db; }}
                .caret {{ cursor: pointer; user-select: none; display: inline-block; width: 0; height: 0; margin-right: 6px; vertical-align: middle; border-top: 4px solid transparent; border-bottom: 4px solid transparent; border-left: 6px solid #333; transition: transform 0.2s; }}
                .caret-down {{ transform: rotate(90deg); }}
                .nested {{ display: none; }}
                .active {{ display: block; }}
                #layer-info {{ position: fixed; bottom: 20px; right: 20px; background-color: #fff; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: none; max-width: 300px; }}
                #search-container {{ margin-bottom: 10px; position: relative; }}
                #search {{ width: 100%; padding: 5px; }}
                #search-results {{ position: absolute; right: 5px; top: 50%; transform: translateY(-50%); font-size: 0.8em; color: #666; }}
                .highlight {{ background-color: yellow; }}
                .current-highlight {{ background-color: orange; }}
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
                        <p>Precision: {self.model_info['precision']}</p>
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

                    // Set initial top margin for tree
                    tree.style.marginTop = `${{header.offsetHeight}}px`;

                    // Update tree margin on window resize
                    window.addEventListener('resize', () => {{
                        tree.style.marginTop = `${{header.offsetHeight}}px`;
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
                            let infoHTML = `<h3>${{nodeElement.textContent.trim()}}</h3><p>Parameters: ${{params}}</p>`;
                            if (shape) {{
                                infoHTML += `<p>Shape: ${{shape}}</p><div id="shape-svg">${{generateShapeSVG(shape)}}</div>`;
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
                            const parent = node.closest('li');
                            const nestedUl = parent.querySelector('ul.nested');
                            
                            if (nodeText.includes(searchTerm)) {{
                                matchCount++;
                                node.classList.add('highlight');
                                if (!firstMatch) {{
                                    firstMatch = node;
                                    node.classList.add('current-highlight');
                                }}
                                expandParents(node);
                                if (nestedUl) {{
                                    nestedUl.classList.add('active');
                                    const caret = node.querySelector('.caret');
                                    if (caret) caret.classList.add('caret-down');
                                }}
                            }} else if (!searchTerm) {{
                                if (nestedUl) {{
                                    nestedUl.classList.remove('active');
                                    const caret = node.querySelector('.caret');
                                    if (caret) caret.classList.remove('caret-down');
                                }}
                            }}
                        }});

                        searchResults.textContent = searchTerm ? `${{matchCount}} result${{matchCount !== 1 ? 's' : ''}} found` : '';

                        if (firstMatch) {{
                            firstMatch.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                        }}
                    }});

                    function expandParents(node) {{
                        let parent = node.parentElement;
                        while (parent) {{
                            if (parent.classList.contains('nested')) {{
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

                    function generateShapeSVG(shape) {{
                        const torchMatch = shape.match(/(\d+)\s*×\s*(\d+)\s*×\s*(\d+)\s*×\s*(\d+)/);
                        if (torchMatch) {{
                            const [_, n, c, h, w] = torchMatch;
                            return `<svg width="100" height="100" viewBox="0 0 100 100"><rect x="10" y="10" width="80" height="80" fill="#f0f0f0" stroke="#333"/><text x="50" y="50" text-anchor="middle" dominant-baseline="middle" font-size="14">${{n}}</text><rect x="20" y="20" width="60" height="60" fill="#e0e0e0" stroke="#333"/><text x="50" y="50" text-anchor="middle" dominant-baseline="middle" font-size="12">${{c}}</text><rect x="30" y="30" width="40" height="40" fill="#d0d0d0" stroke="#333"/><text x="50" y="50" text-anchor="middle" dominant-baseline="middle" font-size="10">${{h}}×${{w}}</text></svg>`;
                        }}
                        const generalMatch = shape.match(/(\d+(?:\s*×\s*\d+)*)/);
                        if (generalMatch) {{
                            const dims = generalMatch[1].split('×').map(d => d.trim());
                            return `<svg width="100" height="50" viewBox="0 0 100 50"><rect x="5" y="5" width="90" height="40" fill="#f0f0f0" stroke="#333"/><text x="50" y="25" text-anchor="middle" dominant-baseline="middle" font-size="14">${{dims.join(' × ')}}</text></svg>`;
                        }}
                        return '';
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
        with open(html_file, 'w') as f:
            f.write(html_content)
        self.logger.info(f"HTML visualization generated: {html_file}")
        return html_file

    def serve_html(self, html_file: str) -> None:
        os.chdir(self.output_dir or os.path.dirname(os.path.abspath(self.file_path)))
        try:
            with socketserver.TCPServer(("localhost", self.port), http.server.SimpleHTTPRequestHandler) as httpd:
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

    async def generate_visualization(self) -> None:
        try:
            self.logger.info(f"Loading model from: {self.file_path}")
            await self.load_and_process_safetensors()
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
    asyncio.run(visualizer.generate_visualization())

if __name__ == "__main__":
    main()
