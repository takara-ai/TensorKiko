import argparse, asyncio, os, re, sys, logging, webbrowser, http.server, socketserver
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, Any
import torch
from anytree import Node
from safetensors import safe_open

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ModelVisualizer:
    def __init__(self, file_path: str, debug: bool, no_tree: bool, port: int, output_dir: str):
        self.file_path = file_path
        self.debug = debug
        self.no_tree = no_tree
        self.port = port
        self.output_dir = output_dir
        self.state_dict = {}
        self.model_tree = None
        self.tree_data = {}
        self.total_params = 0
        self.layer_types = {}
        self.model_info = {}

    async def load_safetensors(self) -> Dict[str, torch.Tensor]:
        if not os.path.exists(self.file_path) or not self.file_path.lower().endswith('.safetensors'):
            raise ValueError(f"Invalid file: {self.file_path}")
        try:
            with safe_open(self.file_path, framework="pt", device="cpu") as f:
                return {key: f.get_tensor(key) for key in f.keys()}
        except Exception as e:
            raise ValueError(f"Failed to load the safetensors file: {str(e)}") from e

    def create_tree_structure(self, state_dict: Dict[str, torch.Tensor]) -> Node:
        root = Node("Model", type="root")
        nodes = {"": root}
        for key, tensor in state_dict.items():
            parts = key.split('.')
            for i in range(len(parts)):
                parent_path, current_path = '.'.join(parts[:i]), '.'.join(parts[:i+1])
                if current_path not in nodes:
                    nodes[current_path] = Node(parts[i], parent=nodes.get(parent_path, root),
                                               type=parts[-2] if i == len(parts) - 1 else "layer",
                                               params=tensor.numel() if i == len(parts) - 1 else 0,
                                               shape=str(tensor.shape) if i == len(parts) - 1 else "")
        return root
    
    def analyze_model_structure(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, Dict[str, int]]:
        self.total_params = sum(tensor.numel() for tensor in state_dict.values())
        self.layer_types = {}
        self.model_info = {
            'total_params': self.total_params,
            'memory_usage': 0,
            'precision': None,
            'estimated_flops': 0
        }

        for key, tensor in state_dict.items():
            parts = key.split('.')
            layer_type = parts[-2] if parts[-1] in {'weight', 'bias'} else parts[-1]
            self.layer_types[layer_type] = self.layer_types.get(layer_type, 0) + 1

            # Determine precision
            if self.model_info['precision'] is None:
                self.model_info['precision'] = tensor.dtype

            # Calculate memory usage
            self.model_info['memory_usage'] += tensor.numel() * tensor.element_size()

            # Estimate FLOPs (simplified estimation)
            if 'weight' in key and len(tensor.shape) == 4:  # Assume it's a conv layer
                n, c, h, w = tensor.shape
                self.model_info['estimated_flops'] += 2 * n * c * h * w * 224 * 224  # Assuming 224x224 input
            elif 'weight' in key and len(tensor.shape) == 2:  # Assume it's a linear layer
                m, n = tensor.shape
                self.model_info['estimated_flops'] += 2 * m * n

        # Convert memory usage to MB
        self.model_info['memory_usage'] = self.model_info['memory_usage'] / (1024 * 1024)
        return self.total_params, self.layer_types


    def tree_to_dict(self, node: Node) -> Dict[str, Any]:
        return {
            'name': node.name, 'type': getattr(node, 'type', 'unknown'),
            'params': getattr(node, 'params', 0), 'shape': getattr(node, 'shape', ''),
            'children': [self.tree_to_dict(child) for child in node.children] if node.children else None
        }

    def generate_html(self, tree_data: Dict[str, Any], total_params: int, layer_types: Dict[str, int], model_name: str) -> str:
        def format_shape(shape_str: str) -> str:
            torch_match = re.match(r'torch\.Size\(\[(.*)\]\)', shape_str)
            return ' × '.join(torch_match.group(1).split(', ')) if torch_match else shape_str

        def calculate_totals(node: Dict[str, Any]) -> Tuple[int, str]:
            if not isinstance(node, dict): return 0, ""
            params, shape, children = node.get('params', 0), node.get('shape', ''), node.get('children', [])
            if children:
                child_params = sum(calculate_totals(child)[0] for child in children)
                node['total_params'] = child_params if params == 0 else child_params
            else:
                node['total_params'] = params
            return params, shape

        def generate_tree_html(node: Dict[str, Any], depth=0) -> str:
            if not isinstance(node, dict): return ""
            node_name, children = node.get('name', 'Unknown'), node.get('children', [])
            params, total_params, shape = node.get('params', 0), node.get('total_params', 0), node.get('shape', '')
            param_display = f"{params:,}" if params > 0 else f"Total: {total_params:,}"
            formatted_shape = format_shape(shape)
            caret = '<span class="caret"></span>' if children else ''
            html = f'<li><div class="node" data-params="{param_display}" data-shape="{formatted_shape}">{caret}{node_name}</div>'
            if children:
                html += f'<ul class="nested">' + ''.join(generate_tree_html(child, depth + 1) for child in children) + '</ul>'
            return html + '</li>'

        calculate_totals(tree_data)
        tree_html = generate_tree_html(tree_data)
        layer_type_html = ''.join(f'<li>{layer}: {count}</li>' for layer, count in sorted(layer_types.items(), key=lambda x: x[1], reverse=True))

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
                #search-container {{ margin-bottom: 10px; }}
                #search {{ width: 100%; padding: 5px; }}
                .highlight {{ background-color: yellow; }}
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

                    // Set initial top margin for tree
                    tree.style.marginTop = `${{header.offsetHeight}}px`;

                    // Update tree margin on window resize
                    window.addEventListener('resize', () => {{
                        tree.style.marginTop = `${{header.offsetHeight}}px`;
                    }});

                    // Collapsible tree
                    tree.addEventListener('click', function(e) {{
                        if (e.target.classList.contains('caret')) {{
                            e.target.classList.toggle('caret-down');
                            e.target.parentElement.nextElementSibling.classList.toggle('active');
                            e.stopPropagation();
                        }} else if (e.target.classList.contains('node')) {{
                            // Remove 'selected' class from all nodes
                            nodes.forEach(n => n.classList.remove('selected'));
                            // Add 'selected' class to clicked node
                            e.target.classList.add('selected');
                            const params = e.target.dataset.params;
                            const shape = e.target.dataset.shape;
                            let infoHTML = `<h3>${{e.target.textContent.trim()}}</h3><p>Parameters: ${{params}}</p>`;
                            if (shape) {{
                                infoHTML += `<p>Shape: ${{shape}}</p><div id="shape-svg">${{generateShapeSVG(shape)}}</div>`;
                            }}
                            layerInfo.innerHTML = infoHTML;
                            layerInfo.style.display = 'block';
                        }}
                    }});

                    document.addEventListener('click', (e) => {{
                        if (!e.target.closest('.node') && !e.target.closest('#layer-info')) {{
                            layerInfo.style.display = 'none';
                            // Remove 'selected' class from all nodes when clicking outside
                            nodes.forEach(n => n.classList.remove('selected'));
                        }}
                    }});

                    // Search functionality
                    searchInput.addEventListener('input', function() {{
                        const searchTerm = this.value.toLowerCase();
                        let firstMatch = null;
                        
                        nodes.forEach(node => {{
                            const nodeText = node.textContent.toLowerCase();
                            const parent = node.closest('li');
                            const nestedUl = parent.querySelector('ul.nested');
                            
                            if (nodeText.includes(searchTerm)) {{
                                if (!firstMatch) firstMatch = node;
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
        html_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_visualization.html")
        with open(html_file, 'w') as f:
            f.write(html_content)
        logger.info(f"HTML visualization generated: {html_file}")
        return html_file

    def serve_html(self, html_file: str) -> None:
        try:
            os.chdir(self.output_dir or os.path.dirname(os.path.abspath(self.file_path)))
            with socketserver.TCPServer(("localhost", self.port), http.server.SimpleHTTPRequestHandler) as httpd:
                url = f"http://localhost:{httpd.server_address[1]}/{os.path.basename(html_file)}"
                logger.info(f"Serving visualization at {url}")
                webbrowser.open(url)
                logger.info("Press Ctrl+C to stop the server and exit.")
                httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("\nServer stopped by user.")
        except Exception as server_error:
            logger.error(f"Error starting the server: {str(server_error)}")
            logger.info(f"You can manually open the HTML file: {html_file}")

    async def generate_visualization(self) -> None:
        try:
            logger.info(f"Loading model from: {self.file_path}")
            self.state_dict = await self.load_safetensors()
            with ThreadPoolExecutor() as executor:
                loop = asyncio.get_running_loop()
                self.model_tree = await loop.run_in_executor(executor, self.create_tree_structure, self.state_dict)
                await loop.run_in_executor(executor, self.analyze_model_structure, self.state_dict)
            self.tree_data = self.tree_to_dict(self.model_tree)
            model_name = os.path.splitext(os.path.basename(self.file_path))[0]
            html_content = self.generate_html(self.tree_data, self.total_params, self.layer_types, model_name)
            html_file = self.save_html(html_content, model_name)
            self.serve_html(html_file)
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            if self.debug: import traceback; traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze the structure of a model stored in a safetensors file.")
    parser.add_argument("file_path", help="Path to the safetensors file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for detailed error information")
    parser.add_argument("--no-tree", action="store_true", help="Disable tree visualization")
    parser.add_argument("--port", type=int, default=8000, help="Port number for the local HTTP server (default: 8000)")
    parser.add_argument("--output-dir", type=str, default="", help="Directory to save the HTML visualization")
    args = parser.parse_args()

    visualizer = ModelVisualizer(
        file_path=args.file_path,
        debug=args.debug,
        no_tree=args.no_tree,
        port=args.port,
        output_dir=args.output_dir
    )
    asyncio.run(visualizer.generate_visualization())

if __name__ == "__main__":
    main()