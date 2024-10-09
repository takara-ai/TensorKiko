import re
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional
import torch
import numpy as np
from tqdm import tqdm

def process_tensors(model_visualizer, state_dict: Dict[str, torch.Tensor]) -> None:
    def safe_to_numpy(tensor):
        return tensor.cpu().numpy() if tensor.dtype != torch.bfloat16 else tensor.to(torch.float32).cpu().numpy()

    def calculate_tensor_stats(tensor_data: np.ndarray) -> Dict[str, Any]:
        stats = {
            'min': float(np.min(tensor_data)),
            'max': float(np.max(tensor_data)),
            'num_zeros': int(np.sum(tensor_data == 0)),
            'num_elements': tensor_data.size,
        }

        with np.errstate(all='ignore'):
            stats['mean'] = float(np.mean(tensor_data))
            stats['std'] = float(np.std(tensor_data))

        if tensor_data.size <= 1e6:
            hist_counts, bin_edges = np.histogram(tensor_data, bins='auto')
            stats['histogram'] = [hist_counts.tolist(), bin_edges.tolist()]

        return stats

    def detect_anomalies(tensor_data: np.ndarray, stats: Dict[str, Any]) -> Optional[str]:
        if np.isnan(tensor_data).any() or np.isinf(tensor_data).any():
            return 'Contains NaN or Inf values'
        if stats['std'] == 0:
            return 'Zero variance'
        if stats['std'] != 0:
            z_scores = np.abs((tensor_data - stats['mean']) / stats['std'])
            outliers = np.sum(z_scores > 6)
            if outliers > 0:
                return f'Outliers detected: {outliers} values beyond 6 std dev'
        return None

    def process_tensor(key, tensor):
        if ((model_visualizer.include_layers and not re.search(model_visualizer.include_layers, key)) or
            (model_visualizer.exclude_layers and re.search(model_visualizer.exclude_layers, key))):
            return

        parts = key.split('.')
        layer_type = parts[-2] if parts[-1] in {'weight', 'bias', 'running_mean', 'running_var'} else parts[-1]
        model_visualizer.layer_types[layer_type] += 1

        if 'weight' in key:
            if tensor.dim() == 4:
                n, c, h, w = tensor.shape
                input_size = 224
                model_visualizer.model_info['estimated_flops'] += 2 * n * c * h * w * input_size * input_size
            elif tensor.dim() == 2:
                m, n = tensor.shape
                model_visualizer.model_info['estimated_flops'] += 2 * m * n

        try:
            tensor_data = safe_to_numpy(tensor)
            stats = calculate_tensor_stats(tensor_data)
            model_visualizer.tensor_stats[key] = stats

            anomaly = detect_anomalies(tensor_data, stats)
            if anomaly:
                model_visualizer.anomalies[key] = anomaly
        except Exception as e:
            model_visualizer.logger.error(f"Error processing tensor {key}: {e}")
            model_visualizer.anomalies[key] = f"Error during processing: {e}"
            model_visualizer.tensor_stats[key] = {
                'mean': None, 'std': None, 'min': None, 'max': None,
                'num_zeros': None, 'num_elements': tensor.numel(), 'histogram': None
            }

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_tensor, key, tensor) for key, tensor in state_dict.items()]
        list(tqdm(as_completed(futures), total=len(futures), desc="Processing tensors"))

    model_visualizer.model_info['memory_usage'] /= (1024 * 1024)  # Convert to MB