import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torch.nn.functional as F

logger = logging.getLogger(__name__)

@torch.jit.script
def calculate_basic_stats(t: torch.Tensor) -> Tuple[float, float, int, int, float, float]:
    t_float = t.float()
    num_elements = t.numel()
    if num_elements == 0:
        return (0.0, 0.0, 0, 0, 0.0, 0.0)
    min_val = float(t_float.min().item())
    max_val = float(t_float.max().item())
    num_zeros = int((t == 0).sum().item())
    mean = float(t_float.mean().item())
    std = float(t_float.std().item()) if num_elements > 1 else 0.0
    return (min_val, max_val, num_zeros, num_elements, mean, std)

def calculate_tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    tensor = tensor.cpu()
    
    min_val, max_val, num_zeros, num_elements, mean, std = calculate_basic_stats(tensor)
    
    stats = {
        'min': min_val,
        'max': max_val,
        'num_zeros': num_zeros,
        'num_elements': num_elements,
        'mean': mean,
        'std': std,
        'dtype': str(tensor.dtype),
    }
    
    try:
        if num_elements > 1 and min_val != max_val:
            tensor_float = tensor.to(torch.float32)
            
            num_bins = min(100, num_elements)
            hist, bin_edges = torch.histogram(tensor_float, bins=num_bins)
            stats['histogram'] = [hist.tolist(), bin_edges.tolist()]
        else:
            stats['histogram'] = "Not enough unique elements for histogram"
    except Exception as e:
        stats['histogram'] = f"Histogram computation failed: {str(e)}"
    
    return stats

def detect_anomalies(tensor: torch.Tensor, stats: Dict[str, Any]) -> str:
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return 'Contains NaN or Inf values'
    if stats['std'] == 0 and stats['num_elements'] > 1:
        return 'Zero variance'
    if stats['std'] != 0:
        z_scores = torch.abs((tensor.float() - stats['mean']) / stats['std'])
        outliers = torch.sum(z_scores > 6).item()
        if outliers > 0:
            return f'Outliers detected: {outliers} values beyond 6 std dev'
    return None

def estimate_flops(tensor: torch.Tensor, key: str) -> int:
    if 'weight' in key:
        if tensor.dim() == 4:  # Convolutional layer
            out_channels, in_channels, kernel_h, kernel_w = tensor.shape
            # Retrieve stride and padding from model if possible
            # For this example, assume stride=1, padding=0
            flops_per_instance = 2 * in_channels * kernel_h * kernel_w
            output_size = 1  # Placeholder; calculate based on input size, stride, padding
            flops = flops_per_instance * output_size * output_size * out_channels
            return flops
        elif tensor.dim() == 2:  # Fully connected layer
            out_features, in_features = tensor.shape
            return 2 * out_features * in_features
    return 0

def get_param_size(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()

def process_tensors(model_visualizer, state_dict: Dict[str, torch.Tensor]) -> None:
    if model_visualizer.debug:
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())
        logger.propagate = False

    if model_visualizer.debug:
        logger.info(f"Processing {len(state_dict)} tensors...")
    
    total_params = 0
    for key, tensor in state_dict.items():
        try:
            stats = calculate_tensor_stats(tensor)
            model_visualizer.tensor_stats[key] = stats

            if model_visualizer.debug:
                if isinstance(stats['histogram'], str):
                    logger.info(f"Histogram for {key}: {stats['histogram']}")
                else:
                    logger.info(f"Histogram computed for {key}")

            # Update layer types
            parts = key.split('.')
            layer_type = parts[-2] if parts[-1] in {'weight', 'bias', 'running_mean', 'running_var'} else parts[-1]
            model_visualizer.layer_types[layer_type] = model_visualizer.layer_types.get(layer_type, 0) + 1

            # Estimate FLOPs
            flops = estimate_flops(tensor, key)
            model_visualizer.model_info['estimated_flops'] += flops
            if model_visualizer.debug and flops > 0:
                logger.debug(f"Layer {key}: {flops:,} estimated FLOPs")

            # Calculate parameter size
            param_size = get_param_size(tensor)
            total_params += param_size

            # Detect anomalies
            anomaly = detect_anomalies(tensor, stats)
            if anomaly:
                model_visualizer.anomalies[key] = anomaly

        except Exception as e:
            if model_visualizer.debug:
                logger.error(f"Error processing tensor {key}: {e}")

    model_visualizer.model_info['total_params'] = total_params

    if model_visualizer.debug:
        logger.info(f"Processed {len(model_visualizer.tensor_stats)} tensors.")
        logger.info(f"Layer types: {model_visualizer.layer_types}")
        logger.info(f"Estimated total FLOPs: {model_visualizer.model_info['estimated_flops']:,}")
        logger.info(f"Total parameters: {total_params:,} bytes")
        logger.info(f"Number of anomalies detected: {len(model_visualizer.anomalies)}")
