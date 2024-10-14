# tensor_processing.py
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)

@torch.jit.script
def calculate_basic_stats(t: torch.Tensor) -> Tuple[float, float, int, int, float, float, torch.Tensor]:
    t_float = t.float()
    num_elements = t.numel()
    if num_elements == 0:
        return (0.0, 0.0, 0, 0, 0.0, 0.0, t_float)
    min_val = float(t_float.min().item())
    max_val = float(t_float.max().item())
    num_zeros = int((t == 0).sum().item())
    mean = float(t_float.mean().item())
    std = float(t_float.std().item()) if num_elements > 1 else 0.0
    return (min_val, max_val, num_zeros, num_elements, mean, std, t_float)

@torch.jit.script
def fast_histogram(tensor: torch.Tensor, num_bins: int) -> Tuple[torch.Tensor, torch.Tensor]:
    min_val, max_val = tensor.min(), tensor.max()
    if min_val == max_val:
        return torch.tensor([tensor.numel()]), torch.tensor([min_val, max_val])
    
    range_val = max_val - min_val
    bin_edges = torch.linspace(min_val, max_val, num_bins + 1)
    hist = torch.histc(tensor, bins=num_bins, min=min_val, max=max_val)
    return hist, bin_edges

def calculate_tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    min_val, max_val, num_zeros, num_elements, mean, std, tensor_float = calculate_basic_stats(tensor.cpu())
    
    stats = {
        'min': min_val,
        'max': max_val,
        'num_zeros': num_zeros,
        'num_elements': num_elements,
        'mean': mean,
        'std': std,
        'dtype': str(tensor.dtype),
    }
    
    if num_elements > 1 and min_val != max_val:
        try:
            num_bins = min(100, num_elements)
            hist, bin_edges = fast_histogram(tensor_float, num_bins)
            stats['histogram'] = [hist.tolist(), bin_edges.tolist()]
        except Exception as e:
            stats['histogram'] = f"Histogram computation failed: {str(e)}"
    else:
        stats['histogram'] = "Not enough unique elements for histogram"
    
    return stats, tensor_float

@torch.jit.script
def detect_anomalies(tensor_float: torch.Tensor, mean: float, std: float, num_elements: int) -> str:
    if torch.isfinite(tensor_float).all():
        if std == 0 and num_elements > 1:
            return 'Zero variance'
        if std != 0:
            z_scores = torch.abs((tensor_float - mean) / std)
            outliers = torch.sum(z_scores > 6).item()
            if outliers > 0:
                return f'Outliers detected: {outliers} values beyond 6 std dev'
    else:
        return 'Contains NaN or Inf values'
    return ""

def estimate_flops(shape: Tuple[int, ...], key: str) -> int:
    if 'weight' in key:
        if len(shape) == 4:  # Convolutional layer
            out_channels, in_channels, kernel_h, kernel_w = shape
            flops_per_instance = 2 * in_channels * kernel_h * kernel_w
            output_size = 1  # Placeholder; calculate based on input size, stride, padding
            return flops_per_instance * output_size * output_size * out_channels
        elif len(shape) == 2:  # Fully connected layer
            out_features, in_features = shape
            return 2 * out_features * in_features
    return 0

def count_parameters(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, float]:
    total_params = 0
    total_bytes = 0
    for tensor in state_dict.values():
        total_params += tensor.numel()
        total_bytes += get_param_size(tensor)
    memory_usage = round(total_bytes / (1024 * 1024), 2)  # Convert to MB
    return total_params, memory_usage

def get_param_size(tensor: torch.Tensor) -> int:
    if tensor.dtype == torch.qint8 or tensor.dtype == torch.quint8:
        # For quantized tensors, we need to account for scale and zero_point
        return tensor.numel() + 2 * 4  # 1 byte per element + 4 bytes each for scale and zero_point
    elif hasattr(tensor, 'quant_state'):
        # For dynamically quantized tensors (e.g., in PyTorch 2.0+)
        return tensor.numel() * tensor.element_size() + 2 * 4
    else:
        return tensor.numel() * tensor.element_size()

def dtype_to_str(dtype: torch.dtype) -> str:
    if dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        return str(dtype)
    elif dtype in [torch.qint8, torch.quint8]:
        return f"quantized({dtype})"
    elif hasattr(dtype, 'quant_state'):
        return f"dynamic_quantized({dtype})"
    else:
        return str(dtype)

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
    for key, tensor in tqdm(state_dict.items(), desc="Processing tensors", unit="tensor"):
        try:
            stats, tensor_float = calculate_tensor_stats(tensor)
            model_visualizer.tensor_stats[key] = stats

            # Update layer types
            parts = key.split('.')
            layer_type = parts[-2] if parts[-1] in {'weight', 'bias', 'running_mean', 'running_var'} else parts[-1]
            model_visualizer.layer_types[layer_type] = model_visualizer.layer_types.get(layer_type, 0) + 1

            # Estimate FLOPs
            flops = estimate_flops(tensor.shape, key)
            model_visualizer.model_info['estimated_flops'] += flops

            # Detect anomalies
            anomaly = detect_anomalies(tensor_float, stats['mean'], stats['std'], stats['num_elements'])
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