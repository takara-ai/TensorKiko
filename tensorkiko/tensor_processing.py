import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torch.nn.functional as F

def process_tensors(model_visualizer, state_dict: Dict[str, torch.Tensor]) -> None:
    logger = logging.getLogger(__name__)
    
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
        # Move tensor to CPU if it's not already there
        tensor = tensor.cpu()
        
        min_val, max_val, num_zeros, num_elements, mean, std = calculate_basic_stats(tensor)
        
        stats = {
            'min': min_val,
            'max': max_val,
            'num_zeros': num_zeros,
            'num_elements': num_elements,
            'mean': mean,
            'std': std,
        }
        
        # Compute histogram
        try:
            if num_elements > 1:  # Ensure we have at least 2 elements
                if min_val != max_val:  # Ensure we have a range of values
                    # Use numpy for histogram calculation to handle edge cases better
                    import numpy as np
                    tensor_np = tensor.numpy().flatten()
                    hist, bin_edges = np.histogram(tensor_np, bins=100)
                    stats['histogram'] = [hist.tolist(), bin_edges.tolist()]
                else:
                    stats['histogram'] = "All values are identical"
            else:
                stats['histogram'] = "Not enough elements for histogram"
        except Exception as e:
            stats['histogram'] = f"Histogram computation failed: {str(e)}"
        
        return stats

    def detect_anomalies(tensor: torch.Tensor, stats: Dict[str, Any]) -> str:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return 'Contains NaN or Inf values'
        if stats['std'] == 0:
            return 'Zero variance'
        if stats['std'] != 0:
            z_scores = torch.abs((tensor - stats['mean']) / stats['std'])
            outliers = torch.sum(z_scores > 6).item()
            if outliers > 0:
                return f'Outliers detected: {outliers} values beyond 6 std dev'
        return None

    def estimate_flops(tensor: torch.Tensor, key: str) -> int:
        if 'weight' in key:
            if tensor.dim() == 4:  # Convolutional layer
                n, c, h, w = tensor.shape
                # Assume square input, size based on tensor shape
                input_size = max(h, w)
                flops = 2 * n * c * h * w * input_size * input_size
                return flops
            elif tensor.dim() == 2:  # Fully connected layer
                m, n = tensor.shape
                return 2 * m * n
        return 0

    if model_visualizer.debug:
        logger.info(f"Processing {len(state_dict)} tensors...")
    
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

        except Exception as e:
            if model_visualizer.debug:
                logger.error(f"Error processing tensor {key}: {e}")

    if model_visualizer.debug:
        logger.info(f"Processed {len(model_visualizer.tensor_stats)} tensors.")
        logger.info(f"Layer types: {model_visualizer.layer_types}")
        logger.info(f"Estimated total FLOPs: {model_visualizer.model_info['estimated_flops']:,}")
        logger.info(f"Number of anomalies detected: {len(model_visualizer.anomalies)}")

# New function to demonstrate mixed precision and DataLoader usage
def evaluate_model(model, test_loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad(), autocast(enabled=True):
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy