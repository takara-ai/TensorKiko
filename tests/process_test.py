import unittest
import torch
import cProfile
import pstats
import io
from modules.tensor_processing import process_tensors

class MockModelVisualizer:
    def __init__(self):
        self.tensor_stats = {}
        self.layer_types = {}
        self.model_info = {'estimated_flops': 0}
        self.anomalies = {}
        self.debug = False

class TestLargeScaleTensorProcessing(unittest.TestCase):
    def setUp(self):
        self.model_visualizer = MockModelVisualizer()
        
        # Simulate a 1B parameter language model
        # Assuming 32-bit floats, 1B parameters ≈ 4GB of memory
        self.state_dict = {
            # Large embedding layer
            'embedding.weight': torch.randn(250000, 1024),  # 250K vocab size, 1024 embedding dim
            
            # Multiple transformer layers
            **{f'transformer.layer.{i}.attention.self.query.weight': torch.randn(1024, 1024) for i in range(24)},
            **{f'transformer.layer.{i}.attention.self.key.weight': torch.randn(1024, 1024) for i in range(24)},
            **{f'transformer.layer.{i}.attention.self.value.weight': torch.randn(1024, 1024) for i in range(24)},
            **{f'transformer.layer.{i}.attention.output.dense.weight': torch.randn(1024, 1024) for i in range(24)},
            **{f'transformer.layer.{i}.intermediate.dense.weight': torch.randn(4096, 1024) for i in range(24)},
            **{f'transformer.layer.{i}.output.dense.weight': torch.randn(1024, 4096) for i in range(24)},
            
            # Layer norms
            **{f'transformer.layer.{i}.attention.output.LayerNorm.weight': torch.randn(1024) for i in range(24)},
            **{f'transformer.layer.{i}.output.LayerNorm.weight': torch.randn(1024) for i in range(24)},
            
            # Output layer
            'lm_head.weight': torch.randn(250000, 1024),  # Tied with embedding usually, but included for completeness
        }
        
        # Calculate and print total parameters
        total_params = sum(tensor.numel() for tensor in self.state_dict.values())
        print(f"Total parameters: {total_params:,}")

    def test_process_tensors(self):
        process_tensors(self.model_visualizer, self.state_dict)
        
        # Basic assertions
        self.assertEqual(len(self.model_visualizer.tensor_stats), len(self.state_dict))
        for key, stats in self.model_visualizer.tensor_stats.items():
            self.assertIn('min', stats)
            self.assertIn('max', stats)
            self.assertIn('mean', stats)
            self.assertIn('std', stats)
            self.assertIn('histogram', stats)

    def test_performance(self):
        # Profile the process_tensors function
        pr = cProfile.Profile()
        pr.enable()
        process_tensors(self.model_visualizer, self.state_dict)
        pr.disable()

        # Sort stats by cumulative time
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Print top 10 time-consuming functions
        print(s.getvalue())

if __name__ == '__main__':
    unittest.main(verbosity=2)