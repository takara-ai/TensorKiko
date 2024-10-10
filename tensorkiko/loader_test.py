import unittest
import os
import tempfile
import torch
import tensorflow as tf
from model_loader import load_model, convert_to_safetensors, is_supported_format
from safetensors.torch import save_file

class TestModelLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()

    def setUp(self):
        # Create dummy models for testing
        self.pytorch_model = torch.nn.Linear(10, 5)
        self.tensorflow_model = tf.keras.Sequential([tf.keras.layers.Dense(5, input_shape=(10,))])

        # Save dummy models
        self.pytorch_path = os.path.join(self.temp_dir, 'pytorch_model.pt')
        torch.save(self.pytorch_model.state_dict(), self.pytorch_path)

        self.tensorflow_path = os.path.join(self.temp_dir, 'tensorflow_model.h5')
        self.tensorflow_model.save(self.tensorflow_path, save_format='keras')

        self.safetensors_path = os.path.join(self.temp_dir, 'safetensors_model.safetensors')
        save_file(self.pytorch_model.state_dict(), self.safetensors_path)

    def test_load_pytorch_model(self):
        state_dict, converted_path = load_model(self.pytorch_path)
        self.assertIsNotNone(state_dict)
        self.assertEqual(set(state_dict.keys()), set(self.pytorch_model.state_dict().keys()))

    def test_load_tensorflow_model(self):
        state_dict, converted_path = load_model(self.tensorflow_path)
        self.assertIsNotNone(state_dict)
        self.assertEqual(len(state_dict), len(self.tensorflow_model.weights))

    def test_load_safetensors_model(self):
        state_dict, converted_path = load_model(self.safetensors_path)
        self.assertIsNotNone(state_dict)
        self.assertEqual(converted_path, self.safetensors_path)

    def test_convert_to_safetensors(self):
        state_dict = self.pytorch_model.state_dict()
        output_path = os.path.join(self.temp_dir, 'converted.safetensors')
        converted_path = convert_to_safetensors(state_dict, output_path)
        self.assertIsNotNone(converted_path)
        self.assertTrue(os.path.exists(converted_path))

    def test_is_supported_format(self):
        self.assertTrue(is_supported_format(self.pytorch_path))
        self.assertTrue(is_supported_format(self.tensorflow_path))
        self.assertTrue(is_supported_format(self.safetensors_path))
        self.assertFalse(is_supported_format('unsupported.txt'))

    def tearDown(self):
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.temp_dir)

if __name__ == '__main__':
    unittest.main()