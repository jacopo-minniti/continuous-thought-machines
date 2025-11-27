import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ctm import ContinuousThoughtMachine

class TestCTMAblation(unittest.TestCase):
    def setUp(self):
        self.common_args = {
            'iterations': 5,
            'd_model': 32,
            'd_input': 16,
            'heads': 2,
            'n_synch_out': 4,
            'n_synch_action': 4,
            'synapse_depth': 1,
            'memory_length': 5,
            'deep_nlms': False,
            'memory_hidden_dims': 8,
            'do_layernorm_nlm': False,
            'backbone_type': 'none',
            'positional_embedding_type': 'none',
            'out_dims': 10,
            'prediction_reshaper': [-1],
            'dropout': 0.0,
            'neuron_select_type': 'random-pairing',
            'n_random_pairing_self': 0,
        }
        self.input_tensor = torch.randn(2, 16, 1, 1) # B, C, H, W (dummy for none backbone)
        # Actually for 'none' backbone it expects input that matches d_input after some processing?
        # Let's check compute_features.
        # initial_rgb is Identity. backbone is Identity.
        # pos_emb is 0.
        # combined_features = (x + 0).flatten(2).transpose(1, 2)
        # kv_proj expects d_input.
        # So input should be (B, d_input, 1, 1) or similar.
        self.input_tensor = torch.randn(2, 16, 1, 1)

    def test_ablation_none(self):
        model = ContinuousThoughtMachine(**self.common_args, ablation_type='none')
        predictions, certainties, synchronisation = model(self.input_tensor)
        self.assertIsNotNone(model.latest_retention)
        # Check retention is not all zeros (unless initialized that way, but sigmoid output should be > 0)
        self.assertTrue((model.latest_retention > 0).any())
        print("Ablation 'none' passed.")

    def test_ablation_no_retention(self):
        model = ContinuousThoughtMachine(**self.common_args, ablation_type='no_retention')
        predictions, certainties, synchronisation = model(self.input_tensor)
        self.assertIsNotNone(model.latest_retention)
        # Check retention IS all zeros
        self.assertTrue((model.latest_retention == 0).all())
        print("Ablation 'no_retention' passed.")

    def test_ablation_no_loss(self):
        # In the model, 'no_loss' behaves like 'none' (normal forward pass).
        # The difference is in the training loop (loss calculation).
        # Here we just verify it runs and produces retentions.
        model = ContinuousThoughtMachine(**self.common_args, ablation_type='no_loss')
        predictions, certainties, synchronisation = model(self.input_tensor)
        self.assertIsNotNone(model.latest_retention)
        self.assertTrue((model.latest_retention > 0).any())
        print("Ablation 'no_loss' (model forward) passed.")

if __name__ == '__main__':
    unittest.main()
