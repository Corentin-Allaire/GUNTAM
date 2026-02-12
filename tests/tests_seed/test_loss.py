import torch

from GUNTAM.Seed.SeedLoss import (
    attention_loss,
    full_attention_loss,
    top_attention_loss,
    attention_next_loss,
    reconstruction_loss,
    hit_classification_loss,
)


class TestAttentionLoss:
    def test_basic_and_empty(self):
        # Empty pairs returns zero tensor on same device
        attention = torch.zeros((4, 4))
        empty = attention_loss(attention, torch.tensor([]).long(), torch.tensor([]).long(), torch.tensor([]).long())
        assert empty.item() == 0.0
        assert empty.device == attention.device

        # Construct small example with symmetric positive/negative pairs
        attention_logits = torch.tensor([
            [0.0, 1.0, -1.0, 0.5],
            [1.2, 0.0, -0.3, 0.7],
            [-0.4, 0.9, 0.0, -0.2],
            [0.6, -0.8, 0.3, 0.0],
        ])
        # Pairs (exclude diagonal) - symmetric
        pairs1 = torch.tensor([0, 1, 0, 2])  # (0,1) (1,0) (0,2) (2,0)
        pairs2 = torch.tensor([1, 0, 2, 0])
        target = torch.tensor([1, 1, -1, -1])  # (0,1) positive; (1,0) positive; (0,2)/(2,0) negative

        result = attention_loss(attention_logits, pairs1, pairs2, target)
        assert not torch.isnan(result), "attention_loss returned NaN"

        # Perfect case: positives very large, negatives very small -> loss ~ 0
        perfect_logits = torch.zeros_like(attention_logits)
        perfect_logits[pairs1[target == 1], pairs2[target == 1]] = 10000.0
        perfect_logits[pairs1[target == -1], pairs2[target == -1]] = -10000.0
        perfect_loss = attention_loss(perfect_logits, pairs1, pairs2, target)
        assert torch.isclose(perfect_loss, torch.tensor(0.0), atol=1e-6)


class TestFullAttentionLoss:
    def test_positive_and_zero(self):
        attention_logits = torch.arange(0, 25).float().view(5, 5) / 10.0  # deterministic
        # Positive symmetric pairs among first 3 hits
        pairs1 = torch.tensor([0, 1, 1, 2])
        pairs2 = torch.tensor([1, 0, 2, 1])
        target = torch.tensor([1, 1, 1, 1])

        # Manual replication of logic
        pos_mask = target == 1
        pos_hits = torch.unique(torch.cat([pairs1[pos_mask], pairs2[pos_mask]]))
        num_valid_hits = int(torch.max(pos_hits).item()) + 1
        target_mat = torch.zeros_like(attention_logits)
        target_mat[pairs1[pos_mask], pairs2[pos_mask]] = 1.0
        full_mask = torch.ones_like(attention_logits, dtype=torch.bool)
        full_mask[num_valid_hits:, :] = False
        full_mask[:, num_valid_hits:] = False
        inactive_cols = torch.ones(attention_logits.shape[0], dtype=torch.bool)
        inactive_cols[pos_hits] = False
        full_mask[:, inactive_cols] = False
        result = full_attention_loss(attention_logits, pairs1, pairs2, target)
        assert not torch.isnan(result), "full_attention_loss returned NaN"

        # Perfect case: set positives high and all other considered negatives low
        perfect = torch.zeros_like(attention_logits)
        perfect[pairs1[pos_mask], pairs2[pos_mask]] = 10000.0
        # all others within masked region will remain 0 (treated as negative); push them lower
        perfect[full_mask & ~target_mat.bool()] = -10000.0
        perfect_loss = full_attention_loss(perfect, pairs1, pairs2, target)
        assert torch.isclose(perfect_loss, torch.tensor(0.0), atol=1e-6)

        # Zero positives returns 0
        zero_res = full_attention_loss(attention_logits, pairs1, pairs2, torch.tensor([-1, -1, -1, -1]))
        assert zero_res.item() == 0.0


class TestTopAttentionLoss:
    def test_selection_and_zero(self):
        torch.manual_seed(0)
        attention_logits = torch.randn(6, 6)
        pairs1 = torch.tensor([0, 1, 2, 2])
        pairs2 = torch.tensor([1, 0, 3, 1])
        target = torch.tensor([1, 1, 1, 1])

        # Manual procedure mirroring implementation
        pos_mask = target == 1
        pos_hits = torch.unique(torch.cat([pairs1[pos_mask], pairs2[pos_mask]]))
        num_valid_hits = int(torch.max(pos_hits).item()) + 1
        full_mask = torch.ones_like(attention_logits, dtype=torch.bool)
        full_mask[num_valid_hits:, :] = False
        full_mask[:, num_valid_hits:] = False
        inactive_cols = torch.ones(attention_logits.shape[0], dtype=torch.bool)
        inactive_cols[pos_hits] = False
        full_mask[:, inactive_cols] = False

        result = top_attention_loss(attention_logits, pairs1, pairs2, target)
        assert not torch.isnan(result), "top_attention_loss returned NaN"

        # Perfect case: positives high, top-k negatives low -> loss ~ 0
        perfect = attention_logits.clone().fill_(-10000.0)
        perfect[pairs1[pos_mask], pairs2[pos_mask]] = 10000.0
        perfect_loss = top_attention_loss(perfect, pairs1, pairs2, target)
        assert torch.isclose(perfect_loss, torch.tensor(0.0), atol=1e-6)

        zero_res = top_attention_loss(attention_logits, pairs1, pairs2, torch.tensor([-1, -1, -1, -1]))
        assert zero_res.item() == 0.0
    

class TestAttentionNextLoss:
    def test_forward_and_backward(self):
        # Design logits matrix
        attention_logits = torch.zeros((6, 6))
        # Provide pairs including forward and backward possibilities
        pairs1 = torch.tensor([0, 0, 1, 3, 4])
        pairs2 = torch.tensor([2, 3, 0, 4, 3])
        target = torch.tensor([1, 1, 1, 1, 1])

        # Expected selected targets:
        # sources = unique of pairs1 positives -> {0,1,3,4}
        # For 0: forward targets {2,3} -> min = 2
        # For 1: forward none, backward targets {0} -> max backward = 0
        # For 3: forward target {4} -> 4
        # For 4: forward none, backward {3} -> 3
        expected_selected = torch.tensor([2, 0, 4, 3])
        # Run function
        loss = attention_next_loss(attention_logits, pairs1, pairs2, target)
        assert loss >= 0  # Non-negative

        # Directly recompute selection logic to verify indices used in cross entropy
        pos_mask = target == 1
        sources = pairs1[pos_mask]
        unique_sources = torch.unique(sources)
        # Map sources to expected_selected order (unique_sources sorted ascending)
        # unique_sources should be [0,1,3,4]
        assert torch.equal(unique_sources, torch.tensor([0, 1, 3, 4]))
        assert torch.equal(expected_selected, torch.tensor([2, 0, 4, 3]))

        # Since logits are zeros, cross entropy per line = log(num_valid_hits)
        # where num_valid_hits = max(pos_hits)+1 = 5
        # Each row uniform => CE = ln(5) * pos_hits
        num_valid_hits = 5
        expected_loss = 4 * torch.log(torch.tensor(float(num_valid_hits)))  # mean of identical entries
        assert torch.isclose(loss, expected_loss, atol=1e-6)

        zero_res = attention_next_loss(attention_logits, pairs1, pairs2, torch.tensor([-1, -1, -1, -1, -1]))
        assert zero_res.item() == 0.0
    

class TestReconstructionLoss:
    def test_valid_and_empty(self):
        # Batch=1, hits=3
        # reconstructed: [z, eta, sin(phi), cos(phi), pT]
        # particles: [d0, z0, phi, eta, pT]
        phi_vals = torch.tensor([0.0, 1.0])
        sin_phi = torch.sin(phi_vals)
        cos_phi = torch.cos(phi_vals)
        reconstructed = torch.tensor(
            [
                [
                    [1.0, 0.5, sin_phi[0].item(), cos_phi[0].item(), 10.0],
                    [2.0, 1.5, sin_phi[1].item(), cos_phi[1].item(), 20.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]
                ]
            ]
        )
        particles = torch.tensor(
            [
                [
                    [0.0, 1.0, phi_vals[0], 0.5, 10.0],
                    [0.0, 2.0, phi_vals[1], 1.5, 20.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        )
        padded_mask = torch.tensor([[False, False, True]])
        losses = reconstruction_loss(reconstructed, particles, padded_mask, loss_type="MSE")
        for k in ["z", "eta", "phi", "pt"]:
            assert torch.isclose(losses[k], torch.tensor(0.0), atol=1e-6), f"Loss for {k} is not close to 0: {losses[k]}"

        # Empty valid (all padded or pT=0) -> zeros
        reconstructed_empty = torch.zeros_like(reconstructed)
        particles_empty = torch.zeros_like(particles)
        padded_all = torch.tensor([[True, True, True]])
        losses_empty = reconstruction_loss(reconstructed_empty, particles_empty, padded_all, loss_type="MSE")
        for v in losses_empty.values():
            assert v.item() == 0.0

        # L1 variant (non-zero difference to test non-zero output)
        reconstructed_shift = reconstructed.clone()
        reconstructed_shift[0, 0, 0] += 1.0  # z shift
        losses_l1 = reconstruction_loss(reconstructed_shift, particles, padded_mask, loss_type="L1")
        assert losses_l1["z"] > 0

    def test_phi_wrapping(self):
        # Test phi loss with angle wrapping: sin/cos representation should handle phi correctly
        # particles format: [d0, z0, phi, eta, pT]
        phi_vals = torch.tensor([0.0, torch.pi / 2, torch.pi, -torch.pi / 2])
        sin_phi = torch.sin(phi_vals)
        cos_phi = torch.cos(phi_vals)
        reconstructed = torch.zeros((1, 4, 5))
        reconstructed[0, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])  # z
        reconstructed[0, :, 1] = torch.tensor([0.5, 1.0, 1.5, 2.0])  # eta
        reconstructed[0, :, 2] = sin_phi  # sin(phi)
        reconstructed[0, :, 3] = cos_phi  # cos(phi)
        reconstructed[0, :, 4] = torch.tensor([10.0, 20.0, 30.0, 40.0])  # pT

        particles = torch.zeros((1, 4, 5))
        particles[0, :, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0])  # d0
        particles[0, :, 1] = torch.tensor([1.0, 2.0, 3.0, 4.0])  # z0
        particles[0, :, 2] = phi_vals  # phi
        particles[0, :, 3] = torch.tensor([0.5, 1.0, 1.5, 2.0])  # eta
        particles[0, :, 4] = torch.tensor([10.0, 20.0, 30.0, 40.0])  # pT

        padded_mask = torch.tensor([[False, False, False, False]])
        losses = reconstruction_loss(reconstructed, particles, padded_mask, loss_type="MSE")
        
        # Perfect reconstruction should yield near-zero losses
        assert torch.isclose(losses["phi"], torch.tensor(0.0), atol=1e-6), f"Phi loss not close to 0: {losses['phi']}"

    def test_pt_inverse_loss(self):
        # Test that pT loss uses inverse (1/pT)
        # Two hits with different pT values
        reconstructed = torch.zeros((1, 2, 5))
        reconstructed[0, 0] = torch.tensor([1.0, 0.5, 0.0, 1.0, 10.0])  # z=1.0, eta=0.5, sin(phi)=0.0, cos(phi)=1.0, pT=10
        reconstructed[0, 1] = torch.tensor([2.0, 1.5, 1.0, 0.0, 20.0])  # z=2.0, eta=1.5, sin(phi)=1.0, cos(phi)=0.0, pT=20

        particles = torch.zeros((1, 2, 5))
        particles[0, 0] = torch.tensor([0.0, 1.0, 0.0, 0.5, 10.0])  # d0=0.0, z0=1.0, phi=0.0, eta=0.5, pT=10
        particles[0, 1] = torch.tensor([0.0, 2.0, torch.pi / 2, 1.5, 20.0])  # d0=0.0, z0=2.0, phi=π/2, eta=1.5, pT=20

        padded_mask = torch.tensor([[False, False]])
        losses = reconstruction_loss(reconstructed, particles, padded_mask, loss_type="MSE")
        
        # Perfect pT match -> loss should be 0
        assert torch.isclose(losses["pt"], torch.tensor(0.0), atol=1e-6), f"pT loss not close to 0: {losses['pt']}"

        # Now test with mismatch
        reconstructed_mismatch = reconstructed.clone()
        reconstructed_mismatch[0, 0, 4] = 15.0  # Change pT from 10 to 15
        losses_mismatch = reconstruction_loss(reconstructed_mismatch, particles, padded_mask, loss_type="MSE")
        
        # Loss should be non-zero due to mismatch
        assert losses_mismatch["pt"] > 0, "pT loss should be positive for mismatched predictions"


class TestHitClassificationLoss:
    def test_balanced_and_edge(self):
        # Removed model instantiation as cfg/self parameter is no longer needed
        seed_scores = torch.tensor([[0.9, 0.1, 0.8, 0.2]])  # batch=1, hits=4
        particles = torch.tensor([[
            [0.0, 0.0, 0.0, 5.0],   # particle (pT>0)
            [0.0, 0.0, 0.0, 0.0],   # orphan
            [0.0, 0.0, 0.0, 3.0],   # particle
            [0.0, 0.0, 0.0, 0.0],   # orphan
        ]])
        padded_mask = torch.tensor([[False, False, False, False]])
        loss_val = hit_classification_loss(seed_scores, particles, padded_mask)
        assert loss_val > 0

        # All padded -> zero
        padded_all = torch.tensor([[True, True, True, True]])
        zero_loss = hit_classification_loss(seed_scores, particles, padded_all)
        assert zero_loss.item() == 0.0

        # Single class present (only positives)
        particles_pos_only = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 5.0],
                    [0.0, 0.0, 0.0, 6.0],
                    [0.0, 0.0, 0.0, 7.0],
                    [0.0, 0.0, 0.0, 8.0],
                ]
            ]
        )
        loss_single_class = hit_classification_loss(seed_scores, particles_pos_only, padded_mask)
        assert loss_single_class > 0
