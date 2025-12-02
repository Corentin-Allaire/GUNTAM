import math
import pytest
import torch

from GUNTAM.Transformer.Transformer import (
    manual_scaled_dot_product_attention,
    MultiHeadAttention,
    TransformerFeedForward,
    EncoderLayer,
    TransformerEncoder,
)


class TestScaledDotProductAttention:
    """Test suite for scaled dot-product attention mechanism.

    Tests basic attention computation, masking, and edge cases.
    """

    def test_basic_attention(self):
        """Test basic attention computation without mask.

        Validates output shapes, attention weight normalization, and numerical stability.
        """
        batch_size, num_heads, seq_len, d_k = 2, 4, 8, 16

        q = torch.randn(batch_size, num_heads, seq_len, d_k)
        k = torch.randn(batch_size, num_heads, seq_len, d_k)
        v = torch.randn(batch_size, num_heads, seq_len, d_k)

        output, logits = manual_scaled_dot_product_attention(q, k, v)

        # Check output and logits shapes
        assert output.shape == (batch_size, num_heads, seq_len, d_k)
        assert logits.shape == (batch_size, num_heads, seq_len, seq_len)

        # No NaNs in logits or outputs
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isnan(logits).any(), "Attention logits contain NaN values"

        # Softmax over logits should give valid distributions
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(
            probs.sum(dim=-1),
            torch.ones(batch_size, num_heads, seq_len),
            atol=1e-6,
        )
        assert torch.all(probs >= 0)

    def test_attention_with_mask(self):
        """Test attention computation with mask.

        Validates proper masking behavior and output shapes.
        """
        batch_size, num_heads, seq_len, d_k = 2, 4, 8, 16

        q = torch.randn(batch_size, num_heads, seq_len, d_k)
        k = torch.randn(batch_size, num_heads, seq_len, d_k)
        v = torch.randn(batch_size, num_heads, seq_len, d_k)

        # Mask last 2 query positions (True = masked)
        mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, dtype=torch.bool)
        mask[:, :, :, -2:] = True

        output, logits = manual_scaled_dot_product_attention(q, k, v, mask)

        # Shapes
        assert output.shape == (batch_size, num_heads, seq_len, d_k)
        assert logits.shape == (batch_size, num_heads, seq_len, seq_len)

        # Masked query rows in logits should be -inf everywhere
        assert torch.isinf(logits[:, :, :, -2:]).all()

        # Unmasked query rows should be finite
        assert torch.isfinite(logits[:, :, :, :-2]).all()

        # For unmasked rows, softmax(logits) should be valid distributions
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(
            probs.sum(dim=-1),
            torch.ones(batch_size, num_heads, seq_len),
            atol=1e-6,
        )

        # Sanity: no NaNs in outputs
        assert not torch.isnan(logits).any(), "Attention logits contain NaN values"

    def test_manual_vs_pytorch_attention_equivalence(self):
        """Manual attention should match PyTorch scaled_dot_product_attention."""
        batch_size, num_heads, seq_len, d_k = 2, 3, 7, 16

        torch.manual_seed(123)
        q = torch.randn(batch_size, num_heads, seq_len, d_k)
        k = torch.randn(batch_size, num_heads, seq_len, d_k)
        v = torch.randn(batch_size, num_heads, seq_len, d_k)

        # Build a boolean mask: True = masked (blocked) for the manual implementation
        mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, dtype=torch.bool)
        mask[:, :, :, -2:] = True  # mask last 2 query rows

        # Manual implementation
        out_manual, logits_manual = manual_scaled_dot_product_attention(q, k, v, mask)

        # PyTorch implementation expects inverted boolean (True = keep)
        attn_mask = ~mask
        out_torch = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )

        # Compare outputs
        assert torch.allclose(out_manual, out_torch, atol=1e-6)

        # Recompute expected logits to compare explicitly
        scale = 1.0 / math.sqrt(d_k)
        logits_expected = torch.matmul(q, k.transpose(-2, -1)) * scale
        logits_expected = logits_expected.masked_fill(mask, float("-inf"))
        assert torch.allclose(logits_manual, logits_expected, atol=1e-6)

class TestMultiHeadAttention:
    """Test suite for multi-head attention layer."""

    @pytest.mark.parametrize("use_pytorch", [False, True])
    def test_initialization(self, use_pytorch):
        """Test proper initialization of multi-head attention (manual & PyTorch)."""
        input_dim, model_dim, num_heads = 512, 256, 8
        mha = MultiHeadAttention(input_dim, model_dim, num_heads, use_pytorch=use_pytorch)

        assert mha.input_dim == input_dim
        assert mha.model_dim == model_dim
        assert mha.num_heads == num_heads
        assert mha.head_dim == model_dim // num_heads
        assert mha.use_pytorch is use_pytorch

    def test_initialization_error(self):
        """Test that initialization raises error when model_dim not divisible by num_heads."""
        with pytest.raises(ValueError):
            MultiHeadAttention(512, 257, 8, use_pytorch=False)  # 257 is not divisible by 8

    @pytest.mark.parametrize("use_pytorch", [False, True])
    def test_forward_pass(self, use_pytorch):
        """Test forward pass through multi-head attention (manual & PyTorch)."""
        batch_size, seq_len, input_dim = 2, 10, 512
        model_dim, num_heads = 256, 1

        mha = MultiHeadAttention(input_dim, model_dim, num_heads, use_pytorch=use_pytorch)
        x = torch.randn(batch_size, seq_len, input_dim)

        output, attention_weights = mha(x)

        # Check output shapes
        assert output.shape == (batch_size, seq_len, input_dim)
        if use_pytorch:
            # PyTorch flash attention path returns no attention weights
            assert attention_weights is None
        else:
            assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
            assert not torch.isnan(attention_weights).any(), "Attention weights contain NaN values"

        # Check for NaN values
        assert not torch.isnan(output).any(), "Output contains NaN values"

    @pytest.mark.parametrize("use_pytorch", [False, True])
    def test_forward_with_mask(self, use_pytorch):
        """Test forward pass with mask (manual & PyTorch)."""
        batch_size, seq_len, input_dim = 2, 10, 512
        model_dim, num_heads = 256, 1

        mha = MultiHeadAttention(input_dim, model_dim, num_heads, use_pytorch=use_pytorch)
        x = torch.randn(batch_size, seq_len, input_dim)

        # Mask last 2 query positions (True = masked) with shape (B, S).
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, -2:] = True

        output, attention_weights = mha(x, mask)

        # Check output shapes
        assert output.shape == (batch_size, seq_len, input_dim)
        if use_pytorch:
            assert attention_weights is None
        else:
            assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
            assert not torch.isnan(attention_weights).any(), "Attention weights contain NaN values"

        # Sanity: no NaNs in outputs
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_mask_dim3_valid(self):
        """dim=3 mask `(B, S, S)` should be accepted and work."""
        batch_size, seq_len, input_dim = 2, 10, 512
        model_dim, num_heads = 256, 1

        mha = MultiHeadAttention(input_dim, model_dim, num_heads, use_pytorch=False)
        x = torch.randn(batch_size, seq_len, input_dim)

        mask_dim3 = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask_dim3[:, :, -2:] = True

        out3, attn3 = mha(x, mask_dim3)

        assert out3.shape == (batch_size, seq_len, input_dim)
        assert attn3.shape == (batch_size, num_heads, seq_len, seq_len)
        assert not torch.isnan(out3).any()
        assert not torch.isnan(attn3).any()

    def test_mask_dim3_equivalence_with_dim2(self):
        """dim=3 mask `(B, S, S)` should match behavior of dim=2 mask."""
        batch_size, seq_len, input_dim = 2, 10, 512
        model_dim, num_heads = 256, 1

        mha = MultiHeadAttention(input_dim, model_dim, num_heads, dropout=0, use_pytorch=False)
        x = torch.randn(batch_size, seq_len, input_dim)

        mask_dim2 = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask_dim2[:, -2:] = True

        mask_dim3 = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask_dim3[:, :, -2:] = True

        out2, attn2 = mha(x, mask_dim2)
        out3, attn3 = mha(x, mask_dim3)

        assert torch.allclose(out2, out3, atol=1e-6)
        assert torch.allclose(attn2, attn3, atol=1e-6)

    def test_mask_bad_dims_raise(self):
        """Masks with unsupported dims should raise errors (dim=1, dim=4)."""
        batch_size, seq_len, input_dim = 2, 10, 512
        model_dim, num_heads = 256, 1

        mha = MultiHeadAttention(input_dim, model_dim, num_heads, use_pytorch=False)
        x = torch.randn(batch_size, seq_len, input_dim)

        bad_mask_dim1 = torch.zeros(seq_len, dtype=torch.bool)
        with pytest.raises((ValueError, RuntimeError)):
            _ = mha(x, bad_mask_dim1)

        bad_mask_dim4 = torch.zeros(batch_size, 2, seq_len, seq_len, dtype=torch.bool)
        with pytest.raises((ValueError, RuntimeError)):
            _ = mha(x, bad_mask_dim4)

    @pytest.mark.parametrize("use_pytorch", [False, True])
    def test_dimension_mismatch_error(self, use_pytorch):
        """Test that forward pass raises error for dimension mismatch (manual & PyTorch)."""
        mha = MultiHeadAttention(512, 256, 8, use_pytorch=use_pytorch)
        x = torch.randn(2, 10, 256)  # Wrong input dimension

        with pytest.raises(ValueError):
            mha(x)

    @pytest.mark.parametrize("use_pytorch", [False, True])
    def test_weight_initialization(self, use_pytorch):
        """Test weight initialization (manual & PyTorch)."""
        mha = MultiHeadAttention(512, 256, 8, use_pytorch=use_pytorch)
        mha._init_weights()

        # Check that biases are initialized to zero
        assert torch.allclose(mha.qkv_linear.bias, torch.zeros_like(mha.qkv_linear.bias))
        assert torch.allclose(mha.out_linear.bias, torch.zeros_like(mha.out_linear.bias))


class TestTransformerFeedForward:
    """Test suite for transformer feed-forward layer."""

    def test_initialization(self):
        """Test proper initialization of feed-forward layer."""
        d_model, d_ff = 512, 2048
        ff = TransformerFeedForward(d_model, d_ff)

        assert ff.linear1.in_features == d_model
        assert ff.linear1.out_features == d_ff
        assert ff.linear2.in_features == d_ff
        assert ff.linear2.out_features == d_model

    def test_forward_pass(self):
        """Test forward pass through feed-forward layer."""
        batch_size, seq_len, d_model = 2, 10, 512
        d_ff = 2048

        ff = TransformerFeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ff(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)

        # Check that ReLU activation is applied (no negative values after first linear layer)
        with torch.no_grad():
            intermediate = ff.linear1(x)
            assert torch.all(torch.relu(intermediate) >= 0)

        # Check for NaN values
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_weight_initialization(self):
        """Test weight initialization."""
        ff = TransformerFeedForward(512, 2048)
        ff._init_weights()

        # Check that biases are initialized to zero
        assert torch.allclose(ff.linear1.bias, torch.zeros_like(ff.linear1.bias))
        assert torch.allclose(ff.linear2.bias, torch.zeros_like(ff.linear2.bias))


class TestEncoderLayer:
    """Test suite for transformer encoder layer."""

    def test_initialization(self):
        """Test proper initialization of encoder layer."""
        input_dim, model_dim, num_heads = 512, 256, 8
        encoder = EncoderLayer(input_dim, model_dim, num_heads, use_pytorch=False)

        assert isinstance(encoder.self_attn, MultiHeadAttention)
        assert isinstance(encoder.feed_forward, TransformerFeedForward)
        assert isinstance(encoder.layer_norm1, torch.nn.LayerNorm)
        assert isinstance(encoder.layer_norm2, torch.nn.LayerNorm)

    def test_forward_pass(self):
        """Test forward pass through encoder layer."""
        batch_size, seq_len, input_dim = 2, 10, 512
        model_dim, num_heads = 256, 8

        encoder = EncoderLayer(input_dim, model_dim, num_heads, use_pytorch=False)
        x = torch.randn(batch_size, seq_len, input_dim)

        output = encoder(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, input_dim)

        # Check for NaN values
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_forward_with_mask(self):
        """Test forward pass with mask."""
        batch_size, seq_len, input_dim = 2, 10, 512
        model_dim, num_heads = 256, 8

        encoder = EncoderLayer(input_dim, model_dim, num_heads, use_pytorch=False)
        x = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, -2:] = True

        output = encoder(x, mask)

        # Check output shape
        assert output.shape == (batch_size, seq_len, input_dim)

        # Check for NaN values
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_residual_connections(self):
        """Test that residual connections work properly."""
        batch_size, seq_len, input_dim = 2, 10, 512
        model_dim, num_heads = 256, 8

        encoder = EncoderLayer(input_dim, model_dim, num_heads, dropout=0.0, use_pytorch=False)
        x = torch.randn(batch_size, seq_len, input_dim)

        # Get intermediate outputs to verify residual connections
        attn_output, _ = encoder.self_attn(x)
        x_after_attn = x + attn_output
        x_after_norm1 = encoder.layer_norm1(x_after_attn)

        ff_output = encoder.feed_forward(x_after_norm1)
        x_after_ff = x_after_norm1 + ff_output
        expected_output = encoder.layer_norm2(x_after_ff)

        actual_output = encoder(x)

        # The outputs should be close (may not be exactly equal due to numerical precision)
        assert torch.allclose(actual_output, expected_output, atol=1e-6)


    def test_dropout_tuple_applied_to_submodules(self):
        input_dim, model_dim, num_heads = 32, 16, 4
        encoder = EncoderLayer(input_dim, model_dim, num_heads, dropout=(0.1, 0.2), use_pytorch=False)

        assert pytest.approx(encoder.self_attn.dropout.p, rel=0, abs=1e-6) == 0.1
        assert pytest.approx(encoder.feed_forward.dropout.p, rel=0, abs=1e-6) == 0.2

    def test_bad_dropout_tuple_length_raises(self):
        input_dim, model_dim, num_heads = 32, 16, 4
        with pytest.raises(ValueError):
            EncoderLayer(input_dim, model_dim, num_heads, dropout=(0.1, 0.2, 0.3), use_pytorch=False)

class TestTransformerEncoder:
    """Test suite for transformer encoder."""

    def test_initialization(self):
        """Test proper initialization of transformer encoder."""
        n_layers, input_dim, model_dim, num_heads = 6, 512, 256, 8
        encoder = TransformerEncoder(n_layers, input_dim, model_dim, num_heads, use_pytorch=False)

        assert len(encoder.layers) == n_layers
        assert len(encoder.layers) == n_layers
        assert all(isinstance(layer, EncoderLayer) for layer in encoder.layers)

    def test_initialization_errors(self):
        """Test that initialization raises errors for invalid parameters."""
        with pytest.raises(ValueError):
            TransformerEncoder(0, 512, 256, 8, use_pytorch=False)  # n_layers <= 0

    def test_forward_pass(self):
        """Test forward pass through transformer encoder."""
        batch_size, seq_len, input_dim = 2, 10, 512
        n_layers, model_dim, num_heads = 6, 256, 8

        encoder = TransformerEncoder(n_layers, input_dim, model_dim, num_heads, use_pytorch=False)
        x = torch.randn(batch_size, seq_len, input_dim)

        output = encoder(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, input_dim)

        # Check for NaN values
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_forward_with_mask(self):
        """Test forward pass with mask."""
        batch_size, seq_len, input_dim = 2, 10, 512
        n_layers, model_dim, num_heads = 6, 256, 8

        encoder = TransformerEncoder(n_layers, input_dim, model_dim, num_heads, use_pytorch=False)
        x = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, -2:] = True

        output = encoder(x, mask)

        # Check output shape
        assert output.shape == (batch_size, seq_len, input_dim)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the encoder."""
        batch_size, seq_len, input_dim = 2, 10, 512
        n_layers, model_dim, num_heads = 2, 256, 8

        encoder = TransformerEncoder(n_layers, input_dim, model_dim, num_heads, use_pytorch=False)
        x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

        # Check that all parameters have gradients
        for param in encoder.parameters():
            assert param.grad is not None

    def test_per_layer_num_heads_and_dropout_and_use_pytorch(self):
        n_layers = 2
        input_dim, model_dim = 32, 16
        num_heads = [2, 4]
        dropout = [0.1, (0.2, 0.3)]
        use_pytorch = [False, True]

        encoder = TransformerEncoder(
            n_layers,
            input_dim,
            model_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_pytorch=use_pytorch,
        )

        # Layer 0
        layer0 = encoder.layers[0]
        assert layer0.self_attn.num_heads == 2
        assert pytest.approx(layer0.self_attn.dropout.p, rel=0, abs=1e-6) == 0.1
        assert pytest.approx(layer0.feed_forward.dropout.p, rel=0, abs=1e-6) == 0.1
        assert layer0.self_attn.use_pytorch is False

        # Layer 1
        layer1 = encoder.layers[1]
        assert layer1.self_attn.num_heads == 4
        assert pytest.approx(layer1.self_attn.dropout.p, rel=0, abs=1e-6) == 0.2
        assert pytest.approx(layer1.feed_forward.dropout.p, rel=0, abs=1e-6) == 0.3
        assert layer1.self_attn.use_pytorch is True

    def test_num_heads_list_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            TransformerEncoder(2, 32, 16, num_heads=[2], dropout=0.1, use_pytorch=False)

    def test_dropout_list_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            TransformerEncoder(2, 32, 16, num_heads=2, dropout=[0.1], use_pytorch=False)

    def test_use_pytorch_list_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            TransformerEncoder(2, 32, 16, num_heads=2, dropout=0.1, use_pytorch=[True])


class TestIntegration:
    """Integration tests for the complete transformer."""

    def test_reproducibility(self):
        """Test that transformer outputs are reproducible with same seed."""
        torch.manual_seed(42)
        input_dim, model_dim, num_heads = 64, 32, 4
        encoder1 = TransformerEncoder(2, input_dim, model_dim, num_heads, dropout=0.0, use_pytorch=False)
        x = torch.randn(1, 5, input_dim)
        output1 = encoder1(x)

        torch.manual_seed(42)
        encoder2 = TransformerEncoder(2, input_dim, model_dim, num_heads, dropout=0.0, use_pytorch=False)
        output2 = encoder2(x)

        assert torch.allclose(output1, output2, atol=1e-6)

        # Check for NaN values
        assert not torch.isnan(output1).any(), "Output1 contains NaN values"
        assert not torch.isnan(output2).any(), "Output2 contains NaN values"

    def test_different_sequence_lengths(self):
        """Test transformer with different sequence lengths."""
        input_dim, model_dim, num_heads = 64, 32, 4
        encoder = TransformerEncoder(2, input_dim, model_dim, num_heads, use_pytorch=False)

        for seq_len in [1, 5, 10, 20]:
            x = torch.randn(2, seq_len, input_dim)
            output = encoder(x)
            assert output.shape == (2, seq_len, input_dim)

            # Check for NaN values
            assert not torch.isnan(
                output
            ).any(), f"Output contains NaN values for seq_len={seq_len}"

    def test_memory_efficiency(self):
        """Test that transformer doesn't have memory leaks during training."""
        input_dim, model_dim, num_heads = 64, 32, 4
        encoder = TransformerEncoder(2, input_dim, model_dim, num_heads, use_pytorch=False)
        optimizer = torch.optim.Adam(encoder.parameters())

        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        for _ in range(5):
            x = torch.randn(2, 10, input_dim)
            output = encoder(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Memory usage shouldn't grow significantly (allowing for some variance)
        if torch.cuda.is_available():
            memory_growth = final_memory - initial_memory
            assert memory_growth < 1024 * 1024  # Less than 1MB growth

    def test_numerical_stability(self):
        """Test that transformer doesn't produce NaN values under various conditions."""
        input_dim, model_dim, num_heads = 64, 32, 4
        encoder = TransformerEncoder(2, input_dim, model_dim, num_heads, use_pytorch=False)

        # Test with normal input
        x_normal = torch.randn(2, 10, input_dim)
        output_normal = encoder(x_normal)
        assert not torch.isnan(output_normal).any(), "Output contains NaN with normal input"
        assert not torch.isinf(output_normal).any(), "Output contains Inf with normal input"

        # Test with large values
        x_large = torch.randn(2, 10, input_dim) * 10
        output_large = encoder(x_large)
        assert not torch.isnan(output_large).any(), "Output contains NaN with large input values"
        assert not torch.isinf(output_large).any(), "Output contains Inf with large input values"

        # Test with small values
        x_small = torch.randn(2, 10, input_dim) * 0.001
        output_small = encoder(x_small)
        assert not torch.isnan(output_small).any(), "Output contains NaN with small input values"
        assert not torch.isinf(output_small).any(), "Output contains Inf with small input values"

        # Test with zero input
        x_zero = torch.zeros(2, 10, input_dim)
        output_zero = encoder(x_zero)
        assert not torch.isnan(output_zero).any(), "Output contains NaN with zero input"
        assert not torch.isinf(output_zero).any(), "Output contains Inf with zero input"

        # Test gradient computation doesn't produce NaN
        x_grad = torch.randn(2, 10, input_dim, requires_grad=True)
        output_grad = encoder(x_grad)
        loss = output_grad.sum()
        loss.backward()

        assert not torch.isnan(x_grad.grad).any(), "Input gradients contain NaN"
        for name, param in encoder.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"Parameter {name} gradients contain NaN"

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
