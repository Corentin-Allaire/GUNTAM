import math
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def load_state_dict_flex(model: torch.nn.Module, raw_state_dict: Dict[str, Any], desc: str = "model") -> None:
    """
    Attempt to load a raw state_dict handling compile / DP prefixes.

    Order of operations:
      1. Normalize key prefixes.
      2. Try strict load.
      3. If strict load fails, fall back to non-strict with a warning listing missing/unexpected keys.

    Args:
        model: The model to load the state_dict into.
        raw_state_dict: The raw state_dict to load.
        desc: Description of the model for logging purposes.
    """
    normalized = _normalize_state_dict_keys(raw_state_dict)
    try:
        model.load_state_dict(normalized, strict=True)
        print(f"Loaded {desc} state_dict (strict)")
    except RuntimeError as e:
        print(f"Warning: strict load failed for {desc}: {e}. Retrying non-strict...")
        missing_unexpected = model.load_state_dict(normalized, strict=False)
        # missing_unexpected is a namedtuple (missing_keys, unexpected_keys)
        print(
            f"Non-strict load results for {desc}: missing={missing_unexpected.missing_keys}, "
            f"unexpected={missing_unexpected.unexpected_keys}"
        )


def _normalize_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip torch.compile ('_orig_mod.') and optional DataParallel ('module.') prefixes.

    We iteratively strip known prefixes so that models saved from compiled or DP-wrapped
    modules can load into a plain, uncompiled instance. Saving code is left untouched
    per user request; we only normalize on load.

    Args:
        state_dict: The original state_dict with potential prefixes.
    Returns:
        A new state_dict with prefixes stripped.
    """
    prefixes = ["_orig_mod.", "module."]
    changed = True
    # Iterate until no further prefix stripping occurs (handles nested cases).
    while changed:
        changed = False
        for p in prefixes:
            if any(k.startswith(p) for k in state_dict.keys()):
                state_dict = {(k[len(p) :] if k.startswith(p) else k): v for k, v in state_dict.items()}
                changed = True
    return state_dict


def manual_scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """Scaled dot-product attention mechanism.

    Args:
          q: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
          k: Key tensor of shape (batch_size, num_heads, seq_len, d_k)
          v: Value tensor of shape (batch_size, num_heads, seq_len, d_v)
          mask: Optional mask tensor of shape (batch_size, num_heads, seq_len, seq_len)

    Returns:
        Tuple of (output tensor, attention weights matrix on the last attention layer)
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1))
    scale = 1.0 / math.sqrt(d_k)
    scores = scores * scale

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    score = F.softmax(scores, dim=-1)
    output = torch.matmul(score, v)
    return output, scores


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer.

    Args:
        input_dim: Input dimension
        model_dim: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        dropout: Dropout rate
        device: Device for computation (cpu or cuda)
        use_pytorch: Use PyTorch's flash attention (no attention weights returned)
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        device: torch.device = torch.device("cpu"),
        use_pytorch: bool = True,
    ) -> None:
        super(MultiHeadAttention, self).__init__()

        if model_dim <= 0:
            raise ValueError("Model dimension must be greater than 0")
        if num_heads <= 0:
            raise ValueError("Number of heads must be greater than 0")
        if input_dim <= 0:
            raise ValueError("Input dimension must be greater than 0")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("Dropout must be between 0 and 1")

        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.use_pytorch = use_pytorch

        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.head_dim = model_dim // num_heads
        self.qkv_linear = nn.Linear(self.input_dim, self.model_dim * 3, device=device)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(self.model_dim, self.input_dim, device=device)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor | None]:
        """Forward pass through multi-head attention layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)

        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, seq_len, d_model)
                - Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
                  or None if use_pytorch=True
        """

        batch_size = x.size(0)

        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Input tensor dimension must be the same as the transformer dim: {self.input_dim}, got {x.size(-1)}"
            )

        if mask is not None:
            if mask.dim() == 2:
                key_mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                key_mask = mask.unsqueeze(1)
                if torch.any(torch.all(key_mask, dim=-1)):
                    raise ValueError("All positions are masked for at least one query position.")
            else:
                raise ValueError(
                    "Mask tensor must have 2 or 3 dimensions (batch_size, seq_len) or (batch_size, seq_len, seq_len)"
                )
        else:
            key_mask = None

        if self.use_pytorch:
            # Use multi-head splitting for flash attention
            qkv = self.qkv_linear(x)
            qkv = qkv.view(
                batch_size, -1, self.num_heads, 3 * self.head_dim
            )  # (batch_size, seq_len, num_heads, 3 * head_dim)
            q, k, v = qkv.transpose(1, 2).chunk(3, dim=-1)  # each: (batch_size, num_heads, seq_len, head_dim)

            # Invert mask for PyTorch flash attention (True = keep, False = mask)
            key_mask = ~key_mask if key_mask is not None else None

            #  Use torch's flash attention (no attn weights returned!)
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=key_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            attn_weights = None  # no weights available

            # Concatenate heads: transpose back and reshape to combine heads
            output = (
                attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)
            )  # (batch_size, seq_len, model_dim)
        else:
            # Use manual multi-head attention with same q/k/v shapes as PyTorch path
            qkv = self.qkv_linear(x)
            # (batch_size, seq_len, 3 * model_dim)
            qkv = qkv.view(batch_size, -1, self.num_heads, 3 * self.head_dim)
            # (batch_size, seq_len, num_heads, 3 * head_dim)
            q, k, v = qkv.transpose(1, 2).chunk(3, dim=-1)

            attn_output, attn_weights = manual_scaled_dot_product_attention(q, k, v, key_mask)

            # If multiple heads, extracting the attention weights doesn't make sense
            if self.num_heads > 1:
                attn_weights = None

            # Concatenate heads back: (batch_size, seq_len, model_dim)
            output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)

        output = self.out_linear(output)
        output = self.dropout(output)
        return output, attn_weights

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.qkv_linear.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)
        self.qkv_linear.bias.data.fill_(0.0)
        self.out_linear.bias.data.fill_(0.0)


class TransformerFeedForward(nn.Module):
    """Feed-forward layer for transformer.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward layer dimension
        dropout: Dropout rate
        device: Device for computation (cpu or cuda)
    """

    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.1, device: torch.device = torch.device("cpu"),
    ) -> None:
        super(TransformerFeedForward, self).__init__()

        if d_model <= 0:
            raise ValueError("Model dimension must be greater than 0")
        if d_ff <= 0:
            raise ValueError("Feed-forward dimension must be greater than 0")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("Dropout must be between 0 and 1")

        self.linear1 = nn.Linear(d_model, d_ff, device=device)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model, device=device)
        self.gelu = nn.GELU()
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through feed-forward layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        x_ff = self.linear1(x)
        x_ff = self.gelu(x_ff)
        x_ff = self.dropout(x_ff)
        x_ff = self.linear2(x_ff)
        return x_ff

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.linear1.bias.data.fill_(0.0)
        self.linear2.bias.data.fill_(0.0)


class EncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and feed-forward.

    Args:
        input_dim: Input feature dimension.
        model_dim: Model dimension used inside attention.
        num_heads: Number of attention heads.
        dropout: Either a single dropout rate applied to both the
            attention and feed-forward sublayers, or a tuple
            ``(attn_dropout, ff_dropout)`` specifying them separately.
        device: Device for computation ("cpu" or "cuda").
        use_pytorch: If True, use PyTorch's flash attention in
            ``MultiHeadAttention``; otherwise fall back to the
            manual implementation.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        dropout: float | Tuple[float, float] = 0.1,
        device: torch.device = torch.device("cpu"),
        use_pytorch: bool = True,
    ) -> None:
        super(EncoderLayer, self).__init__()
        # Allow either single dropout for both sublayers or a pair
        if isinstance(dropout, tuple):
            if len(dropout) != 2:
                raise ValueError("dropout tuple must have length 2 (attn, ff)")
            attn_dropout, ff_dropout = dropout
        else:
            attn_dropout = ff_dropout = dropout

        self.self_attn = MultiHeadAttention(
            input_dim,
            model_dim,
            num_heads,
            attn_dropout,
            device=device,
            use_pytorch=use_pytorch,
        )
        self.feed_forward = TransformerFeedForward(input_dim, 4 * input_dim, ff_dropout, device=device)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self._init_weights()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the encoder layer.
        The attention weights are discarded by this method.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor): Optional mask tensor of shape (batch_size, seq_len, seq_len).
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x_ff, _ = self.encode(x, mask)

        return x_ff

    def encode(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Path trough the encoder layer that returns attention weights.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor): Optional mask tensor of shape (batch_size, seq_len).
        Returns:
            Tuple[Tensor, Tensor]:
                - Output tensor after encoder layer (batch_size, seq_len, d_model).
                - Attention weights (batch_size, num_heads, seq_len, seq_len).
        """
        attn_output, attn_weights = self.self_attn(x, mask)
        x_attn = x + attn_output
        x_attn = self.layer_norm1(x_attn)

        ff_output = self.feed_forward(x_attn)
        x_ff = x_attn + ff_output
        x_ff = self.layer_norm2(x_ff)
        return x_ff, attn_weights

    def _init_weights(self) -> None:
        """
        Initialize weights of the encoder layer.
        """
        self.self_attn._init_weights()
        self.feed_forward._init_weights()
        nn.init.constant_(self.layer_norm1.bias, 0.0)
        nn.init.constant_(self.layer_norm1.weight, 1.0)
        nn.init.constant_(self.layer_norm2.bias, 0.0)
        nn.init.constant_(self.layer_norm2.weight, 1.0)


class TransformerEncoder(nn.Module):
    """A full transformer encoder consisting of multiple encoder layers.

    Args:
        n_layers: Number of encoder layers.
        input_dim: Input feature dimension for each layer.
        model_dim: Internal model dimension used in attention.
        num_heads: Number of attention heads, either a single integer
            applied to all layers or a list of length ``n_layers`` with
            per-layer head counts.
        dropout: Dropout rate configuration, either a single float
            applied to all layers, or a list of length ``n_layers``
            whose elements are either floats or ``(attn, ff)`` tuples
            passed to ``EncoderLayer``.
        use_pytorch: Whether to use PyTorch's flash attention for each
            layer. Can be a single bool applied to all layers, or a
            list of length ``n_layers`` for per-layer control.
        device: Device for computation ("cpu" or "cuda").
    """

    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        model_dim: int,
        num_heads: int | list,
        dropout: float | list = 0.1,
        use_pytorch: bool | list = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super(TransformerEncoder, self).__init__()
        if n_layers <= 0:
            raise ValueError("Number of layers must be greater than 0")

        self.layers = nn.ModuleList()

        # Normalize num_heads to a per-layer list
        if isinstance(num_heads, list):
            if len(num_heads) != n_layers:
                raise ValueError("num_heads list length must equal n_layers")
            num_heads_list = num_heads
        else:
            num_heads_list = [num_heads] * n_layers

        # Normalize dropout to a per-layer list
        if isinstance(dropout, list):
            if len(dropout) != n_layers:
                raise ValueError("dropout list length must equal n_layers")
            dropout_list = dropout
        else:
            dropout_list = [dropout] * n_layers

        # Normalize use_pytorch to a per-layer list
        if isinstance(use_pytorch, list):
            if len(use_pytorch) != n_layers:
                raise ValueError("use_pytorch list length must equal n_layers")
            use_pytorch_list = use_pytorch
        else:
            # Default behavior: flash attention for all layers
            use_pytorch_list = [use_pytorch] * n_layers

        for i in range(n_layers):
            self.layers.append(
                EncoderLayer(
                    input_dim,
                    model_dim,
                    num_heads_list[i],
                    dropout_list[i],
                    device=device,
                    use_pytorch=use_pytorch_list[i],
                )
            )
        self._init_weights()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the transformer encoder layer.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor): Optional mask tensor of shape (batch_size, seq_len).
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x_layer = x
        for layer in self.layers:
            x_layer = layer(x_layer, mask)

        return x_layer

    def _init_weights(self) -> None:
        """
        Initialize weights of the transformer encoder.
        """
        for layer in self.layers:
            layer._init_weights()

    def get_layers(self) -> nn.ModuleList:
        """Return the list of encoder layers.

        This exposes the internal ``EncoderLayer`` modules so callers can
        inspect per-layer configuration, weights, or register hooks.

        Returns:
            nn.ModuleList: The underlying list of encoder layers.
        """
        return self.layers
