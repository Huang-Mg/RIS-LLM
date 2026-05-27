"""RIS-LLM: Reasoning-Informed Semantic Large Language Model.

Main model class assembling RFE, TSS, and the frozen LLM backbone
for electricity price forecasting.

"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, LlamaConfig, LlamaModel, LlamaTokenizer,
)
import transformers
transformers.logging.set_verbosity_error()

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

from layers.StandardNorm import Normalize
from layers.Embed import PatchEmbedding
from .rfe import RFEModule
from .tss import TSSModule
from .framework import compute_patch_stats
from .prototype import PrototypeBank, PrototypeCrossAttention
from .prompt import build_structured_prompt, load_data_description
from .granger import compute_granger_matrix


class RISLLMModel(nn.Module):
    """RIS-LLM forecasting model.

    Args:
        configs: configuration object with all hyperparameters
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # RFE output feature dimension: trend [D] + morphology [4*D] = 5*D
        self.d_feat = 5 * configs.enc_in

        # ---- Load frozen LLM ----
        self._load_llm(configs)

        # ---- RIS-LLM submodules ----
        self.normalize = Normalize(configs.enc_in, affine=False)

        self.rfe = RFEModule(
            h0=getattr(configs, 'rfe_h0', 0.3),
            k=getattr(configs, 'rfe_k', 2.0),
            vol_window=getattr(configs, 'rfe_window', 10),
            min_distance=getattr(configs, 'rfe_min_distance', 3),
        )

        self.tss = TSSModule(
            d_feat=self.d_feat,
            n_heads=configs.n_heads,
            dropout=configs.dropout,
            n_clusters=getattr(configs, 'n_clusters', 5),
        )

        self.patch_embed = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout
        )

        n_prototypes = getattr(configs, 'n_prototypes', 1000)
        self.prototype_bank = PrototypeBank(
            n_prototypes=n_prototypes, d_model=configs.d_model
        )

        self.proto_cross_attn = PrototypeCrossAttention(
            configs.d_model, configs.n_heads, configs.dropout
        )

        # Number of patches (overlapping if stride < patch_len)
        self.patch_nums = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        # Output projection: LLM hidden -> prediction values
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.output_projection = nn.Linear(
                self.d_llm, self.pred_len * configs.c_out
            )
        else:
            raise NotImplementedError(
                f"Task {self.task_name} not supported"
            )

        self.dropout = nn.Dropout(configs.dropout)

        # Data description for prompt
        if getattr(configs, 'prompt_domain', False):
            self.data_desc = configs.content
        else:
            self.data_desc = load_data_description(configs.data)

        print(f"RIS-LLM initialized | d_feat={self.d_feat} | "
              f"patch_nums={self.patch_nums} | d_llm={self.d_llm}")

    def _load_llm(self, configs):
        """Load frozen LLaMA backbone."""
        quant_config = None
        if getattr(configs, 'load_in_4bit', False) and BitsAndBytesConfig is not None:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        llama_config = LlamaConfig.from_pretrained('./llama')
        llama_config.num_hidden_layers = configs.llm_layers
        llama_config.output_attentions = True
        llama_config.output_hidden_states = True
        try:
            self.llm_model = LlamaModel.from_pretrained(
                './llama', trust_remote_code=True, local_files_only=True,
                config=llama_config, quantization_config=quant_config,
                torch_dtype=torch.bfloat16, device_map='auto',
                low_cpu_mem_usage=True,
            )
        except (EnvironmentError, OSError):
            print("Local LLAMA files not found. Downloading from HF...")
            self.llm_model = LlamaModel.from_pretrained(
                'huggyllama/llama-7b', trust_remote_code=True,
                local_files_only=False, config=llama_config,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16, device_map='auto',
            )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                './llama', trust_remote_code=True, local_files_only=True,
                use_fast=True,
            )
        except (EnvironmentError, OSError):
            self.tokenizer = LlamaTokenizer.from_pretrained(
                'huggyllama/llama-7b', trust_remote_code=True,
                local_files_only=False,
            )

        # Freeze LLM
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # Set pad token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

    def precompute_granger(self, train_loader, device, max_samples=200):
        """Precompute Granger causal matrix from training data.

        Args:
            train_loader: training DataLoader
            device: torch device
            max_samples: max number of batches to use (for efficiency)
        """
        self.eval()
        all_feats = []
        with torch.no_grad():
            for i, (batch_x, _, batch_x_mark, _) in enumerate(train_loader):
                if i >= max_samples:
                    break
                batch_x = batch_x.to(dtype=torch.bfloat16, device=device)
                x_norm = self.normalize(batch_x, 'norm')
                rfe_out = self.rfe(x_norm)
                all_feats.append(rfe_out['m_feat'].cpu())

        if not all_feats:
            return

        m_feat_all = torch.cat([f.reshape(-1, f.shape[-1]) for f in all_feats], dim=0)
        # For efficiency, sample at most 5000 timesteps
        if m_feat_all.shape[0] > 5000:
            idx = torch.randperm(m_feat_all.shape[0])[:5000]
            m_feat_all = m_feat_all[idx]

        C = compute_granger_matrix(
            m_feat_all,
            max_lag=getattr(self, '_granger_lag', 5),
            gamma=getattr(self, '_granger_gamma', 0.05),
        )
        self.tss.set_causal_matrix(C.to(device))
        print(f"Granger causal matrix computed: shape={C.shape}, "
              f"density={C.sum().item() / C.numel():.2%}")

    def precompute_clusters(self, train_loader, device, max_samples=100):
        """Precompute K-Means clusters from training features.

        Context vectors are extracted specifically at IDP positions where
        the price variable (last channel, index -1) exhibits significant
        fluctuation.

        Args:
            train_loader: training DataLoader
            device: torch device
            max_samples: max batches to use
        """
        self.eval()
        all_attn = []
        with torch.no_grad():
            for i, (batch_x, _, batch_x_mark, _) in enumerate(train_loader):
                if i >= max_samples:
                    break
                batch_x = batch_x.to(dtype=torch.bfloat16, device=device)
                x_norm = self.normalize(batch_x, 'norm')
                rfe_out = self.rfe(x_norm)
                m_feat = rfe_out['m_feat']
                m_attn = self.tss(m_feat)  # [B, L, D'] after causal attention

                # Extract context vectors at IDP positions of the price channel
                # (last variable = current real-time price, index -1)
                idp_mask = rfe_out['idp_mask']  # [B, L, D], D = enc_in
                price_idp = idp_mask[:, :, -1]   # [B, L] bool, IDPs on price channel

                for b in range(m_attn.shape[0]):
                    idx = torch.where(price_idp[b])[0]
                    if len(idx) > 0:
                        # Gather M_attn vectors at price fluctuation positions
                        ctx = m_attn[b, idx, :]  # [num_idps, D']
                        all_attn.append(ctx.cpu())

        if not all_attn:
            return

        m_attn_pool = torch.cat(all_attn, dim=0)  # [total_idps, D']
        # For efficiency, sample at most 5000 context vectors
        if m_attn_pool.shape[0] > 5000:
            idx = torch.randperm(m_attn_pool.shape[0])[:5000]
            m_attn_pool = m_attn_pool[idx]

        self.tss.fit_clusters(m_attn_pool)
        print(f"TSS clusters fitted from {m_attn_pool.shape[0]} price-fluctuation "
              f"context vectors | n_clusters={self.tss.n_clusters}, "
              f"labels={self.tss.label_map}")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass.

        Args:
            x_enc: [B, L, D] input time series
            x_mark_enc: [B, L, T] time features
            x_dec: [B, H+L_label, D] decoder input
            x_mark_dec: [B, H+L_label, T] decoder time marks

        Returns:
            dec_out: [B, H, D_out] predicted future values
        """
        if self.task_name not in ('long_term_forecast', 'short_term_forecast'):
            return None

        # Convert to bf16
        x_enc = x_enc.to(dtype=torch.bfloat16)

        # 1. Normalize
        x_norm = self.normalize(x_enc, 'norm')  # [B, L, D]
        B, L, D = x_norm.shape

        # 2. RFE: feature extraction
        rfe_out = self.rfe(x_norm)
        m_feat = rfe_out['m_feat']  # [B, L, 5*D]

        # 3. TSS: causal attention
        m_attn = self.tss(m_feat)  # [B, L, 5*D]

        # 4. Patching (overlapping if stride < patch_len)
        x_patch = x_norm.permute(0, 2, 1)  # [B, D, L]
        patch_emb, n_vars = self.patch_embed(x_patch)  # [B*D, N, d_model]
        N = patch_emb.shape[1]
        patch_emb = patch_emb.reshape(B, D * N, -1)  # [B, D*N, d_model]

        # Flatten batch and variable dims for prototype selection
        patch_flat = patch_emb.reshape(B * D * N, -1)  # [B*D*N, d_model]

        proto_emb, proto_idx = self.prototype_bank(
            patch_flat, top_k=getattr(self, '_proto_top_k', 5)
        )

        # Cross-attention fusion
        fused = self.proto_cross_attn(patch_flat, proto_emb)  # [B*D*N, d_model]
        fused = fused.reshape(B, D * N, -1)  # [B, D*N, d_model]

        # 5. Patch statistics for prompt
        patch_stats = compute_patch_stats(
            x_norm.float(), self.patch_len, self.stride)  # [B, N, 4*D]

        # 6. Semantic labels for patches
        patch_boundaries = [(i * self.stride, i * self.stride + self.patch_len)
                            for i in range(N)]
        patch_labels = self.tss.assign_patch_labels(m_feat, patch_boundaries)

        # 7. Build structured prompt
        task_inst = (
            f"<|TaskInst|> Predict the next {self.pred_len} steps "
            f"given the previous {self.seq_len} steps of electricity price data. "
            f"Focus on capturing volatility patterns and structural trends."
        )
        prompts = build_structured_prompt(
            self.data_desc, '', patch_stats, patch_labels,
            self.seq_len, self.pred_len,
        )

        # Tokenize
        prompt_ids = self.tokenizer(
            prompts, return_tensors='pt', padding=True,
            truncation=True, max_length=2048,
        ).input_ids.to(x_enc.device)

        # Get prompt embeddings from LLM
        embed_layer = self.llm_model.get_input_embeddings()
        prompt_embeddings = embed_layer(prompt_ids)  # [B, prompt_len, d_llm]

        # 8. Combine prompt + fused patch embeddings -> LLM
        llm_input = torch.cat([prompt_embeddings, fused], dim=1)

        dec_out = self.llm_model(
            inputs_embeds=llm_input
        ).last_hidden_state  # [B, prompt_len + D*N, d_llm]

        # 9. Output projection
        dec_out = dec_out[:, -self.pred_len:, :self.d_ff]  # [B, H, d_ff]
        y_hat = self.output_projection(dec_out)  # [B, H, pred_len * c_out or d_llm]
        # Reshape depending on output projection
        if y_hat.shape[-1] != getattr(self, 'c_out_final', D):
            y_hat = y_hat.reshape(B, self.pred_len, -1)
            y_hat = y_hat[:, :, :D]  # Take first D channels

        # 10. De-normalize
        y_hat = self.normalize(y_hat.float(), 'denorm')

        return y_hat[:, -self.pred_len:, :]
