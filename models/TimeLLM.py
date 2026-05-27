from math import sqrt
import torch
import torch.nn as nn

from transformers import AutoTokenizer, LlamaConfig, LlamaModel, LlamaTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        self.llama_config = LlamaConfig.from_pretrained('./llama')
        self.llama_config.num_hidden_layers = configs.llm_layers
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True

        quant_config = None
        if getattr(configs, 'load_in_4bit', False) and BitsAndBytesConfig is not None:
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

        try:
            self.llm_model = LlamaModel.from_pretrained(
                './llama',
                trust_remote_code=True,
                local_files_only=True,
                config=self.llama_config,
                low_cpu_mem_usage=True,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        except EnvironmentError:
            print("Local model files not found. Attempting to download...")
            self.llm_model = LlamaModel.from_pretrained(
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=False,
                config=self.llama_config,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                './llama',
                trust_remote_code=True,
                local_files_only=True,
                use_fast=True,
            )
        except EnvironmentError:
            print("Local tokenizer files not found. Atempting to download them..")
            self.tokenizer = LlamaTokenizer.from_pretrained(
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=False,
            )

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        # 自定义层放到 GPU0（首设备）
        # self.primary_device = torch.device('cuda:0')
        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        print(f"LLM 模型占用显存: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"LLM 模型参数 dtype: {next(self.llm_model.parameters()).dtype}")

        # self.output_projection = self.output_projection.to(self.primary_device)
        # self.normalize_layers = self.normalize_layers.to(self.primary_device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            x_enc = x_enc.float()
            x_mark_enc = x_mark_enc.float()
            x_dec = x_dec.float()
            x_mark_dec = x_mark_dec.float()
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc = x_enc.to(dtype=torch.bfloat16)
        # x_mark_enc = x_mark_enc.to(dtype=torch.bfloat16)
        # x_dec = x_dec.to(dtype=torch.bfloat16)
        # x_mark_dec = x_mark_dec.to(dtype=torch.bfloat16)

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()

        original_vars = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        # dt = x_enc.dtype
        # original_dtype = original_vars.dtype
        # print(original_dtype)
        # 将输入数据转换为 Float 类型
        # original_vars = original_vars.to(torch.float32)

        min_values = torch.min(original_vars, dim=1)[0]
        max_values = torch.max(original_vars, dim=1)[0]
        medians = torch.median(original_vars.float(), dim=1).values#.to(dt)
        lags = self.calcute_lags(original_vars.float())
        trends = original_vars.diff(dim=1).sum(dim=1)
        # 将结果转换回原始数据类型
        # min_values = min_values.to(original_dtype)
        # max_values = max_values.to(original_dtype)
        # medians = medians.to(original_dtype)
        # trends = trends.to(original_dtype)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids

        embed_layer = self.llm_model.get_input_embeddings()
        # embed_device = embed_layer.weight.device
        # print("embed_device:", embed_device, "prompt_ids.device:", prompt.device)
        prompt_embeddings = embed_layer(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)#.to(dt)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)#.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings.float(), source_embeddings.float())
        enc_out = enc_out.to(prompt_embeddings.dtype)

        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # 前向会跨设备执行；输出最后一层所在 GPU（这里是 cuda:1）
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out.float()
        dec_out = dec_out[:, :, :self.d_ff]

        # 把结果移回 primary_device 做后处理
        # dec_out = dec_out.to(self.primary_device)
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
