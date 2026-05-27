"""RIS-LLM configuration: all argument definitions in one place."""
import argparse


def get_config_parser():
    parser = argparse.ArgumentParser(description='RIS-LLM')

    # --- basic ---
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='training status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model_comment', type=str, required=True, default='none',
                        help='prefix when saving test results')
    parser.add_argument('--model', type=str, required=True, default='RISLLM',
                        help='model name: [RISLLM, TimeLLM, TSSLLM, Autoformer, DLinear, PatchTST, TimesNet, CNNLSTM, XGBOOST]')
    # --- data loader ---
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file name')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task: M, S, MS')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features: s, t, h, d, b, w, m')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='model checkpoints dir')
    parser.add_argument('--percent', type=int, default=100)

    # --- forecasting task ---
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # --- model define ---
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding: timeF, fixed, learned')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true',
                        help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8,
                        help='patch stride; =patch_len for non-overlap, <patch_len for overlap')
    parser.add_argument('--prompt_domain', type=int, default=0, help='use domain prompt')
    parser.add_argument('--llm_model', type=str, default='LLAMA',
                        help='LLM model: LLAMA, GPT2, BERT')
    parser.add_argument('--llm_dim', type=int, default=4096,
                        help='LLM model dimension: LLama7b=4096, GPT2-small=768, BERT-base=768')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--top_k', type=int, default=5, help='top k lags')

    # --- optimization ---
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true',
                        help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--finetune', action='store_true', help='finetune the LLM')
    parser.add_argument('--llm_lr', type=float, default=0.00001,
                        help='learning rate for LLM fine-tuning')
    parser.add_argument('--finetune_layers', type=int, default=0,
                        help='number of layers to finetune from top; 0 = full finetuning')
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                        help='4bit loading LLM')
    parser.add_argument('--result_path', type=str, default='./results/',
                        help='path for saving results')

    # --- RIS-LLM specific: RFE ---
    parser.add_argument('--rfe_h0', type=float, default=0.3,
                        help='base LOESS bandwidth for RFE')
    parser.add_argument('--rfe_k', type=float, default=2.0,
                        help='volatility sensitivity coefficient for RFE')
    parser.add_argument('--rfe_window', type=int, default=10,
                        help='rolling std window size for volatility computation')
    parser.add_argument('--rfe_min_distance', type=int, default=3,
                        help='minimum steps between consecutive IDPs')

    # --- RIS-LLM specific: TSS ---
    parser.add_argument('--granger_gamma', type=float, default=0.05,
                        help='Granger causality p-value threshold')
    parser.add_argument('--granger_max_lag', type=int, default=5,
                        help='maximum lag for VAR model in Granger test')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='number of K-Means clusters for semantic labeling')
    parser.add_argument('--n_prototypes', type=int, default=1000,
                        help='number of prototype embeddings')
    parser.add_argument('--proto_top_k', type=int, default=5,
                        help='top-k prototypes selected per patch')

    return parser


def get_config():
    parser = get_config_parser()
    args = parser.parse_args()
    return args
