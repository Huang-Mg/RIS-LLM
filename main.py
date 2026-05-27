"""RIS-LLM main training and evaluation script."""

import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import numpy as np
import os
import sys

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, vali, load_content, predict
from utils.losses import smape_loss
from config import get_config_parser

# ---- Model imports ----
from models import TimeLLM
from risllm.model import RISLLMModel


def build_model(args):
    """Build and return the specified model."""
    if args.model == 'TimeLLM':
        return TimeLLM.Model(args).float()
    elif args.model == 'RISLLM':
        return RISLLMModel(args)
    else:
        raise ValueError(f"Unknown model: {args.model}. Options: TimeLLM, RISLLM")


def get_optimizer(model, args):
    """Build optimizer with optional LLM fine-tuning parameter groups."""
    if args.finetune and hasattr(model, 'llm_model'):
        llm_params = []
        other_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'llm_model' in name:
                    llm_params.append(param)
                else:
                    other_params.append(param)
        optim_groups = [
            {'params': llm_params, 'lr': args.llm_lr},
            {'params': other_params, 'lr': args.learning_rate},
        ]
        return torch.optim.Adam(optim_groups)
    else:
        trained_params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(trained_params, lr=args.learning_rate)


def get_scheduler(optimizer, args, train_steps):
    """Build learning rate scheduler."""
    if args.lradj == 'COS':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=1e-8
        )
    else:
        return lr_scheduler.OneCycleLR(
            optimizer=optimizer, steps_per_epoch=train_steps,
            pct_start=args.pct_start, epochs=args.train_epochs,
            max_lr=args.learning_rate,
        )


def train_epoch(model, train_loader, optimizer, criterion, scheduler,
                device, epoch, args, scaler=None):
    """Train one epoch."""
    model.train()
    train_loss = []
    epoch_time = time.time()
    time_now = time.time()
    iter_count = 0

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
            tqdm(train_loader, desc=f'Epoch {epoch+1}')):
        iter_count += 1
        optimizer.zero_grad()

        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        if args.use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
        else:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

        if (i + 1) % 100 == 0:
            print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((args.train_epochs - epoch) * len(train_loader) - i)
            print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
            iter_count = 0
            time_now = time.time()

        if args.use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.lradj == 'TST':
            adjust_learning_rate(optimizer, scheduler, epoch + 1, args, printout=False)
            scheduler.step()

    print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time:.2f}s")
    return np.average(train_loss)


def main():
    parser = get_config_parser()
    args = parser.parse_args()

    # Logging setup
    log_file = open('log.txt', 'w')
    original_stdout = sys.stdout
    sys.stdout = log_file

    # Run experiments
    for ii in range(args.itr):
        setting = (
            f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_'
            f'ft{args.features}_sl{args.seq_len}_ll{args.label_len}_'
            f'pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_'
            f'el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_'
            f'fc{args.factor}_eb{args.embed}_{args.des}_{ii}'
        )

        # Load data
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        args.content = load_content(args)

        # Build model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.model == 'RISLLM':
            model = RISLLMModel(args).to(device)
        else:
            model = build_model(args).to(device)

        # RISLLM precompute steps
        if args.model == 'RISLLM':
            print("Precomputing Granger causal matrix...")
            model.precompute_granger(train_loader, device)
            print("Precomputing TSS clusters...")
            model.precompute_clusters(train_loader, device)

        # Setup
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=args.patience)
        name = f'{args.model}_{args.pred_len}'

        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args, train_steps)
        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()
        smape_criterion = smape_loss()

        scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

        # Training loop
        for epoch in range(args.train_epochs):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, scheduler,
                device, epoch, args, scaler,
            )

            vali_loss, vali_mae_loss, vali_smape_loss = vali(
                args, model, vali_data, vali_loader, criterion,
                mae_metric, smape_criterion, device,
            )
            test_loss, test_mae_loss, test_smape_loss = vali(
                args, model, test_data, test_loader, criterion,
                mae_metric, smape_criterion, device,
            )
            print(
                f"Epoch: {epoch+1} | Train Loss: {train_loss:.7f} "
                f"Vali Loss: {vali_loss:.7f} "
                f"MSE: {test_loss:.7f} MAE: {test_mae_loss:.7f}"
            )

            early_stopping(vali_loss, model, name)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                    print(f"lr = {optimizer.param_groups[0]['lr']:.10f}")
                else:
                    if epoch == 0:
                        args.learning_rate = optimizer.param_groups[0]['lr']
                    adjust_learning_rate(optimizer, scheduler, epoch + 1, args, printout=True)
            else:
                print(f'Updating learning rate to {scheduler.get_last_lr()[0]}')

        # Final prediction
        sys.stdout = original_stdout
        log_file.close()

        print("-----------------Final Test and Prediction-----------------")
        model_path = os.path.join('./checkpoints/', name + '.pth')
        result_path = os.path.join(args.result_path, name)

        if os.path.exists(model_path):
            print("Loading best model for final prediction...")
            predict(args, model, test_loader, device,
                    load_model_path=model_path, save_path=result_path,
                    test_mse=test_loss, test_mae=test_mae_loss,
                    test_smape=test_smape_loss)
        else:
            print("No checkpoint found to run final prediction.")


if __name__ == '__main__':
    main()
