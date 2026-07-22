import time
import datetime
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, recall_score
import sys
import os
import warnings

sys.path.append('./')
warnings.filterwarnings("ignore")

import config
from utils import Logger, get_loaders, build_model, generate_mask, generate_inputs
from loss import MaskedCELoss, MaskedMSELoss

total_start_time = time.time()


def add_gaussian_noise(feat, sigma=0.1, seed=None):
    """Add Gaussian noise to a feature tensor for robustness testing."""
    if seed is not None:
        torch.manual_seed(seed)
    return feat + torch.randn_like(feat) * sigma


def train_or_eval_ebmc(args, model, reg_loss, cls_loss, dataloader,
                       optimizer=None, train=False, first_stage=True):
    """Run one epoch of training or evaluation.

    Returns:
        (mae, corr_or_ua, acc, f1, [acc_a, acc_t, acc_v])
    """
    dataset = args.dataset
    preds, preds_a, preds_t, preds_v = [], [], [], []
    loss_accumulator = []
    labels_all, masks_all = [], []

    device = args.device
    cuda = torch.cuda.is_available() and not args.no_cuda

    if train:
        model.train()
    else:
        model.eval()

    assert not train or optimizer is not None

    for data_idx, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()

        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]

        if args.add_noise:
            audio_host  = add_gaussian_noise(audio_host,  args.guassian_sigma, seed=args.seed)
            text_host   = add_gaussian_noise(text_host,   args.guassian_sigma, seed=args.seed)
            visual_host = add_gaussian_noise(visual_host, args.guassian_sigma, seed=args.seed)
            audio_guest  = add_gaussian_noise(audio_guest,  args.guassian_sigma, seed=args.seed)
            text_guest   = add_gaussian_noise(text_guest,   args.guassian_sigma, seed=args.seed)
            visual_guest = add_gaussian_noise(visual_guest, args.guassian_sigma, seed=args.seed)

        qmask, umask, label = data[6], data[7], data[8]

        seqlen = audio_host.size(0)
        batch  = audio_host.size(1)

        # Generate modality masks for host and guest
        def make_mask(cond):
            mat = generate_mask(seqlen, batch, cond, first_stage)
            a = torch.LongTensor(np.reshape(mat[0], (batch, seqlen, 1)).transpose(1, 0, 2))
            t = torch.LongTensor(np.reshape(mat[1], (batch, seqlen, 1)).transpose(1, 0, 2))
            v = torch.LongTensor(np.reshape(mat[2], (batch, seqlen, 1)).transpose(1, 0, 2))
            return a, t, v

        am_h, tm_h, vm_h = make_mask(args.test_condition)
        am_g, tm_g, vm_g = make_mask(args.test_condition)

        masked_audio_host  = audio_host  * am_h
        masked_text_host   = text_host   * tm_h
        masked_visual_host = visual_host * vm_h
        masked_audio_guest  = audio_guest  * am_g
        masked_text_guest   = text_guest   * tm_g
        masked_visual_guest = visual_guest * vm_g

        if cuda:
            masked_audio_host   = masked_audio_host.to(device)
            masked_text_host    = masked_text_host.to(device)
            masked_visual_host  = masked_visual_host.to(device)
            masked_audio_guest  = masked_audio_guest.to(device)
            masked_text_guest   = masked_text_guest.to(device)
            masked_visual_guest = masked_visual_guest.to(device)
            am_h = am_h.to(device); tm_h = tm_h.to(device); vm_h = vm_h.to(device)
            am_g = am_g.to(device); tm_g = tm_g.to(device); vm_g = vm_g.to(device)
            qmask = qmask.to(device)
            umask = umask.to(device)
            label = label.to(device)

        masked_inputs = generate_inputs(
            masked_audio_host, masked_text_host, masked_visual_host,
            masked_audio_guest, masked_text_guest, masked_visual_guest,
            qmask
        )
        mask_inputs = generate_inputs(
            am_h, tm_h, vm_h, am_g, tm_g, vm_g, qmask
        )

        _, logits_c, logits_a, logits_t, logits_v, losses, _ = model(
            masked_inputs[0], mask_inputs[0], umask, first_stage, label, data_idx
        )

        disentangle_loss = losses.get('disentangle', torch.tensor(0.0, device=device))
        cfd_loss         = losses.get('cfd',         torch.tensor(0.0, device=device))
        emc_loss         = losses.get('emc',         torch.tensor(0.0, device=device))
        imtd_loss        = losses.get('imtd',        torch.tensor(0.0, device=device))

        pred_c = logits_c.view(-1, logits_c.size(-1))
        pred_a = logits_a.view(-1, logits_a.size(-1))
        pred_t = logits_t.view(-1, logits_t.size(-1))
        pred_v = logits_v.view(-1, logits_v.size(-1))

        # Task loss
        if dataset in ['CMUMOSI', 'CMUMOSEI']:
            if first_stage:
                loss_a = reg_loss(pred_a, label, umask)
                loss_t = reg_loss(pred_t, label, umask)
                loss_v = reg_loss(pred_v, label, umask)
            else:
                loss_c = reg_loss(pred_c, label, umask)
        else:
            if first_stage:
                loss_a = cls_loss(pred_a, label, umask)
                loss_t = cls_loss(pred_t, label, umask)
                loss_v = cls_loss(pred_v, label, umask)
            else:
                loss_c = cls_loss(pred_c, label, umask)

        valid_count = umask.view(-1).sum().item()

        if train and first_stage:
            loss = (loss_a + loss_t + loss_v
                    + args.lambda_msd * disentangle_loss
                    + args.lambda_cce * cfd_loss)
            loss_accumulator.append(loss.item() * max(valid_count, 1.0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        elif train and not first_stage:
            loss = (loss_c
                    + args.lambda_msd  * disentangle_loss
                    + args.lambda_cce  * cfd_loss
                    + args.lambda_emc  * emc_loss
                    + args.lambda_imtd * imtd_loss)
            loss_accumulator.append(loss.item() * max(valid_count, 1.0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            task_loss = (loss_a + loss_t + loss_v) if first_stage else loss_c
            loss_accumulator.append(task_loss.item() * max(valid_count, 1.0))

        preds.append(pred_c.detach().cpu().numpy())
        preds_a.append(pred_a.detach().cpu().numpy())
        preds_t.append(pred_t.detach().cpu().numpy())
        preds_v.append(pred_v.detach().cpu().numpy())
        labels_all.append(label.view(-1).cpu().numpy())
        masks_all.append(umask.view(-1).cpu().numpy())

    preds     = np.concatenate(preds)
    preds_a   = np.concatenate(preds_a)
    preds_t   = np.concatenate(preds_t)
    preds_v   = np.concatenate(preds_v)
    labels_all = np.concatenate(labels_all)
    masks_all  = np.concatenate(masks_all)

    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        preds_label   = np.argmax(preds,   axis=1)
        preds_a_label = np.argmax(preds_a, axis=1)
        preds_t_label = np.argmax(preds_t, axis=1)
        preds_v_label = np.argmax(preds_v, axis=1)
        acc  = accuracy_score(labels_all, preds_label,   sample_weight=masks_all)
        f1   = f1_score(labels_all, preds_label,         sample_weight=masks_all, average='weighted')
        ua   = recall_score(labels_all, preds_label,     sample_weight=masks_all, average='macro')
        acc_a = accuracy_score(labels_all, preds_a_label, sample_weight=masks_all)
        acc_t = accuracy_score(labels_all, preds_t_label, sample_weight=masks_all)
        acc_v = accuracy_score(labels_all, preds_v_label, sample_weight=masks_all)
        return 0, ua, acc, f1, [acc_a, acc_t, acc_v]

    else:  # CMUMOSI / CMUMOSEI (regression, Non-0 binary eval)
        nz = np.array([i for i, e in enumerate(labels_all) if e != 0])
        mae  = np.mean(np.abs(labels_all[nz] - preds[nz].squeeze()))
        corr = np.corrcoef(labels_all[nz], preds[nz].squeeze())[0][1]
        acc  = accuracy_score((labels_all[nz] > 0), (preds[nz]   > 0))
        f1   = f1_score((labels_all[nz] > 0),       (preds[nz]   > 0), average='weighted')
        acc_a = accuracy_score((labels_all[nz] > 0), (preds_a[nz] > 0))
        acc_t = accuracy_score((labels_all[nz] > 0), (preds_t[nz] > 0))
        acc_v = accuracy_score((labels_all[nz] > 0), (preds_v[nz] > 0))
        # Acc-7: 7-class accuracy over ALL samples, clip to [-3,3] then round-and-compare
        # (same as LNLN core/metric.py __multiclass_acc)
        p7 = np.clip(preds.squeeze(), -3.0, 3.0)
        l7 = np.clip(labels_all, -3.0, 3.0)
        acc7 = float(np.mean(np.round(p7) == np.round(l7)))
        return mae, corr, acc, f1, [acc_a, acc_t, acc_v], acc7


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input features
    parser.add_argument('--audio-feature', type=str, required=True)
    parser.add_argument('--text-feature',  type=str, required=True)
    parser.add_argument('--video-feature', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='IEMOCAPFour')

    # Model architecture
    parser.add_argument('--depth',         type=int,   default=4)
    parser.add_argument('--num_heads',     type=int,   default=4)
    parser.add_argument('--drop_rate',     type=float, default=0.1)
    parser.add_argument('--attn_drop_rate',type=float, default=0.1)
    parser.add_argument('--hidden',        type=int,   default=128)
    parser.add_argument('--n_classes',     type=int,   default=4)
    parser.add_argument('--n_speakers',    type=int,   default=2)

    # Training parameters
    parser.add_argument('--no-cuda',       action='store_true', default=False)
    parser.add_argument('--gpu',           type=int,   default=0)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--l2',            type=float, default=1e-5)
    parser.add_argument('--batch-size',    type=int,   default=32)
    parser.add_argument('--epochs',        type=int,   default=30)
    parser.add_argument('--num-folder',    type=int,   default=5)
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--add_noise',     action='store_true', help='add Gaussian noise for robustness testing')
    parser.add_argument('--guassian_sigma',type=float, default=0.05)
    parser.add_argument('--test_condition',type=str,   default='atv')

    # Loss weights
    parser.add_argument('--stage_epoch', type=int,   default=50,
                        help='epoch to switch from Stage-I (unimodal) to Stage-II (fusion)')
    parser.add_argument('--lambda_msd',  type=float, default=0.01)
    parser.add_argument('--lambda_cce',  type=float, default=0.1)
    parser.add_argument('--lambda_emc',  type=float, default=0.1)
    parser.add_argument('--lambda_imtd', type=float, default=0.1)

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # --- Logger ---
    save_folder_name = args.dataset
    save_log = os.path.join(config.LOG_DIR, 'main_result', save_folder_name)
    os.makedirs(save_log, exist_ok=True)
    time_tag = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    sys.stdout = Logger(
        filename=f"{save_log}/{time_tag}_bs-{args.batch_size}_ep-{args.epochs}"
                 f"_seed-{args.seed}_drop-{args.drop_rate}_cond-{args.test_condition}.txt",
        stream=sys.stdout
    )

    # --- Seed ---
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Dataset configuration
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        args.num_folder, args.n_classes = 1, 1
    elif args.dataset == 'IEMOCAPFour':
        args.num_folder, args.n_classes = 5, 4
    elif args.dataset == 'IEMOCAPSix':
        args.num_folder, args.n_classes = 5, 6

    cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    print('====== Loading Data ======')
    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], args.audio_feature)
    text_root  = os.path.join(config.PATH_TO_FEATURES[args.dataset], args.text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], args.video_feature)
    train_loaders, test_loaders, adim, tdim, vdim = get_loaders(
        audio_root=audio_root, text_root=text_root, video_root=video_root,
        num_folder=args.num_folder, batch_size=args.batch_size,
        dataset=args.dataset, num_workers=0
    )

    folder_mae, folder_corr, folder_acc, folder_f1 = [], [], [], []
    folder_acc7 = []

    for fold_idx in range(args.num_folder):
        print(f"\n===== Fold {fold_idx + 1}/{args.num_folder} =====")
        train_loader = train_loaders[fold_idx]
        test_loader  = test_loaders[fold_idx]

        model = build_model(args, adim, tdim, vdim)
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        reg_loss, cls_loss = MaskedMSELoss(), MaskedCELoss()
        if cuda:
            model.to(args.device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        print(f"\n[Training] Stage-I: 0~{args.stage_epoch-1} | Stage-II: {args.stage_epoch}~{args.epochs-1}")

        best_acc = -1.0
        best_model = None
        current_stage = 1
        test_maes, test_corrs, test_accs, test_fscores, test_acc7s = [], [], [], [], []

        for epoch in range(args.epochs):
            first_stage = epoch < args.stage_epoch

            if not first_stage and current_stage == 1:
                current_stage = 2
                print(f"\n>>> [Stage Switch] Entering Stage-II at epoch {epoch} <<<")

            # Train
            train_res = train_or_eval_ebmc(
                args, model, reg_loss, cls_loss, train_loader,
                optimizer=optimizer, train=True, first_stage=first_stage
            )
            # Eval
            test_res = train_or_eval_ebmc(
                args, model, reg_loss, cls_loss, test_loader,
                optimizer=None, train=False, first_stage=first_stage
            )

            train_acc_atv = train_res[4]
            mae, corr, acc, f1 = test_res[0], test_res[1], test_res[2], test_res[3]
            acc7 = test_res[5] if len(test_res) > 5 else 0.0

            test_maes.append(mae)
            test_corrs.append(corr)
            test_accs.append(acc)
            test_fscores.append(f1)
            test_acc7s.append(acc7)

            if first_stage:
                print(f'epoch:{epoch}; a_acc:{train_acc_atv[0]:.3f}; t_acc:{train_acc_atv[1]:.3f}; v_acc:{train_acc_atv[2]:.3f}')
            else:
                print(f'epoch:{epoch}; mae:{mae:.3f}; corr:{corr:.3f}; f1:{f1:.2%}; acc:{acc:.2%}')
            print('-' * 10)

            # At end of Stage-I: cache teacher features and save checkpoint
            if epoch == args.stage_epoch - 1:
                last_outputs = getattr(model, '_last_outputs', None)
                if last_outputs is not None and 'features_teacher' in last_outputs:
                    model.teacher_outputs = last_outputs['features_teacher']
                    print(f"[Stage-I Done] Teacher features cached.")
                else:
                    print("[Warning] No teacher features in model._last_outputs.")

                stage1_dir = os.path.join(config.MODEL_DIR, 'stage_1')
                os.makedirs(stage1_dir, exist_ok=True)
                stage1_path = os.path.join(stage1_dir, f"{args.dataset}_{args.test_condition}_stage1_best.pth")
                torch.save(best_model if best_model else model.state_dict(), stage1_path)
                print(f"[Stage-I Done] Checkpoint saved: {stage1_path}")

            if acc > best_acc:
                best_acc = acc
                best_model = model.state_dict()

        # Select best epoch
        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            best_idx = int(np.argmax(test_fscores))
        else:
            best_idx = int(np.argmax(test_accs))

        folder_mae.append(test_maes[best_idx])
        folder_corr.append(test_corrs[best_idx])
        folder_acc.append(test_accs[best_idx])
        folder_f1.append(test_fscores[best_idx])
        folder_acc7.append(test_acc7s[best_idx])

        print(f'Step3: Fold {fold_idx+1} best epoch={best_idx}')
        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            print(f"  mae={test_maes[best_idx]:.4f}  corr={test_corrs[best_idx]:.4f}"
                  f"  f1={test_fscores[best_idx]:.4f}  acc={test_accs[best_idx]:.4f}"
                  f"  acc7={test_acc7s[best_idx]:.4f}")
        else:
            print(f"  acc={test_accs[best_idx]:.4f}  ua={test_corrs[best_idx]:.4f}")

    print('-' * 80)
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        print(f"Folder avg ({args.test_condition}): "
              f"mae={np.mean(folder_mae):.4f}  corr={np.mean(folder_corr):.4f}"
              f"  f1={np.mean(folder_f1):.4f}  acc={np.mean(folder_acc):.4f}"
              f"  acc7={np.mean(folder_acc7):.4f}")
    else:
        print(f"Folder avg ({args.test_condition}): "
              f"acc={np.mean(folder_acc):.4f}  ua={np.mean(folder_corr):.4f}")

    # Save final model
    print('====== Saving ======')
    save_model_dir = os.path.join(config.MODEL_DIR, 'main_result', save_folder_name)
    os.makedirs(save_model_dir, exist_ok=True)

    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        res_name = (f"mae-{np.mean(folder_mae):.3f}_corr-{np.mean(folder_corr):.3f}"
                    f"_f1-{np.mean(folder_f1):.4f}_acc-{np.mean(folder_acc):.4f}")
    else:
        res_name = f"acc-{np.mean(folder_acc):.4f}_ua-{np.mean(folder_corr):.4f}"

    save_path = os.path.join(save_model_dir,
                             f"{time_tag}_hidden-{args.hidden}_bs-{args.batch_size}"
                             f"_{res_name}_cond-{args.test_condition}.pth")
    torch.save({'model': model.state_dict()}, save_path)
    print(f"Total runtime: {time.time() - total_start_time:.2f}s")
    print(save_path)
