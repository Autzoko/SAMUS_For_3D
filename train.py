from ast import arg
import os
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module

from torch.nn.modules.loss import CrossEntropyLoss
from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt


def main():

    #  ============================================================================= parameters setting ====================================================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='US30K', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')
    parser.add_argument('--data_path', type=str, default=None, help='Override data path from config')
    parser.add_argument('--shard_dir', type=str, default=None, help='WebDataset shard directory (overrides file-based loading)')
    parser.add_argument('--load_path', type=str, default=None, help='Override checkpoint load path from config')

    args = parser.parse_args()
    opt = get_config(args.task)
    if args.data_path:
        opt.data_path = args.data_path
    if args.shard_dir:
        opt.shard_dir = args.shard_dir
    if args.load_path:
        opt.load_path = args.load_path

    device = torch.device(opt.device)
    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    
    # register the sam model
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = args.batch_size * args.n_gpu

    tf_train = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)

    use_wds = args.task == "ABUS" and getattr(opt, 'shard_dir', '') and os.path.isdir(getattr(opt, 'shard_dir', ''))

    if use_wds:
        from utils.data_abus import build_abus_wds_loader
        print(f"Using WebDataset shards from: {opt.shard_dir}")
        _, trainloader = build_abus_wds_loader(
            opt.shard_dir, 'train', tf_train, img_size=args.encoder_input_size,
            batch_size=opt.batch_size, num_workers=opt.workers)
        _, valloader = build_abus_wds_loader(
            opt.shard_dir, 'val', tf_val, img_size=args.encoder_input_size,
            batch_size=opt.batch_size, num_workers=opt.workers)
    elif args.task == "ABUS":
        from utils.data_abus import ABUSDataset
        print(f"Using direct file loading from: {opt.data_path}")
        train_dataset = ABUSDataset(opt.data_path, 'train', tf_train, img_size=args.encoder_input_size)
        val_dataset = ABUSDataset(opt.data_path, 'val', tf_val, img_size=args.encoder_input_size)
        trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    else:
        train_dataset = ImageToImage2D(opt.data_path, opt.train_split, tf_train, img_size=args.encoder_input_size)
        val_dataset = ImageToImage2D(opt.data_path, opt.val_split, tf_val, img_size=args.encoder_input_size)
        trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model.to(device)
    if opt.pre_trained:
        checkpoint = torch.load(opt.load_path)
        new_state_dict = {}
        for k,v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
      
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
   
    criterion = get_criterion(modelname=args.modelname, opt=opt)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    #  ========================================================================= begin to train the model ============================================================================
    iter_num = 0
    # WebDataset-based DataLoader doesn't support len(); estimate from shard index
    try:
        steps_per_epoch = len(trainloader)
    except TypeError:
        import json as _json
        _idx_path = os.path.join(getattr(opt, 'shard_dir', ''), 'train', 'index.json')
        if os.path.isfile(_idx_path):
            with open(_idx_path) as _f:
                _n_samples = _json.load(_f).get('n_samples', 3000)
            steps_per_epoch = _n_samples // opt.batch_size
        else:
            steps_per_epoch = 3000 // opt.batch_size
    max_iterations = opt.epochs * steps_per_epoch

    # Print training config summary
    print("\n" + "=" * 60)
    print(f"  Model:            {args.modelname}")
    print(f"  Task:             {args.task}")
    print(f"  Trainable params: {pytorch_total_params:,} / {total_params:,} ({100*pytorch_total_params/total_params:.1f}%)")
    print(f"  Batch size:       {opt.batch_size}")
    print(f"  Learning rate:    {args.base_lr}")
    print(f"  Epochs:           {opt.epochs}")
    print(f"  Steps/epoch:      {steps_per_epoch}")
    print(f"  Total iterations: {max_iterations:,}")
    print(f"  Warmup:           {args.warmup} ({args.warmup_period} iters)")
    print(f"  Save path:        {opt.save_path}")
    print("=" * 60 + "\n")

    best_dice, loss_log, dice_log = 0.0, np.zeros(opt.epochs+1), np.zeros(opt.epochs+1)
    train_start = time.time()
    for epoch in range(opt.epochs):
        #  --------------------------------------------------------- training ---------------------------------------------------------
        model.train()
        train_losses = 0
        epoch_start = time.time()
        for batch_idx, (datapack) in enumerate(trainloader):
            imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
            masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
            bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)
            pt = get_click_prompt(datapack, opt)
            # -------------------------------------------------------- forward --------------------------------------------------------
            pred = model(imgs, pt, bbox)
            train_loss = criterion(pred, masks)
            # -------------------------------------------------------- backward -------------------------------------------------------
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()
            # ------------------------------------------- adjust the learning rate when needed-----------------------------------------
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            iter_num = iter_num + 1

            # Print progress every 50 steps
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                avg_loss = train_losses / (batch_idx + 1)
                print(f"  [Step {batch_idx+1:>4d}/{steps_per_epoch}] loss: {train_loss.item():.4f}  avg: {avg_loss:.4f}  lr: {cur_lr:.2e}")

        #  -------------------------------------------------- log the train progress --------------------------------------------------
        epoch_time = time.time() - epoch_start
        avg_train_loss = train_losses / (batch_idx + 1)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{opt.epochs} | train_loss: {avg_train_loss:.4f} | lr: {cur_lr:.2e} | time: {epoch_time:.0f}s")
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', avg_train_loss, epoch)
            TensorWriter.add_scalar('learning rate', cur_lr, epoch)
            loss_log[epoch] = avg_train_loss

        #  --------------------------------------------------------- evaluation ----------------------------------------------------------
        if epoch % opt.eval_freq == 0:
            model.eval()
            dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
            print(f"           | val_loss:   {val_losses:.4f} | val_dice:  {mean_dice:.4f} | best_dice: {best_dice:.4f}")
            if args.keep_log:
                TensorWriter.add_scalar('val_loss', val_losses, epoch)
                TensorWriter.add_scalar('dices', mean_dice, epoch)
                dice_log[epoch] = mean_dice
            if mean_dice > best_dice:
                best_dice = mean_dice
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(epoch) + '_' + str(best_dice)
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
                print(f"           >> New best! Saved to {save_path}.pth")
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1):
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            save_path = opt.save_path + args.modelname + opt.save_path_code + '_' + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
            if args.keep_log:
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/trainloss.txt', 'w') as f:
                    for i in range(len(loss_log)):
                        f.write(str(loss_log[i])+'\n')
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/dice.txt', 'w') as f:
                    for i in range(len(dice_log)):
                        f.write(str(dice_log[i])+'\n')

    total_time = time.time() - train_start
    print(f"\nTraining complete! Total time: {total_time/3600:.1f}h | Best dice: {best_dice:.4f}")

if __name__ == '__main__':
    main()