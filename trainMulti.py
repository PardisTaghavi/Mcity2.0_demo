# ------------------------------------------------------------------------------
# The original code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# moddified by Pardis Taghavi (taghavi.pardis@gmail.com)
# For non-commercial purpose only (research, evaluation etc).
# -----------------------------------------------------------------------------

import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from models.modelMulti import GLPDepth, Critc
from models.optimizer import build_optimizers
import utils.metrics as metrics
from utils.criterion import SiLogLoss, CrossEntropyLoss
import utils.logging as logging
import torch.nn as nn
from dataset.mcity import mcity
import albumentations as A

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions
from tqdm import tqdm
import time
import glob
from tqdm import tqdm


metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog', 'mIoU', 'pixel_acc']


torch.manual_seed(0)
def load_model(ckpt, model, optimizer=None):
    ckpt_dict = torch.load(ckpt, map_location='cpu')
    
    state_dict = ckpt_dict['model']
    weights = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            weights[key[len('module.'):]] = value
        else:
            weights[key] = value
    model.load_state_dict(weights)

    if optimizer is not None:
        optimizer_state = ckpt_dict['optimizer']
        optimizer.load_state_dict(optimizer_state)


def freeze_encoder(model):
    #freeze encoder except prompt layers
    for name, param in model.named_parameters():
        if 'prompt' not in name and 'encoder' in name:
            param.requires_grad = False


def get_gradient(crit, real, fake, epsilon):

    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)
    gradient= torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient
    

def gradient_penalty(gradient):

    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def main():
    opt = TrainOptions()
    args = opt.initialize().parse_args()
    print(args)

    pretrain = args.pretrained.split('.')[0]
    maxlrstr = str(args.max_lr).replace('.', '')
    minlrstr = str(args.min_lr).replace('.', '')
    layer_decaystr = str(args.layer_decay).replace('.', '')
    weight_decaystr = str(args.weight_decay).replace('.', '')
    num_filter = str(args.num_filters[0]) if args.num_deconv > 0 else ''
    num_kernel = str(args.deconv_kernels[0]) if args.num_deconv > 0 else ''
    name = [args.dataset, str(args.batch_size), pretrain.split('/')[-1], 'deconv'+str(args.num_deconv), \
        str(num_filter), str(num_kernel), str(args.crop_h), str(args.crop_w), maxlrstr, minlrstr, \
        layer_decaystr, weight_decaystr, str(args.epochs)]
    if 'swin' in args.backbone:
        for i in args.window_size:
            name.append(str(i))
        for i in args.depths:
            name.append(str(i))
    if args.exp_name != '':
        name.append(args.exp_name)

    exp_name = '_'.join(name)
    print('This experiments: ', exp_name)

    # Logging
    exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + exp_name
    log_dir = os.path.join(args.log_dir, exp_name)
    logging.check_and_make_dirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)
    log_txt = os.path.join(log_dir, 'logs.txt')  
    logging.log_args_to_txt(log_txt, args)

    global result_dir
    result_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    model = GLPDepth(args=args)
    crit = Critc()

    torch.cuda.empty_cache()

    # CPU-GPU agnostic settings
    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)
    else:
        device = torch.device('cpu')

    model.to(device)
    crit.to(device)

    # Dataset setting
    # dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
    # dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

    # train_dataset = get_dataset(**dataset_kwargs, split='train')
    # val_dataset = get_dataset(**dataset_kwargs, split='val')

    transform = A.Compose([
        #orig_height = 600, orig_width = 1200 
        A.Resize(320, 608),
        A.HorizontalFlip(p=0.5),
        # A.RandomCrop(height=518, width=518, p=1),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5)

    ])


    dataset =  mcity(root_dir='/home/avalocal/mcity/data', transform=transform)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

    # #limit dataset to 1000 train and 100 val
    # train_dataset = torch.utils.data.Subset(train_dataset, range(3000))
    # val_dataset = torch.utils.data.Subset(val_dataset, range(300))

    print("train_dataset: ", len(train_dataset))
    print("val_dataset: ", len(val_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers, 
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True)
    
    # Training settings
    criterion_d = SiLogLoss()
    criterion_seg = CrossEntropyLoss()

    optimizer = build_optimizers(model, dict(type='AdamW', lr=args.max_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
                constructor='SwinLayerDecayOptimizerConstructor',
                paramwise_cfg=dict(num_layers=args.depths, layer_decay_rate=args.layer_decay, no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale'])))
    optimizer_crit = optim.Adam(crit.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001)

    
    start_ep = 1
    if args.resume_from:
        load_model(args.resume_from, model.module, optimizer)
        strlength = len('_model.ckpt')
        resume_ep = int(args.resume_from[-strlength-2:-strlength])
        print(f'resumed from epoch {resume_ep}, ckpt {args.resume_from}')
        start_ep = resume_ep + 1
    if args.auto_resume:
        ckpt_list = glob.glob(f'{log_dir}/epoch_*_model.ckpt')
        strlength = len('_model.ckpt')
        idx = [ckpt[-strlength-2:-strlength] for ckpt in ckpt_list]
        if len(idx) > 0:
            idx.sort(key=lambda x: -int(x))
            ckpt = f'{log_dir}/epoch_{idx[0]}_model.ckpt'
            load_model(ckpt, model.module, optimizer)
            resume_ep = int(idx[0])
            print(f'resumed from epoch {resume_ep}, ckpt {ckpt}')
            start_ep = resume_ep + 1

    global global_step
    iterations = len(train_loader)
    global_step = iterations * (start_ep - 1)

    # Perform experiment
    for epoch in tqdm(range(start_ep, args.epochs + 1)):
    #range(start_ep, args.epochs + 1):
        
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))

        loss_train = train(train_loader, model, crit, criterion_d, criterion_seg, log_txt, optimizer, optimizer_crit,
                           device=device, epoch=epoch, alpha=0.5, args=args)
        writer.add_scalar('Training loss', loss_train, epoch) #total loss
        
        #save evry 20 epochs
        
        if args.save_model and epoch % 20 == 0:
            print('Saving model...')
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                os.path.join(log_dir, 'epoch_%02d_model.ckpt' % epoch))

        if epoch % args.val_freq == 0:
            print('Validating...')
            results_dict, loss_val = validate(val_loader, model,crit, criterion_d, criterion_seg, device=device, epoch=epoch, args=args, alpha=0.5,
                                              log_dir=log_dir)
            writer.add_scalar('Val loss', loss_val, epoch) #total loss

            result_lines = logging.display_result(results_dict)
            if args.kitti_crop:
                print("\nCrop Method: ", args.kitti_crop)
            print(result_lines)

            with open(log_txt, 'a') as txtfile:
                txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
                txtfile.write(result_lines)                

            for each_metric, each_results in results_dict.items():
                writer.add_scalar(each_metric, each_results, epoch)
        torch.cuda.empty_cache()


def train(train_loader, model,crit, criterion_d, criterion_seg, log_txt, optimizer,optimizer_crit, device, epoch, alpha, args):   

    global global_step
    model.train()
    crit.train()

    depth_loss = logging.AverageMeter()
    seg_loss = logging.AverageMeter()
    loss = logging.AverageMeter()

    half_epoch = args.epochs // 2
    iterations = len(train_loader)
    result_lines = []
    for batch_idx, batch in enumerate(tqdm(train_loader)):
    #enumerate(train_loader):      

        global_step += 1

        if global_step < iterations * half_epoch:
            current_lr = (args.max_lr - args.min_lr) * (global_step /
                                            iterations/half_epoch) ** 0.9 + args.min_lr
        else:
            current_lr = max(args.min_lr, (args.min_lr - args.max_lr) * (global_step /
                                            iterations/half_epoch - 1) ** 0.9 + args.max_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr*param_group['lr_scale'] if 'swin' in args.backbone else current_lr

        input_RGB = batch['image'].to(device) 
        depth_gt = batch['depth'].to(device)  
        depth_gt = depth_gt/ depth_gt.max()

        dmin = 1e-3
        dmax = 100.0
        # depth_gt = depth_gt/ depth_gt.max() 
        seg_gt = batch['seg'].to(device)      
        
        
        
        preds = model(input_RGB)

        pred_depth = preds['pred_d'].squeeze() # 0-1 
        pred_seg = preds['pred_seg'].squeeze()
        pred_seg_max = torch.argmax(pred_seg, dim=1).unsqueeze(1).float()
        
        #shapes
        # print(pred_depth.shape, depth_gt.shape, "depth") #4, h, w | 4, h, w
        # print(pred_seg.shape, seg_gt.shape, "seg") #4, 20, h, w | 4, h, w
        #max min
        print("\n")
        print(pred_depth.max(), pred_depth.min(), "depth") # 0 - 100
        print(depth_gt.max(), depth_gt.min(), "depth_gt") # 0 -1
        print("\n")
        print(pred_seg_max.max(), pred_seg_max.min(), "seg") # 0 - 19
        print(seg_gt.max(), seg_gt.min(), "seg_gt")
        print(torch.unique(seg_gt), torch.unique(pred_seg_max.squeeze(1)), "unique seg")
        print("\n")
    

        fake_depth_seg = torch.cat((pred_depth.unsqueeze(1), pred_seg_max), dim=1)  
        true_depth_seg = torch.cat((depth_gt.unsqueeze(1), seg_gt.unsqueeze(1)), dim=1)        

        optimizer.zero_grad()  
        optimizer_crit.zero_grad()

        loss_d = criterion_d(pred_depth, depth_gt)
        loss_seg = criterion_seg(pred_seg, seg_gt.to(torch.long))
    
        crit_fake_pred = crit(fake_depth_seg.detach())
        crit_real_pred = crit(true_depth_seg)
        epsilon = torch.rand(len(true_depth_seg ), 1, 1, 1, device=device, requires_grad=True)
        gradient = get_gradient(crit, true_depth_seg, fake_depth_seg.detach(), epsilon)
        gp = gradient_penalty(gradient)
        crit_loss = -(torch.mean(crit_real_pred) - torch.mean(crit_fake_pred)) + 100 * gp
        crit_loss.backward(retain_graph=True)
        optimizer_crit.step()

        crit_fake_pred2 = crit(fake_depth_seg)
        gen_loss = -(torch.mean(crit_fake_pred2))
        total_loss = loss_seg + loss_d + 0.1 * gen_loss

        print("----- losses----")
        print("total_loss: ", total_loss)
        print("loss_d: ", loss_d)
        print("loss_seg: ", loss_seg)
        print("gen_loss: ", gen_loss)
        print("crit_loss: ", crit_loss)
        print("\n")

        # print("total_loss: ", total_loss)
        # print("loss_d: ", loss_d)
        # print("loss_seg: ", loss_seg)
        # print("gen_loss: ", gen_loss)


        total_loss.backward()                
        depth_loss.update(loss_d.item(), input_RGB.size(0)) #logging info   
        seg_loss.update(loss_seg.item(), input_RGB.size(0)) #logging info   
        loss.update(total_loss.item(), input_RGB.size(0)) #logging info   
        
        if args.pro_bar:
            logging.progress_bar(batch_idx, len(train_loader), args.epochs, epoch,   
                                ('Depth Loss: %.4f (%.4f)' %
                                (depth_loss.val, depth_loss.avg)),
                                ('Seg Loss: %.4f (%.4f)' %
                                (seg_loss.val, seg_loss.avg)),
                                ('Total Loss: %.4f (%.4f)' %
                                (loss.val, loss.avg))
                                )

        if batch_idx % args.print_freq == 0:                                  
            result_line = 'Epoch: [{0}][{1}/{2}]\t' \
                'Total Loss: {loss}, LR: {lr}\n'.format(
                    epoch, batch_idx, iterations,
                    loss=loss.avg, lr=current_lr
                )
            result_lines.append(result_line)
            print(result_line)
            
        optimizer.step()

    with open(log_txt, 'a') as txtfile:
        txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        for result_line in result_lines:
            txtfile.write(result_line)   

    return total_loss


def validate(val_loader, model, crit, criterion_d, criterion_seg, device, epoch, args, alpha,log_dir):

    depth_loss = logging.AverageMeter()
    seg_loss = logging.AverageMeter()
    loss = logging.AverageMeter()
    
    model.eval()
    crit.eval()

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in enumerate(tqdm(val_loader)):
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        seg_gt = batch['seg'].to(device)

        with torch.no_grad():
            s=time.time()
            pred = model(input_RGB)
            #print("time: ", time.time()-s)
        pred_d = pred['pred_d']
        pred_seg = pred['pred_seg']

        # if args.flip_test:
        #     batch_s = pred_d.shape[0]//2 #batch_s is 
        #     pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3]))/2.0
        #     pred_seg = (pred_seg[:batch_s] + torch.flip(pred_seg[batch_s:], [3]))/2.0
            
        # if args.shift_window_test:
        #     pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
        #     pred_s2 = torch.zeros((bs, args.num_classes, h, w), device=pred_seg.device)

        #     for i in range(args.shift_size):
        #         pred_s[..., :, i*interval:i*interval+h] += pred_d[i:i+1]
        #         pred_s2[..., :, i*interval:i*interval+h] += pred_seg[i:i+1]
        #     pred_d = pred_s/sliding_masks
        #     pred_seg = pred_s2/sliding_masks

        depth_gt = depth_gt.squeeze()     
        depth_gt = depth_gt/ depth_gt.max()
        pred_seg = pred_seg                 
        pred_seg_max = torch.argmax(pred_seg, dim=1)

        # print("----- validation----")
        # print(pred_d.shape, depth_gt.shape, "depth")
        # print(pred_seg.shape, seg_gt.shape, "seg")
        # print(pred_d.max(), pred_d.min(), "depth")
        # print(depth_gt.max(), depth_gt.min(), "depth_gt")
        # print(pred_seg.max(), pred_seg.min(), "seg")
        # print(seg_gt.max(), seg_gt.min(), "seg_gt")


        fake_depth_seg = torch.cat((pred_d, pred_seg_max.unsqueeze(1)), dim=1) 
        
        loss_d   = criterion_d(pred_d.squeeze(), depth_gt)
        loss_seg = criterion_seg(pred_seg, seg_gt.to(torch.long))
        loss_gen = - (torch.mean(crit(fake_depth_seg)))

        total_loss =   loss_seg + loss_d + 0.1 * loss_gen
        
        depth_loss.update(loss_d.item(), input_RGB.size(0))   
        seg_loss.update(loss_seg.item(), input_RGB.size(0))
        loss.update(total_loss.item(), input_RGB.size(0))


        print(pred_d.max(), pred_d.min(), "depth")
        print(depth_gt.max(), depth_gt.min(), "depth_gt")
        print(pred_seg_max.max(), pred_seg_max.min(), "seg")
        print(seg_gt.max(), seg_gt.min(), "seg_gt")


        ####EVALUATION METRICES Depth
        pred_crop, gt_crop = metrics.cropping_img(args, pred_d.squeeze(), depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)   


        ####EVALUATION METRICES Segmentation
        computed_result_seg = metrics.eval_seg(pred_seg_max, seg_gt)  

        
        loss_d = depth_loss.avg
        loss_seg = seg_loss.avg
        total_loss = loss.avg

        if args.pro_bar:
            logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

        for key in result_metrics.keys():
            #(key)
            if key == 'mIoU':
                result_metrics[key] += computed_result_seg[key]
            elif key == 'pixel_acc':
                result_metrics[key] += computed_result_seg[key]
            else:
                result_metrics[key] += computed_result[key]
            
    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    return result_metrics, total_loss


if __name__ == '__main__':
    main()