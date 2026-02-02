import torch
import argparse
import sys
import os
import random
import numpy as np
import math
from copy import deepcopy

# Add paths để import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.icbhi_dataset import ICBHIDataset
from torchvision import transforms
import time
from util.misc import AverageMeter, accuracy
from util.icbhi_util import get_score
from models import get_backbone_class
from method.dann import ProjectionHead, PCSLWithDANNLighter
import torch.nn as nn
import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluation with pretrained encoder and classifier')
    parser.add_argument('--two_cls_eval', action='store_true',
                        help='evaluate with two classes')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')
    # dataset
    parser.add_argument('--dataset', type=str, default='icbhi')
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # icbhi dataset
    parser.add_argument('--class_split', type=str, default='lungsound',
                        help='lungsound: (normal, crackles, wheezes, both), diagnosis: (healthy, chronic diseases, non-chronic diseases)')
    parser.add_argument('--n_cls', type=int, default=0,
                        help='set k-way classification problem for class')
    parser.add_argument('--d_cls', type=int, default=0,
                        help='set k-way classification problem for device (meta)')
    parser.add_argument('--test_fold', type=str, default='official', choices=['official', '0', '1', '2', '3', '4'],
                        help='test fold to use official 60-40 split or 80-20 split from RespireNet')
    parser.add_argument('--sample_rate', type=int,  default=16000, 
                        help='sampling rate when load audio data, and it denotes the number of samples per one second')
    parser.add_argument('--desired_length', type=int,  default=8, 
                        help='fixed length size of individual cycle')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='the number of mel filter banks')
    parser.add_argument('--pad_types', type=str,  default='repeat', 
                        help='zero: zero-padding, repeat: padding with duplicated samples, aug: padding with augmented samples')
    parser.add_argument('--resz', type=float, default=1, 
                        help='resize the scale of mel-spectrogram')
    parser.add_argument('--raw_augment', type=int, default=0, 
                        help='control how many number of augmented raw audio samples')
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup', 
                        help='policy (argument values) for SpecAugment')
    parser.add_argument('--specaug_mask', type=str, default='mean', 
                        help='specaug mask value', choices=['mean', 'zero'])
    parser.add_argument('--nospec', action='store_true')

    # model
    parser.add_argument('--model', type=str, default='beats')
    parser.add_argument('--pretrained', action='store_true')
    
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained encoder model')
    parser.add_argument('--from_sl_official', action='store_true',
                        help='load from supervised imagenet-pretrained model (official PyTorch)')
    parser.add_argument('--ma_update', action='store_true',
                        help='whether to use moving average update for model')
    parser.add_argument('--ma_beta', type=float, default=0,
                        help='moving average value')
    
    # for AST
    parser.add_argument('--audioset_pretrained', action='store_true',
                        help='load from imagenet- and audioset-pretrained model')
    parser.add_argument('--method', type=str, default='ce') 
    

    
    # Projection
    parser.add_argument('--norm_type', type=str, default='bn',
                        help='normalization type', choices=['bn', 'ln'])
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='projection hidden dimension')
    parser.add_argument('--output_dim', type=int, default=128,
                        help='projection output dimension')
    parser.add_argument('--proj_type', type=str, default='end2end',
                        help='projection type', choices=['end2end', 'feat_fixed', 'proj_fixed'])
    parser.add_argument('--lambda_pcsl', type=float, default=0.1,
                        help='lambda for patient variance ratio loss')
    parser.add_argument('--w_ce', type=float, default=1.0,
                    help='weight for classification loss')
    parser.add_argument('--w_projection', type=float, default=0.5,
                    help='weight for projection loss (PCSL + DANN)')

    # DANN arguments
    parser.add_argument('--use_dann', action='store_true',
                        help='use DANN for domain adaptation')
    parser.add_argument('--lambda_dann', type=float, default=0.1,
                        help='weight for DANN loss')

    
                        
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)

    
    args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    args.d_cls = 2    
            
    if args.dataset == 'icbhi':
        if args.class_split == 'lungsound':
            if args.n_cls == 4:
                args.cls_list = ['normal', 'crackle', 'wheeze', 'both']
            elif args.n_cls == 2:
                args.cls_list = ['normal', 'abnormal']
            
            args.device_list = ['L', 'A', 'M', '3']
                
        elif args.class_split == 'diagnosis':
            if args.n_cls == 3:
                args.cls_list = ['healthy', 'chronic_diseases', 'non-chronic_diseases']
            elif args.n_cls == 2:
                args.cls_list = ['healthy', 'unhealthy']
            else:
                raise NotImplementedError
        
    else:
        raise NotImplementedError
    
    if args.n_cls == 0 and args.m_cls !=0:
        args.n_cls = args.m_cls
        args.cls_list = args.meta_cls_list

    return args

def get_device():
    """Get the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def load_model_from_checkpoint(args):
    """
    Load model, classifier, and projector from checkpoint file
    """
    # Sử dụng checkpoint path từ args hoặc tự động tạo
    if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = os.path.join(args.save_folder, 'best.pth')
    
    # Nếu path bắt đầu bằng /save, thêm root path
    if checkpoint_path.startswith('/save'):
        checkpoint_path = checkpoint_path.replace('/save', args.save_dir)

    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get args from checkpoint if available, otherwise use provided args
    if 'args' in checkpoint:
        saved_args = checkpoint['args']
        
        # Print all important parameters from checkpoint for comparison
        print("\n" + "="*60)
        print("CHECKPOINT PARAMETERS (from saved checkpoint):")
        print("="*60)
        important_params = [
            'model', 'method', 'n_cls', 'seed', 'batch_size', 'desired_length',
            'learning_rate', 'weight_decay', 'optimizer', 'cosine',
            'from_sl_official', 'audioset_pretrained', 'ma_update', 'ma_beta',
            'norm_type', 'hidden_dim', 'output_dim', 'proj_type',
            'lambda_pcsl', 'lambda_dann', 'use_dann',
            'w_ce', 'w_projection', 'nospec', 'class_split', 'test_fold',
            'pad_types', 'n_mels', 'resz', 'sample_rate'
        ]
        
        for param in important_params:
            if hasattr(saved_args, param):
                value = getattr(saved_args, param)
                print(f"  {param:20s}: {value}")
        
        print("="*60)
        print("\nCURRENT ARGUMENTS (from command line):")
        print("="*60)
        for param in important_params:
            if hasattr(args, param):
                value = getattr(args, param)
                print(f"  {param:20s}: {value}")
        
        print("="*60)
        
        # Check for mismatches
        print("\nPARAMETER COMPARISON:")
        print("="*60)
        mismatches = []
        for param in important_params:
            if hasattr(saved_args, param) and hasattr(args, param):
                saved_val = getattr(saved_args, param)
                current_val = getattr(args, param)
                if saved_val != current_val:
                    mismatches.append((param, saved_val, current_val))
                    print(f"  ⚠️  MISMATCH: {param:20s} | Checkpoint: {saved_val} | Current: {current_val}")
        
        if not mismatches:
            print("  ✅ All parameters match!")
        else:
            print(f"\n  ⚠️  Found {len(mismatches)} parameter mismatches!")
            print("  Using checkpoint values for model creation...")
        
        print("="*60 + "\n")
        
        # Update ALL important args with saved values from checkpoint (prioritize checkpoint values)
        # This ensures model is created with exact same parameters as training
        updated_params = []
        for param in important_params:
            if hasattr(saved_args, param):
                saved_val = getattr(saved_args, param)
                if hasattr(args, param):
                    current_val = getattr(args, param)
                    if saved_val != current_val:
                        setattr(args, param, saved_val)
                        updated_params.append(f"{param}={saved_val}")
                else:
                    # If param doesn't exist in args, add it
                    setattr(args, param, saved_val)
                    updated_params.append(f"{param}={saved_val}")
        
        # Also update any other args that might be in checkpoint but not in important_params
        # Copy all attributes from saved_args to args (for any missing ones)
        for attr_name in dir(saved_args):
            if not attr_name.startswith('_') and not callable(getattr(saved_args, attr_name)):
                if not hasattr(args, attr_name):
                    setattr(args, attr_name, getattr(saved_args, attr_name))
        
        if updated_params:
            print(f"✅ Updated {len(updated_params)} parameters from checkpoint:")
            for i, param_str in enumerate(updated_params[:10]):  # Show first 10
                print(f"   - {param_str}")
            if len(updated_params) > 10:
                print(f"   ... and {len(updated_params) - 10} more")
        else:
            print("✅ All parameters already match checkpoint")
        
        print(f"\nKey parameters: model={args.model}, method={args.method}, n_cls={args.n_cls}, seed={args.seed}, desired_length={args.desired_length}")
    
    # Create model
    kwargs = {}
    if args.model == 'beats' or args.model == 'hftt':
        kwargs['spec_transform'] = None  # No augmentation for testing
    
    model_class = get_backbone_class(args.model)
    if callable(model_class):
        model = model_class(**kwargs)
    else:
        model = model_class(**kwargs)
    
    # Load pretrained weights if from_sl_official
    if args.from_sl_official and args.model not in ['ast', 'beats', 'hftt']:
        model.load_sl_official_weights()
        print('Pretrained model loaded from PyTorch ImageNet-pretrained')
    
    # Get model feature dimension
    model_feat_dim = model.final_feat_dim
    
    # Create classifier
    classifier = nn.Linear(model_feat_dim, args.n_cls)
    
    # Create projector (if method is projectors_loss)
    if args.method == 'projectors_loss':
        if args.model in ['beats', 'hftt']:
            projector = ProjectionHead(
                model_feat_dim, 
                args.hidden_dim, 
                args.output_dim, 
                attention=True, 
                norm_type=args.norm_type, 
                proj_type=args.proj_type
            )
        else:
            projector = ProjectionHead(
                model_feat_dim, 
                args.hidden_dim, 
                args.output_dim, 
                attention=False, 
                norm_type=args.norm_type, 
                proj_type=args.proj_type
            )
    else:
        projector = nn.Identity()
    
    # Load state dicts - handle module prefix if model was trained with DataParallel
    model_state = checkpoint['model']
    new_model_state = {}
    for k, v in model_state.items():
        if "module." in k:
            k = k.replace("module.", "")
        new_model_state[k] = v
    
    # Try strict=True first to see if all keys match
    try:
        missing_keys, unexpected_keys = model.load_state_dict(new_model_state, strict=True)
        print("Model loaded with strict=True (all keys matched)")
    except RuntimeError as e:
        print(f"Model load with strict=True failed: {e}")
        # Fall back to strict=False
        missing_keys, unexpected_keys = model.load_state_dict(new_model_state, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in model ({len(missing_keys)} keys): {list(missing_keys)[:10]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in model ({len(unexpected_keys)} keys): {list(unexpected_keys)[:10]}...")
    
    classifier_state = checkpoint['classifier']
    # Handle if classifier is a list
    if isinstance(classifier_state, list):
        classifier_state = classifier_state[0]
    new_classifier_state = {}
    for k, v in classifier_state.items():
        if "module." in k:
            k = k.replace("module.", "")
        new_classifier_state[k] = v
    
    # Try strict=True first
    try:
        missing_keys, unexpected_keys = classifier.load_state_dict(new_classifier_state, strict=True)
        print("Classifier loaded with strict=True (all keys matched)")
    except RuntimeError as e:
        print(f"Classifier load with strict=True failed: {e}")
        missing_keys, unexpected_keys = classifier.load_state_dict(new_classifier_state, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in classifier: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in classifier: {unexpected_keys}")
    
    # Load projector if available in checkpoint
    # Note: Projector may not be saved in checkpoint if method != 'projectors_loss'
    # But we still need to create it for consistency
    if 'projector' in checkpoint and args.method == 'projectors_loss':
        projector_state = checkpoint['projector']
        # Handle if projector is a list
        if isinstance(projector_state, list):
            projector_state = projector_state[0]
        new_projector_state = {}
        for k, v in projector_state.items():
            if "module." in k:
                k = k.replace("module.", "")
            new_projector_state[k] = v
        missing_keys, unexpected_keys = projector.load_state_dict(new_projector_state, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in projector: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in projector: {unexpected_keys}")
        print("Loaded projector from checkpoint")
    else:
        print("Projector not found in checkpoint or not needed (this is normal for projectors_loss method)")
    
    # Move to GPU if available
    device = get_device()
    model = model.to(device)
    classifier = classifier.to(device)
    projector = projector.to(device)
    
    print(f"Model loaded successfully on device: {device}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    return model, classifier, projector, device

def set_loader(args):
    if args.dataset == 'icbhi':        
        args.h = int(args.desired_length * 100 - 2)
        args.w = 128
        
        
        if args.model in ['beats', 'hftt']:  # Both do their own preprocessing
            train_transform = None
            val_transform = None

        else:
            train_transform = [transforms.ToTensor(),
                            transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
            val_transform = [transforms.ToTensor(),
                        transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        
            ##
            train_transform = transforms.Compose(train_transform)
            val_transform = transforms.Compose(val_transform)
        

        train_dataset = ICBHIDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
        val_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=args,  print_flag=True)
    
        
        
    else:
        raise NotImplemented    
    
    
   
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    
    
    return train_loader, val_loader, args

def validate(val_loader, model, classifier, criterion, args, best_acc, best_model=None, projector=None, device=None):
    if device is None:
        device = get_device()
    save_bool = False
    model.eval()

    classifier.eval()
    projector.eval()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, _) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            
            class_labels = labels[0].to(device, non_blocking=True)
        
            labels = class_labels.to(device, non_blocking=True)
            bsz = labels.shape[0]
                  
            # Use autocast only for CUDA, MPS doesn't support it
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    if args.model == 'hftt':
                        features = model(images,training=False)
                        output = classifier(features)
                        output = output.mean(dim=1)
                    
                    else:
                        features = model(images, args=args, training=False)
                        output = classifier(features)
                    
                    # Handle criterion - can be list or single loss
                    if isinstance(criterion, list):
                        loss = criterion[0](output, labels)
                    else:
                        loss = criterion(output, labels)
            else:
                # MPS/CPU: no autocast
                if args.model == 'hftt':
                    features = model(images,training=False)
                    output = classifier(features)
                    output = output.mean(dim=1)
                
                else:
                    features = model(images, args=args, training=False)
                    output = classifier(features)
                
                # Handle criterion - can be list or single loss
                if isinstance(criterion, list):
                    loss = criterion[0](output, labels)
                else:
                    loss = criterion(output, labels)
                
                
            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            _, preds = torch.max(output, 1)
            
            
        
            for idx in range(preds.shape[0]):
                counts[labels[idx].item()] += 1.0
                if not args.two_cls_eval:
                    if preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                else:  # only when args.n_cls == 4
                    if labels[idx].item() == 0 and preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                    elif labels[idx].item() != 0 and preds[idx].item() > 0:  # abnormal
                        hits[labels[idx].item()] += 1.0

            # Calculate metrics after each batch (giống như trong main.py)
            sp, se, sc, f1_normal = get_score(hits, counts)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'S_p {sp:.3f}\t'
                      'S_e {se:.3f}\t'
                      'Score {sc:.3f}\t'
                      'F1 Score {f1:.3f}'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1, sp=sp, se=se, sc=sc,
                       f1=f1_normal))
    
    # Calculate final metrics after all batches (đảm bảo tính lại sau loop)
    sp, se, sc, f1_normal = get_score(hits, counts)
    
    # Debug: Print exact values to compare with training
    print(f"\nDebug - hits: {hits}")
    print(f"Debug - counts: {counts}")
    print(f"Debug - sp: {hits[0]} / ({counts[0]} + 1e-10) * 100 = {sp:.6f}")
    print(f"Debug - se: {sum(hits[1:])} / ({sum(counts[1:])} + 1e-10) * 100 = {se:.6f}")
    print(f"Debug - sc: ({sp:.6f} + {se:.6f}) / 2.0 = {sc:.6f}")
    
    # Update best_acc if current results are better
    if sc > best_acc[-2] and se > 0.1:
        save_bool = True
        best_acc = [sp, se, sc, f1_normal]
        if best_model is not None:
            best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict()), deepcopy(projector.state_dict())]

    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(sp, se, sc, best_acc[0], best_acc[1], best_acc[2]))
    print(' * F1 Score: {:.2f} (F1 Score: {:.2f})'.format(f1_normal, best_acc[3]))
    print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))

    return best_acc, best_model, save_bool

if __name__ == '__main__':
    args = parse_args()
    
    # Initialize best_acc giống như trong main.py
    best_model = None
    if args.dataset == 'icbhi':
        best_acc = [0, 0, 0, 0]  # Specificity, Sensitivity, Score, F1
    
    # Load model from checkpoint (this will update args from checkpoint)
    model, classifier, projector, device = load_model_from_checkpoint(args)
    
    # Fix seed AFTER loading checkpoint (to use seed from checkpoint if different)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    train_loader, val_loader, args = set_loader(args)
    
    # Create criterion as list giống như trong main.py
    criterion = [nn.CrossEntropyLoss().to(device)]

    # Set models to evaluation mode - ensure all submodules are in eval mode
    model.eval()
    classifier.eval()
    projector.eval()
    
    # Double check - ensure no batch norm or dropout layers are in training mode
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout)):
            module.eval()
    
    # Validate
    best_acc, best_model, save_bool = validate(val_loader, model, classifier, criterion, args, best_acc, best_model, projector, device)
    
    # Display final results
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    print(f"Specificity (S_p):  {best_acc[0]:.2f}%")
    print(f"Sensitivity (S_e):  {best_acc[1]:.2f}%")
    print(f"Score:              {best_acc[2]:.2f}%")
    print(f"F1 Score:           {best_acc[3]:.2f}")
    print("="*60)
