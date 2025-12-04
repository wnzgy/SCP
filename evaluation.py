import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import time
import json
import matplotlib.pyplot as plt
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append('./utils')

from torchvision.models import (
    resnet18, resnet34, resnet50,
    vgg16, vgg19, densenet121, densenet169,
    vit_b_16
)

# Import attack libraries
from torchattacks import PGD, CW, AutoAttack  # AutoAttack
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cw_params = {
    'c': 1,
    'steps': 10,
    'lr': 0.01
}

auto_params = {
    'norm': 'L2',
    'eps': 8 / 255,
    'version': 'standard'
}

dataset_configs = {
    'cifar10': {
        'num_classes': 10,
        'input_size': 32,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010)
    },
    'mnist': {
        'num_classes': 10,
        'input_size': 28,
        'mean': (0.1307,),
        'std': (0.3081,)
    },
    'tiny-imagenet': {
        'num_classes': 200,
        'input_size': 64,
        'mean': (0.4802, 0.4481, 0.3975),
        'std': (0.2302, 0.2265, 0.2262)
    },
    'cifar100': {
        'num_classes': 100,
        'input_size': 32,
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761)
    }
}

def get_dataloaders(dataset_name, model_name, batch_size):

    config = dataset_configs[dataset_name]
    if model_name == 'vit_b_16':
        input_size = 224
    else:
        input_size = config['input_size']

    if dataset_name == 'mnist':
        transform_test = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std'])
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std'])
        ])

    if dataset_name == 'cifar10':
        test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'mnist':
        test_set = datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'cifar100':
        test_set = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'tiny-imagenet':
        data_dir = '../data/tiny-imagenet-200'
        val_dir = os.path.join(data_dir, 'val')
        test_set = datasets.ImageFolder(val_dir, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return test_loader, config['num_classes']


def get_model(model_name, num_classes,mean, std):
    model = None
    model_name = model_name.lower()

    if 'resnet' in model_name:
        if model_name == 'resnet18':
            model = resnet18(weights=None)
        elif model_name == 'resnet34':
            model = resnet34(weights=None)
        elif model_name == 'resnet50':
            model = resnet50(weights=None)
        if model is not None:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vgg16':
        model = vgg16(weights=None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'densenet121':
        model = densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vit_b_16':
        model = vit_b_16(weights=None)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)

    if model is None:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.to(device)


def evaluate_defense(model, test_loader, criterion, adv_methods):

    model.eval()

    stats = {
        'clean': {'correct': 0, 'loss': 0.0},
    }


    attack_instances = {}
    for method in adv_methods:
        stats[method] = {'correct': 0, 'loss': 0.0}
        if method == 'cw':
            attack_instances['cw'] = CW(model, **cw_params)
        elif method == 'auto':
            attack_instances['auto'] = AutoAttack(model, norm=auto_params['norm'], eps=auto_params['eps'],
                                                  version=auto_params['version'])
        else:
            logger.warning(f"Unsupported attack method: {method}")

    total = 0
    pbar = tqdm(test_loader, desc='Evaluating', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = labels.size(0)
        total += batch_size


        with torch.no_grad():
            clean_outputs = model(inputs)
            stats['clean']['loss'] += criterion(clean_outputs, labels).item() * batch_size
            stats['clean']['correct'] += clean_outputs.argmax(1).eq(labels).sum().item()

        for method, attacker in attack_instances.items():
            if method == 'auto':

                pbar.set_description(f"Running AutoAttack ({auto_params['version']})")
            else:
                pbar.set_description(f"Running {method.upper()} Attack")

            adv_inputs = attacker(inputs, labels)

            with torch.no_grad():
                adv_outputs = model(adv_inputs)
                stats[method]['loss'] += criterion(adv_outputs, labels).item() * batch_size
                stats[method]['correct'] += adv_outputs.argmax(1).eq(labels).sum().item()

        postfix = {
            'clean_acc': f"{100. * stats['clean']['correct'] / total:.2f}%",
            'clean_loss': f"{stats['clean']['loss'] / total:.4f}"
        }

        for method in adv_methods:
            if method in stats:
                postfix[f'{method}_acc'] = f"{100. * stats[method]['correct'] / total:.2f}%"
                postfix[f'{method}_loss'] = f"{stats[method]['loss'] / total:.4f}"

        pbar.set_postfix(postfix)

    results = {
        'clean': {
            'loss': round(stats['clean']['loss'] / total, 4),
            'acc': round(100. * stats['clean']['correct'] / total, 2)
        }
    }

    for method in adv_methods:
        if method in stats:
            results[method] = {
                'loss': round(stats[method]['loss'] / total, 4),
                'acc': round(100. * stats[method]['correct'] / total, 2)
            }

    return results


def save_results_to_file(results, args, model_name_short):

    save_dir = os.path.join('experiments', args.dataset, args.model_arch)
    os.makedirs(save_dir, exist_ok=True)


    timestamp = time.strftime("%Y%m%d_%H%M%S")
    attacks_str = args.attacks.replace(',', '_')


    filename = f"{model_name_short}_{args.dataset}_{attacks_str}_{timestamp}"
    csv_path = os.path.join(save_dir, f"{filename}.csv")
    json_path = os.path.join(save_dir, f"{filename}.json")


    with open(csv_path, 'w') as f:
        f.write("Method,Loss,Accuracy(%)\n")
        for method, metrics in results.items():
            f.write(f"{method},{metrics['loss']},{metrics['acc']}\n")

    json_data = {
        "model": {
            "architecture": args.model_arch,
            "path": args.model_path,
            "name_short": model_name_short
        },
        "dataset": args.dataset,
        "attacks": args.attacks.split(','),
        "batch_size": args.batch_size,
        "timestamp": timestamp,
        "results": results,
        "auto_attack_config": {
            "version": args.auto_version,
            "eps": float(args.auto_eps),
            "norm": args.auto_norm
        } if 'auto' in args.attacks else None
    }

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"Results saved to: {csv_path} and {json_path}")
    return csv_path, json_path


def main():
    method = 'scp'
    dataset = 'cifar10'
    model = 'resnet18'


    defense_model_path = f'results/{method}/{dataset}/models/{model}_{dataset}_{method}#5_best.pth'

    parser = argparse.ArgumentParser(description='Defense Model Evaluation')
    parser.add_argument('--model_path', type=str, default=defense_model_path,
                        help='Path to trained model weights (e.g., ./results/cifar10/models/resnet18_cifar10_best.pth)')
    parser.add_argument('--model_arch', type=str, default=model,
                        choices=['vit_b_16', 'resnet18', 'vgg16', 'densenet121', 'resnet50', 'resnet34'],
                        help='Model architecture (e.g., resnet18)')
    parser.add_argument('--dataset', type=str, default=dataset,
                        choices=['cifar10', 'cifar100', 'mnist', 'tiny-imagenet'],
                        help='Dataset used for training (e.g., cifar10)')
    parser.add_argument('--attacks', type=str, default='cw,auto',
                        help='Comma-separated list of attack methods (e.g., cw,auto). Include auto for AutoAttack.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--logname', type=str, default='defense_evaluation', help='Log file prefix')
    parser.add_argument('--auto_version', type=str, default='standard',
                        choices=['standard', 'plus', 'rand'], help='AutoAttack version (standard/plus/rand)')
    parser.add_argument('--auto_eps', type=float, default=8 / 255,
                        help='AutoAttack epsilon (default: 8/255)')
    parser.add_argument('--auto_norm', type=str, default='L2', choices=['Linf', 'L2', 'L1'],
                        help='AutoAttack norm (Linf, L2, L1)')
    parser.add_argument('--save_dir', type=str, default='experiments/defense_evaluation/kl',
                        help='Directory to save results')

    args = parser.parse_args()

    # Create log directory
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)

    # Initialize logger
    log_file = os.path.join(logs_dir, f"{args.logname}-{time.strftime('%m-%d-%H-%M-%S')}-{args.logname}.log")
    logger.add(log_file, rotation="10 MB", level="INFO")

    logger.info(f"--- Starting Defense Evaluation ---")
    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Model Architecture: {args.model_arch}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Attacks: {args.attacks}")

    # Get dataloaders
    test_loader, num_classes = get_dataloaders(args.dataset, args.model_arch, args.batch_size)

    # Initialize model
    model = get_model(args.model_arch, num_classes,dataset_configs[dataset]['mean'],dataset_configs[dataset]['std'])

    # Load model weights
    if not os.path.exists(args.model_path):
        logger.error(f"Model weights file not found: {args.model_path}")
        sys.exit(1)

    try:
        # Check if checkpoint contains 'model_state_dict'
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)  # Otherwise assume direct state_dict
        logger.info(f"Successfully loaded model weights from {args.model_path}")
    except Exception as e:
        logger.error(f"Error loading model weights from {args.model_path}: {e}")
        sys.exit(1)

    criterion = nn.CrossEntropyLoss()

    adv_methods = [m.strip() for m in args.attacks.split(',') if m.strip()]


    if 'auto' in adv_methods:
        auto_params['version'] = args.auto_version
        auto_params['eps'] = args.auto_eps
        auto_params['norm'] = args.auto_norm
        logger.info(f"AutoAttack, version: {args.auto_version}, norm: {args.auto_norm}")


    results = evaluate_defense(model, test_loader, criterion, adv_methods)

    logger.info("--- Evaluation ---")
    for method, metrics in results.items():
        logger.info(f"  {method.upper()}: Loss = {metrics['loss']:.4f}, Accuracy = {metrics['acc']:.2f}%")
    logger.info("--------------------------")

    model_name_parts = os.path.basename(args.model_path).split('_')
    model_name_short = model_name_parts[0] if len(model_name_parts) > 0 else args.model_arch

    csv_path, json_path = save_results_to_file(results, args, model_name_short)
    logger.info(f"Evaluation complete. Results saved to CSV: {csv_path}")
    logger.info(f"Detailed results saved to JSON: {json_path}")


if __name__ == '__main__':
    main()
