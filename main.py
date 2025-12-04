import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import (densenet121, densenet169, resnet18, resnet34,
                                resnet50, vgg16, vgg19, vit_l_16,vit_b_16,wide_resnet50_2,wide_resnet101_2)
from tqdm import tqdm
from scp import SCP
from spp import SPP
from torchattacks import CW, GN, PGD, TPGD, AutoAttack, FGSM, MIFGSM,VNIFGSM,PIFGSM


sys.path.append('./utils')
import myutil


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataloaders(dataset_name, model_name, dataset_configs, batch_size):

    config = dataset_configs[dataset_name]


    if model_name == 'vit_b_16' or model_name == 'vit_l_16':
        input_size = 224
    else:
        input_size = config['input_size']


    if dataset_name == 'mnist':
        transform_train = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std'])
        ])
        transform_test = transform_train
    else:
        transform_train = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std'])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std'])
        ])


    if dataset_name == 'cifar10':
        train_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'mnist':
        train_set = datasets.MNIST(root='../data', train=True, download=True, transform=transform_train)
        test_set = datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'cifar100':
        train_set = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'tiny-imagenet':

        data_dir = '../data/tiny-imagenet-200'
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')

        train_set = datasets.ImageFolder(train_dir, transform=transform_train)
        test_set = datasets.ImageFolder(val_dir, transform=transform_test)

    # Create DataLoaders

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)


    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, config['num_classes'], input_size


def get_model(model_name, num_classes):

    if model_name.startswith('resnet') or model_name.startswith('wide_resnet'):
        if model_name == 'resnet18':
            model = resnet18(weights=None)
        elif model_name == 'resnet34':
            model = resnet34(weights=None)
        elif model_name == 'resnet50':
            model = resnet50(weights=None)
        elif model_name == 'wide_resnet101_2':
            model = wide_resnet101_2(weights=None)
        elif model_name == 'wide_resnet50_2':
            model = wide_resnet50_2(weights=None)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vgg16':
        model = vgg16(weights=None)

        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vgg19':
        model = vgg19(weights=None)


        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'densenet121':
        model = densenet121(weights=None)


        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'densenet169':
        model = densenet169(weights=None)

        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vit_b_16':
        model = vit_b_16(weights=None)


        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vit_l_16':
        model = vit_l_16(weights=None)


        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)

    return model.to(device)


def train_model(model, train_loader, criterion, optimizer, epoch, num_epochs, num_classes, adv_method='',history_file=None,train_params=None,data_name=None):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    training_history = {}
    if history_file and os.path.exists(history_file):
        training_history = myutil.read_yaml(history_file)
        if 'epoch_history' not in training_history:
            training_history['epoch_history'] = []
    else:
        training_history = {
            'experiment_info': {},
            'hyperparameters': {},
            'final_results': {},
            'epoch_history': []
        }

    if adv_method.startswith('tpgd'):
        steps = int(adv_method.split('#')[1])
        torch_attack = TPGD(model, eps=train_params['eps'], alpha=train_params['alpha'], steps=steps)
    elif adv_method.startswith('pgd'):
        steps = int(adv_method.split('#')[1])
        torch_attack = PGD(model, eps=train_params['eps'], alpha=train_params['alpha'], steps=steps)
    elif adv_method.startswith('scp'):
        steps = int(adv_method.split('#')[1])
        torch_attack = SCP(model, eps=train_params['eps'], alpha=train_params['alpha'],steps=steps)
    elif adv_method.startswith('spp'):
        steps = int(adv_method.split('#')[1])
        torch_attack = SPP(model, eps=train_params['eps'], alpha=train_params['alpha'],steps=steps)
    elif adv_method.startswith('fgsm'):
        torch_attack = FGSM(model, eps=train_params['eps'])
    elif adv_method.startswith('mifgsm'):
        steps = int(adv_method.split('#')[1])
        torch_attack = MIFGSM(model, eps=train_params['eps'], alpha=train_params['alpha'],steps=steps)
    elif adv_method.startswith('autoattack'):
        torch_attack = AutoAttack(model, eps=train_params['eps'])
    elif adv_method.startswith('pifgsm'):
        steps = int(adv_method.split('#')[1])
        torch_attack = PIFGSM(model, max_epsilon=train_params['eps'], num_iter_set=steps)
    elif adv_method.startswith('vnifgsm'):
        steps = int(adv_method.split('#')[1])
        torch_attack = VNIFGSM(model, eps=train_params['eps'],alpha=train_params['alpha'], steps=steps)

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
    batch_idx = 0

    for inputs, labels in pbar:
        batch_idx += 1
        inputs, labels = inputs.to(device), labels.to(device)


        if adv_method == '':
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        else:

            adv_inputs = torch_attack.forward(inputs, labels)
            inputs = torch.cat([inputs, adv_inputs], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        current_loss = running_loss / (batch_idx)
        current_acc = 100. * correct / total
        pbar.set_postfix({'loss': current_loss, 'acc': current_acc})

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total

    return train_loss, train_acc


def test_model(
        model,
        test_loader,
        criterion,
        num_classes,
        adv_methods=None,
        pgd_params=None,
        cw_params=None,
        dataset_configs=None,
        data_name=None,
):

    model.eval()
    stats = {
        'clean': {'correct': 0, 'loss': 0.0},
    }


    if adv_methods is None:
        adv_methods = []
    for method in adv_methods:
        stats[method] = {'correct': 0, 'loss': 0.0}

    pgd_attack = PGD(model, eps=pgd_params['eps'], alpha=pgd_params['alpha'], steps=pgd_params['steps'],
                     random_start=pgd_params['random_start'])

    cw_attack = CW(model, c=cw_params['c'], steps=cw_params['steps'], lr=cw_params['lr'])

    auto_attack = AutoAttack(model)

    total = 0
    pbar = tqdm(test_loader, desc='Testing', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        batch_size = labels.size(0)
        total += batch_size


        clean_outputs = model(inputs)
        stats['clean']['loss'] += criterion(clean_outputs, labels).item() * batch_size
        stats['clean']['correct'] += clean_outputs.argmax(1).eq(labels).sum().item()


        for method in adv_methods:
            if method == 'pgd':

                adv_inputs = pgd_attack(inputs, labels)

            elif method == 'cw':

                adv_inputs = cw_attack(inputs, labels)

            elif method == 'auto':
                adv_inputs = auto_attack(inputs, labels)

            adv_outputs = model(adv_inputs)

            stats[method]['loss'] += criterion(adv_outputs, labels).item() * batch_size
            stats[method]['correct'] += adv_outputs.argmax(1).eq(labels).sum().item()

        postfix = {
            'clean_acc': 100. * stats['clean']['correct'] / total,
            'clean_loss': stats['clean']['loss'] / total
        }

        if adv_methods is not None:
            for method in adv_methods:
                if method in stats:
                    postfix[f'{method}_acc'] = 100. * stats[method]['correct'] / total
                    postfix[f'{method}_loss'] = stats[method]['loss'] / total

        pbar.set_postfix(postfix)

    results = {
        'clean': [
            round(stats['clean']['loss'] / len(test_loader.dataset), 2),
            round(100. * stats['clean']['correct'] / total, 2)
        ]
    }

    for method in adv_methods:
        if method in stats:
            results[method] = [
                round(stats[method]['loss'] / len(test_loader.dataset), 2),
                round(100. * stats[method]['correct'] / total, 2)
            ]
    return results


def main(model_name, data_name, adv_train_method, num_epochs, batch_size, results_dir, lr):

    pgd_params = {
        'eps': 8 / 255,
        'alpha': 2 / 255,
        'steps': 10,
        'random_start': True
    }

    cw_params = {
        'c': 1,
        'steps': 10,
        'lr': 0.01
    }
    train_params = {
        'eps': 8 / 255,
        'alpha': 2 / 255,
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
    base_results_dir = os.path.join(results_dir, data_name)

    # Correctly create directories without nesting data_name
    models_dir = os.path.join(base_results_dir, 'models')
    logs_dir = os.path.join(base_results_dir, 'logs')
    weights_dir = os.path.join(base_results_dir, 'weights')

    if adv_train_method:
        attack_weights_dir = os.path.join(weights_dir, adv_train_method)
        os.makedirs(attack_weights_dir, exist_ok=True)


    os.makedirs(base_results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)


    train_loader, test_loader, num_classes, input_size = get_dataloaders(data_name, model_name, dataset_configs,
                                                                         batch_size)

    logger.info(f"----- Training model: {model_name} on {data_name} with classes: {num_classes} -----")
    logger.info(f"Adversarial training method: {adv_train_method if adv_train_method != '' else 'None'}")


    model = get_model(model_name, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_acc = 0.0
    best_robust_score = 0.0
    start_epoch = 0
    total_mem = 0.0
    total_time = 0.0


    history_file = os.path.join(logs_dir, f'{model_name}_{data_name}_{adv_train_method}_history.yaml')
    log_file = os.path.join(logs_dir, f'{model_name}_{data_name}_{adv_train_method}.yaml')

    if adv_train_method:
        weights_file = os.path.join(attack_weights_dir, f'{model_name}_{data_name}_{adv_train_method}_weights.pth')
        last_checkpoint_file = os.path.join(attack_weights_dir,
                                            f'{model_name}_{data_name}_{adv_train_method}_last_checkpoint.pth')
    else:
        weights_file = os.path.join(weights_dir, f'{model_name}_{data_name}_weights.pth')
        last_checkpoint_file = os.path.join(weights_dir, f'{model_name}_{data_name}_last_checkpoint.pth')


    training_history = {}
    if os.path.exists(history_file):
        training_history = myutil.read_yaml(history_file)
        if 'epoch_history' in training_history and len(training_history['epoch_history']) > 0:
            start_epoch = training_history['epoch_history'][-1]['epoch']
            best_acc = training_history['final_results'].get('best_accuracy', 0.0)
            logger.info(f"Found training history, resuming from epoch {start_epoch}")
            logger.info(f"Current best accuracy: {best_acc:.2f}%")

    if os.path.exists(last_checkpoint_file):
        checkpoint = torch.load(last_checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_acc = checkpoint.get('best_acc', best_acc)
        saved_epoch = checkpoint.get('epoch', -1)
        logger.info(f"Loaded latest checkpoint: {last_checkpoint_file}. Saved epoch {saved_epoch}")
        logger.info(f"Best accuracy in checkpoint: {best_acc:.2f}%")

        if saved_epoch > start_epoch:
            start_epoch = saved_epoch
            logger.info(f"Resuming from checkpoint, new start epoch: {start_epoch}")
    elif os.path.exists(weights_file):
        checkpoint = torch.load(weights_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_acc = checkpoint.get('best_acc', best_acc)
        logger.info(f"Loading model weights: {weights_file}")
        logger.info(f"Best accuracy in loaded checkpoint: {best_acc:.2f}%")


    if not os.path.exists(history_file) or not (
            training_history.get('epoch_history') and len(training_history['epoch_history']) > 0):
        training_history = {
            'experiment_info': {
                'model': model_name,
                'dataset': data_name,
                'attack_method': adv_train_method,
                'num_classes': num_classes,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'learning_rate': lr,
                'optimizer': 'AdamW'
            },
            'hyperparameters': {
                'eps': train_params['eps'],
                'alpha': train_params['alpha'],
                'pgd_steps': pgd_params['steps'],
                'cw_steps': cw_params['steps'],
            },
            'final_results': {
                'best_accuracy': 0.0,
                'final_accuracy': 0.0,
                'avg_training_time': 0.0,
                'avg_memory_usage': 0.0
            },
            'epoch_history': []
        }


    myutil.write_yaml(history_file, training_history)

    for epoch in range(start_epoch, num_epochs):

        torch.cuda.empty_cache()

        start_time = time.time()
        train_loss, train_acc = train_model(
            model, train_loader, criterion, optimizer, epoch, num_epochs, num_classes,
            adv_method=adv_train_method, history_file=history_file,
            train_params=train_params, data_name=data_name
        )

        epoch_time = time.time() - start_time
        total_time += epoch_time


        after_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        total_mem += after_mem

        logger.info(f"Epoch {epoch} | Time: {epoch_time:.2f}s | Memory: {after_mem:.2f} MB")


        results = test_model(model, test_loader, criterion, num_classes, ['pgd', 'cw'], pgd_params, cw_params,dataset_configs=dataset_configs, data_name=data_name)

        test_acc = results['clean'][1]
        test_loss = results['clean'][0]

        logger.info(f'Epoch {epoch + 1}/{num_epochs}: trloss: {train_loss:.2f} tracc: {train_acc} '
                    f'Clean Loss: {results["clean"][0]:.4f}, Clean Acc: {results["clean"][1]:.2f}% '
                    f'PGD Loss: {results.get("pgd", [0, 0])[0]:.4f}, PGD Acc: {results.get("pgd", [0, 0])[1]:.2f}% '
                    f'CW Loss: {results.get("cw", [0, 0])[0]:.4f}, CW Acc: {results.get("cw", [0, 0])[1]:.2f}%')


        if os.path.exists(history_file):
            training_history = myutil.read_yaml(history_file)


        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 2),
            'test_loss': round(test_loss, 4),
            'test_acc': round(test_acc, 2),
            'pgd_loss': round(results.get('pgd', [0, 0])[0], 4),
            'pgd_acc': round(results.get('pgd', [0, 0])[1], 2),
            'cw_loss': round(results.get('cw', [0, 0])[0], 4),
            'cw_acc': round(results.get('cw', [0, 0])[1], 2),
            'epoch_time': round(epoch_time, 2),
            'memory_usage': round(after_mem, 2),
            'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }


        existing_epoch = next((item for item in training_history['epoch_history']
                               if item.get('epoch') == epoch + 1), None)
        if existing_epoch:
            training_history['epoch_history'].remove(existing_epoch)
        training_history['epoch_history'].append(epoch_record)


        myutil.write_yaml(log_file, results)


        myutil.write_yaml(history_file, training_history)
        logger.info(f"Epoch {epoch + 1} history saved to {history_file}")

        pgd_acc = results.get('pgd', [0, 0])[1]
        cw_acc = results.get('cw', [0, 0])[1]
        robust_score = test_acc + pgd_acc + cw_acc

        if robust_score > best_robust_score:
            best_robust_score = robust_score
            best_acc = test_acc
            model_path = os.path.join(models_dir, f'{model_name}_{data_name}_{adv_train_method}_best.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f'Best model saved at: {model_path} with robust score: {best_robust_score:.2f} '
                        f'(clean: {test_acc:.2f}%, pgd: {pgd_acc:.2f}%, cw: {cw_acc:.2f}%)')


            training_history['final_results']['best_accuracy'] = round(best_acc, 2)
            training_history['final_results']['best_robust_score'] = round(best_robust_score, 2)
            training_history['final_results']['best_pgd_acc'] = round(pgd_acc, 2)
            training_history['final_results']['best_cw_acc'] = round(cw_acc, 2)
            myutil.write_yaml(history_file, training_history)


        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_acc': best_acc,
            'current_acc': test_acc,
        }, last_checkpoint_file)
        logger.info(f'Saved checkpoint at epoch {epoch + 1} to {last_checkpoint_file}')


    avg_time = total_time / num_epochs
    avg_mem = total_mem / num_epochs


    final_model_path = os.path.join(models_dir, f'{model_name}_{data_name}_{adv_train_method}_final.pth')
    torch.save(model.state_dict(), final_model_path)


    if adv_train_method:
        weights_file = os.path.join(attack_weights_dir, f'{model_name}_{data_name}_{adv_train_method}_weights.pth')
    else:
        weights_file = os.path.join(weights_dir, f'{model_name}_{data_name}_weights.pth')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'best_acc': best_acc,
        'final_acc': test_acc,
        'total_time': avg_time,
        'avg_memory': avg_mem
    }, weights_file)


    training_history['final_results'] = {
        'best_accuracy': round(best_acc, 2),
        'final_accuracy': round(test_acc, 2),
        'avg_training_time': round(avg_time, 2),
        'avg_memory_usage': round(avg_mem, 2)
    }


    myutil.write_yaml(history_file, training_history)

    logger.info(f'Final model saved at: {final_model_path} avg mem: {avg_mem:.2f} avg time: {avg_time:.2f}')
    logger.info(f'Weights saved at: {weights_file}')
    logger.info(f'Training history saved at: {history_file}')


if __name__ == '__main__':
    dataset = 'cifar10'
    model = 'resnet18'
    file = 'spp'
    method = f'{file}#5'

    parser = argparse.ArgumentParser(description='Train model with adversarial methods')

    parser.add_argument('--data', help='cifar10,mnist,tiny-imagenet,cifar100', default=dataset)
    parser.add_argument('--model', help='vit_b_16,resnet18,vgg16,resnet34,densenet121,resnet50,wide_resnet101_2,wide_resnet50_2', default=model,
                        choices=['vit_b_16','vit_l_16', 'vgg16','vgg19','densenet169', 'densenet121', 'resnet18','resnet34','resnet50','wide_resnet101_2','wide_resnet50_2'])
    parser.add_argument('--attack', help='pgd#5,fisher#5,tpgd#5,fgsm,mifgsm#5,pifgsm#5,vnifgsm#5', default=method)
    parser.add_argument('--logname', help='log name', default='')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--results_dir', type=str, default=f'./results/{file}/', help='directory to save results')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')

    args = parser.parse_args()

    myutil.init_logger(args.logname)
    myutil.set_seed(42)

    main(args.model, args.data, args.attack, args.epochs, args.batch_size, args.results_dir, args.lr)
