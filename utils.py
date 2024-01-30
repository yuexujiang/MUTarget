
import torch
torch.manual_seed(0)
import os
import numpy as np
from box import Box
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import shutil
from pathlib import Path
import datetime


def calculate_class_weights(class_counts):
    """
    Calculate the weights for each class based on the number of samples.
    :param samples_dict: Dictionary containing classes as keys and number of samples as values.
    :return: Dictionary containing classes as keys and their respective weights as values.
    """
    min_samples = min(class_counts)
    weights_dict = {}
    for i in range(len(class_counts)):
        weights_dict[i] = min_samples / class_counts[i]
    return weights_dict

def load_configs(config):
    """
        Load the configuration file and convert the necessary values to floats.
        Args:
            config (dict): The configuration dictionary.
        Returns:
            The updated configuration dictionary with float values.
        """
    # Convert the dictionary to a Box object for easier access to the values.
    tree_config = Box(config)
    # Convert the necessary values to floats.
    tree_config.optimizer.lr = float(tree_config.optimizer.lr)
    tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    return tree_config

def prepare_saving_dir(configs):
    """
    Prepare a directory for saving a training results.
    Args:
        configs: A python box object containing the configuration options.
    Returns:
        str: The path to the directory where the results will be saved.
    """
    # Create a unique identifier for the run based on the current time.
    run_id = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    # Add '_evaluation' to the run_id if the 'evaluate' flag is True.
    # if configs.evaluate:
    #     run_id += '_evaluation'
    # Create the result directory and the checkpoint subdirectory.
    # curfile_path=os.path.realpath(__file__)
    # curdir_path=os.path.dirname(curfile_path)
    curdir_path=os.getcwd()
    result_path = os.path.abspath(os.path.join(configs.result_path, run_id))
    checkpoint_path = os.path.join(result_path, 'checkpoints')
    logfilepath = os.path.join(result_path, "loginfo.log")
    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    # Copy the config file to the result directory.
    shutil.copy(configs.config_path, result_path)
    # Return the path to the result directory.
    return curdir_path, result_path, checkpoint_path, logfilepath

def prepare_optimizer(net, configs, num_train_samples, logfilepath):
    optimizer, scheduler = load_opt(net, configs)
    if configs.optimizer.mode == "skip":
        scheduler = None
    else:
        if scheduler is None:
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=np.ceil(
                    num_train_samples / configs.train_settings.grad_accumulation) * configs.train_settings.num_epochs / configs.optimizer.decay.num_restarts,
                cycle_mult=1.0,
                max_lr=configs.optimizer.lr,
                min_lr=configs.optimizer.decay.min_lr,
                warmup_steps=configs.optimizer.decay.warmup,
                gamma=configs.optimizer.decay.gamma)
    return optimizer, scheduler


def load_opt(model, config):
    scheduler = None
    if config.optimizer.name.lower() == 'adam':
        # opt = eval('torch.optim.' + config.optimizer.name)(model.parameters(), lr=config.optimizer.lr, eps=eps,
        #                                       weight_decay=config.optimizer.weight_decay)
        opt = torch.optim.AdamW(
        model.parameters(), lr=float(config.optimizer.lr),
        betas=(config.optimizer.beta_1, config.optimizer.beta_2),
        weight_decay=float(config.optimizer.weight_decay),
        eps=float(config.optimizer.eps)
        )
    else:
        raise ValueError('wrong optimizer')
    return opt, scheduler

def save_checkpoint(epoch: int, model_path: str, tools: dict):
    """
    Save the model checkpoints during training.
    Args:
        epoch (int): The current epoch number.
        model_path (str): The path to save the model checkpoint.
        tools (dict): A dictionary containing the necessary tools for saving the model checkpoints.
        accelerator (Accelerator): Accelerator object.
    Returns:
        None
    """
    # # Set the path to save the model checkpoint.
    # model_path = os.path.join(tools['result_path'], 'checkpoints', f'checkpoint_{epoch}.pth')
    # Save the model checkpoint.
    torch.save({
        'epoch': epoch,
        'model_state_dict': tools['net'].state_dict(),
        'optimizer_state_dict': tools['optimizer'].state_dict(),
        'scheduler_state_dict': tools['scheduler'].state_dict(),
    }, model_path)

def load_checkpoints(configs, optimizer, scheduler, logfilepath, net):
    """
    Load saved checkpoints from a previous training session.

    Args:
        configs: A python box object containing the configuration options.
        optimizer (Optimizer): The optimizer to resume training with.
        scheduler (Scheduler): The learning rate scheduler to resume training with.
        logging (Logger): The logger to use for logging messages.
        net (nn.Module): The neural network model to load the saved checkpoints into.

    Returns:
        tuple: A tuple containing the loaded neural network model and the epoch to start training from.
    """
    start_epoch = 1
    # If the 'resume' flag is True, load the saved model checkpoints.
    if configs.resume.resume:
        model_checkpoint = torch.load(configs.resume.resume_path, map_location='cpu')
        net.load_state_dict(model_checkpoint['model_state_dict'])
        # If the saved checkpoint contains the optimizer and scheduler states and the epoch number,
        # resume training from the last saved epoch.
        if 'optimizer_state_dict' in model_checkpoint and 'scheduler_state_dict' in model_checkpoint and 'epoch' in model_checkpoint:
            if not configs.resume.restart_optimizer:
                optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
                # logging.info('Optimizer is loaded to resume training!')
                customlog(logfilepath, "Optimizer is loaded to resume training!\n")
                scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
                # logging.info('Scheduler is loaded to resume training!')
                customlog(logfilepath, "Scheduler is loaded to resume training!\n")
            start_epoch = model_checkpoint['epoch'] + 1
        customlog(logfilepath, "Model is loaded to resume training!\n")
    # Return the loaded model and the epoch to start training from.
    return net, start_epoch

def customlog(filepath, text):
    logfile=open(filepath, "a")
    logfile.write(text)
    logfile.write("\n")
    logfile.close()




