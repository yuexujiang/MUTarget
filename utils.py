
import torch
torch.manual_seed(0)
import os
import numpy as np
from box import Box
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import shutil
from pathlib import Path
import datetime
from torch.utils.tensorboard import SummaryWriter

def create_mask_tensor(input,max_len):
    """
    input is a tuple of all the frag sequence
    """
    # Initialize a mask tensor of zeros with shape (len(A), max_len)
    mask_tensor = torch.zeros((len(input), max_len), dtype=torch.bool)
    
    # Fill the mask tensor with 1s for the length of each string
    for i, s in enumerate(input):
        mask_tensor[i, :len(s)] = True
    
    return mask_tensor

def get_class_id_dict(samples) -> dict:
    class_id = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]} #key is class, id is sample id
    
    protein_index=0
    for id, id_frag_list, seq_frag_list, target_frag_list, type_protein in samples:
        for label,item in enumerate(type_protein):
              if item == 1:
                 if label not in class_id:
                     class_id[label]=[protein_index]
                 else:
                     class_id[label].append(protein_index)
        
        protein_index+=1
    
    return class_id

def binary2label(target):
    """
    target has shape [batchsize,num_classes,sequence_len]
    """
    return target.argmax(dim=1)


def prepare_tensorboard(result_path):
    train_path = os.path.join(result_path, 'train')
    val_path = os.path.join(result_path, 'val')
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(val_path).mkdir(parents=True, exist_ok=True)

    train_log_path = os.path.join(train_path, 'tensorboard')
    train_writer = SummaryWriter(train_log_path)

    val_log_path = os.path.join(val_path, 'tensorboard')
    val_writer = SummaryWriter(val_log_path)

    return train_writer, val_writer

#{0: 0.015360983102918587, 1: 0.3333333333333333, 2: 0.7692307692307693, 3: 0.0625, 4: 0.015360983102918587, 5: 0.078125, 6: 0.4, 7: 0.125, 8: 1.0}
def calculate_class_weights(class_counts):
    """
    Calculate the weights for each class based on the number of samples.
    :param samples_dict: Dictionary containing classes as keys and number of samples as values.
    :return: Dictionary containing classes as keys and their respective weights as values.
    """
    if (class_counts[0]==0):
       min_samples = min(class_counts[1:])
       class_counts[0] = max(class_counts[1:]) #fix div zero
    else:
       min_samples = min(class_counts)
    
    weights_dict = {}
    for i in range(len(class_counts)):
        weights_dict[i] = min_samples / class_counts[i]
    return weights_dict


#residue_class_weights
#{0: 0.008122743682310469, 1: 0.3103448275862069, 2: 0.6923076923076923, 3: 0.05660377358490566, 4: 0.014018691588785047, 5: 0.0703125, 6: 0.3333333333333333, 7: 0.1232876712328767, 8: 1.0}

def calculate_residue_class_weights(n, samples):
    """
    n is the num_classes
    """
    residue_counts=np.zeros(n)
    for id, id_frag_list, seq_frag_list, target_frag_list, type_protein in samples:
        for target_frag in target_frag_list:
            inds = np.argmax(target_frag, axis=0)
            residue_counts[inds]+=1

    if (residue_counts[0]==0):
       min_samples = min(residue_counts[1:])
       residue_counts[0] = max(residue_counts[1:])
    else:
       min_samples = min(residue_counts)
    
    #weights_dict = {}
    weights = torch.zeros(n, dtype=torch.float)
    for i in range(len(residue_counts)):
        weights[i] = min_samples / residue_counts[i]
    
    return weights


def load_configs(config,args=None):
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
    # overwrite parameters if set through commandline
    class defaultObject:
      def __init__(self):
        self.enable = False  # Initialize enable attribute to False by default
   
    if args is not None:
        if args.result_path:
            tree_config.result_path = args.result_path
        try:
           if args.resume:
              tree_config.resume.resume = True
        except:
            pass
        
        try:
            if args.restart_optimizer:
                configs.resume.restart_optimizer = True
            else:
                configs.resume.restart_optimizer = False
        except:
            pass
        if args.resume_path:
            tree_config.resume.resume_path = args.resume_path
            # tree_config.resume.resume = True  # resume model to train - Yichuan
            # print("load model")

            # print(tree_config.resume.resume_path)
            # exit(0)
    
    #set configs value to default if doesn't have the attr
    if not hasattr(tree_config.train_settings, "data_aug"):
        tree_config.train_settings.data_aug = defaultObject()
        tree_config.train_settings.data_aug.enable = False
        tree_config.train_settings.data_aug.per_times = 1
        tree_config.train_settings.data_aug.pos_mutation_rate = 0
        tree_config.train_settings.data_aug.neg_mutation_rate = 0
        tree_config.train_settings.data_aug.add_original = True
    
    if not hasattr(tree_config.train_settings, "MLM"):
        tree_config.train_settings.MLM = defaultObject()
        tree_config.train_settings.MLM.enable = False
        tree_config.train_settings.MLM.mask_ratio = 0.2
    
    
    if not hasattr(tree_config,"decoder"):
           tree_config.decoder = None
           tree_config.decoder.type = "linear"
           tree_config.decoder.combine = False

    tree_config.train_settings.data_aug.pos_mutation_rate = float(tree_config.train_settings.data_aug.pos_mutation_rate)
    tree_config.train_settings.data_aug.neg_mutation_rate = float(tree_config.train_settings.data_aug.neg_mutation_rate)
    #
    return tree_config

def prepare_saving_dir(configs,config_file_path):
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
    shutil.copy(config_file_path, result_path)
    # Return the path to the result directory.
    return curdir_path, result_path, checkpoint_path, logfilepath

def prepare_optimizer(net, configs, num_train_samples, logfilepath):
    optimizer, scheduler = load_opt(net, configs)
    if configs.optimizer.mode == "skip":
        scheduler = None
    else:
        if scheduler is None:
            if configs.optimizer.decay.first_cycle_steps:
                first_cycle_steps = configs.optimizer.decay.first_cycle_steps
            else:
                first_cycle_steps=np.ceil(
                    num_train_samples / configs.train_settings.grad_accumulation) * configs.train_settings.num_epochs / configs.optimizer.decay.num_restarts
            print("first_cycle_steps="+str(first_cycle_steps))
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=first_cycle_steps,
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
    elif config.optimizer.name.lower() == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=float(config.optimizer.lr),
                              momentum=0.9, dampening=0,
                              weight_decay=float(config.optimizer.weight_decay))
    
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
    # print('load check point')
    # print(configs.resume.resume)
    # exit(0)
    if configs.resume.resume:
        model_checkpoint = torch.load(configs.resume.resume_path, map_location='cpu')
        print(f"load checkpoint from {configs.resume.resume_path}")
        customlog(logfilepath, f"load checkpoint from {configs.resume.resume_path}")
        net.load_state_dict(model_checkpoint['model_state_dict'])
        # If the saved checkpoint contains the optimizer and scheduler states and the epoch number,
        # resume training from the last saved epoch.
        if 'optimizer_state_dict' in model_checkpoint and 'scheduler_state_dict' in model_checkpoint and 'epoch' in model_checkpoint:
            if not configs.resume.restart_optimizer:
                # 这段代码有bug, 一部分张量没有被放到cuda上
                optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
                # logging.info('Optimizer is loaded to resume training!')
                customlog(logfilepath, "Optimizer is loaded to resume training!\n")
                print("Optimizer is loaded to resume training!\n")
                scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
                # logging.info('Scheduler is loaded to resume training!')
                customlog(logfilepath, "Scheduler is loaded to resume training!\n")
                print("Scheduler is loaded to resume training!\n")
                start_epoch = model_checkpoint['epoch'] + 1
        
        customlog(logfilepath, "Model is loaded to resume training!\n")
    # Return the loaded model and the epoch to start training from.

        if configs.resume.frozen_esm:  # yichuan 0529
            print('load frozen esm')
            for param in net.model.parameters():
                param.requires_grad = False

    return net, start_epoch


def load_checkpoints_only(encoder,model_path,num_classes = 9):
    model_checkpoint = torch.load(model_path, map_location='cpu')
    encoder.model.load_state_dict(model_checkpoint['shared_model'])
    for class_index in range(1,num_classes):    
        encoder.ParallelDecoders.decoders[class_index-1].load_state_dict(model_checkpoint['task_'+str(class_index)])
    

def load_checkpoints_only_seperate(encoder,model_path,class_index):
    model_checkpoint = torch.load(model_path, map_location='cpu')
    encoder.model.load_state_dict(model_checkpoint['shared_model'])
    encoder.ParallelDecoders.decoders[class_index-1].load_state_dict(model_checkpoint['task_'+str(class_index)])
    return encoder
    


def resume_allseperate_checkpoints(configs,optimizer,scheduler,logfilepath,checkpoint_path,encoder):
    start_epoch = 1
    #will load the shared model at the first time
    model_path = configs.resume.resume_path
    for class_index in range(1,configs.encoder.num_classes):
         if class_index!=0:
            shutil.copy(model_path,os.path.join(checkpoint_path, f'best_model_{class_index}.pth'))
         
         #epoch_num=load_seperate_checkpoints(configs,optimizer,scheduler,logfilepath,encoder,
         #          class_index,model_path=model_path,load_shared=class_index==0,restart_optimizer=configs.resume.restart_optimizer)
    
    return load_all_checkpoints(configs,optimizer,scheduler,logfilepath,encoder,model_path)
    #return encoder, start_epoch


def resume_seperate_checkpoints(configs,optimizer,scheduler,logfilepath,checkpoint_path,encoder):
    start_epoch = 1
    #will load the shared model at the first time
    model_path = configs.resume.resume_path
    last_epoch = 0
    last_task_index=None
    best_valid_loss= np.full(encoder.num_classes,0).tolist()
    counter_task = np.full(encoder.num_classes,0).tolist()
    #just use this to detect the last_epoch and do copy best models to the target paths
    for class_index in range(1,configs.encoder.num_classes):
         if class_index!=0:
            model_path = configs.resume.resume_path+f'_{class_index}.pth'
            target_path = os.path.join(checkpoint_path, f'best_model_{class_index}.pth')
            customlog(logfilepath, f"Copy from {model_path} to {target_path}\n")
            shutil.copy(model_path,target_path)
            model_checkpoint = torch.load(model_path)#, map_location='cpu')
            if last_epoch < model_checkpoint['epoch']:
                last_epoch = model_checkpoint['epoch']
                last_task_index = class_index
            
            # Load environment variables
            best_valid_loss[class_index] = model_checkpoint['best_valid_loss']
            counter_task[class_index] = model_checkpoint['counter_task']
    
    
    model_path = configs.resume.resume_path+f'_{last_task_index}.pth'
    model_checkpoint = torch.load(model_path)#, map_location='cpu')
    customlog(logfilepath, f"Loading checkpoint from {model_path} for all tasks\n")
    encoder.model.load_state_dict(model_checkpoint['shared_model'])
    optimizer['shared'].load_state_dict(model_checkpoint['shared_optimizer'])
    scheduler['shared'].load_state_dict(model_checkpoint['shared_scheduler'])
    customlog(logfilepath, f"Shared optimizer is loaded to resume training!\n")
    
    # Move optimizer states to the correct device
    for state in optimizer['shared'].state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(configs.train_settings.device)
    
    for class_index in range(1,configs.encoder.num_classes):
        encoder.ParallelDecoders.decoders[class_index-1].load_state_dict(model_checkpoint['task_'+str(class_index)])
        optimizer[class_index].load_state_dict(model_checkpoint['optimizer_'+str(class_index)])
        customlog(logfilepath, f"Optimizer {class_index} is loaded to resume training!\n")
        scheduler[class_index].load_state_dict(model_checkpoint['scheduler_'+str(class_index)])
        customlog(logfilepath, "Scheduler is loaded to resume training!\n")

        # Move optimizer and scheduler states to the correct device for each task
        for state in optimizer[class_index].state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(configs.train_settings.device)
    
    return encoder, optimizer, scheduler, last_epoch, best_valid_loss, counter_task



def load_all_checkpoints(configs, optimizer, scheduler, logfilepath, encoder, model_path):
    start_epoch = 1
    customlog(logfilepath, f"Loading checkpoint from {model_path} for all tasks\n")
    model_checkpoint = torch.load(model_path)#, map_location='cpu')
    encoder.model.load_state_dict(model_checkpoint['shared_model'])
    customlog(logfilepath, f"Loading shared parts\n")
    
    optimizer['shared'].load_state_dict(model_checkpoint['shared_optimizer'])
    scheduler['shared'].load_state_dict(model_checkpoint['shared_scheduler'])
    customlog(logfilepath, f"Shared optimizer is loaded to resume training!\n")
    
    # Move optimizer states to the correct device
    for state in optimizer['shared'].state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(configs.train_settings.device)
    
    # Load environment variables
    start_epoch = model_checkpoint['epoch']
    best_valid_loss = model_checkpoint['best_valid_loss']
    counter_task = model_checkpoint['counter_task']
    stop_task = model_checkpoint['stop_task']
    for class_index in range(1,configs.encoder.num_classes):
        encoder.ParallelDecoders.decoders[class_index-1].load_state_dict(model_checkpoint['task_'+str(class_index)])
        optimizer[class_index].load_state_dict(model_checkpoint['optimizer_'+str(class_index)])
        customlog(logfilepath, f"Optimizer {class_index} is loaded to resume training!\n")
        scheduler[class_index].load_state_dict(model_checkpoint['scheduler_'+str(class_index)])
        customlog(logfilepath, "Scheduler is loaded to resume training!\n")

        # Move optimizer and scheduler states to the correct device for each task
        for state in optimizer[class_index].state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(configs.train_settings.device)
    
    return encoder, optimizer, scheduler, start_epoch, best_valid_loss, counter_task,stop_task



def save_all_checkpoint(epoch: int, model_path: str, tools: dict,best_valid_loss,counter_task,stop_task):
    """
    Save all the model checkpoints during training
    Each time load all the saved checkpoints 
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
    try:
        checkpoint = torch.load(model_path)
    except:
        checkpoint = {} #if there is no model_path
    
    #all the environment variables need to be saved
    checkpoint['best_valid_loss'] = best_valid_loss
    checkpoint['counter_task'] = counter_task
    checkpoint['stop_task'] = stop_task
    checkpoint['epoch'] = epoch
    
    checkpoint['shared_model'] = tools['net'].model.state_dict()
    checkpoint['shared_optimizer'] = tools['optimizer_task']['shared'].state_dict()
    checkpoint['shared_scheduler'] = tools['scheduler_task']['shared'].state_dict()
    
    for class_index in range(1,tools['num_classes']):
        #update the checkpoint for class_index and related shared model's checkpoints
        checkpoint['task_'+str(class_index)] = tools['net'].ParallelDecoders.decoders[class_index-1].state_dict()
        checkpoint['optimizer_'+str(class_index)] = tools['optimizer_task'][class_index].state_dict()
        checkpoint['scheduler_'+str(class_index)] = tools['scheduler_task'][class_index].state_dict()
    
    torch.save(checkpoint, model_path)

def save_seperate_checkpoint(epoch, best_valid_loss,counter_task,model_path, tools,class_index):
    checkpoint = {}
    checkpoint['best_valid_loss'] = best_valid_loss
    checkpoint['counter_task'] = counter_task
    checkpoint['epoch'] = epoch
    
    checkpoint['shared_model'] = tools['net'].model.state_dict()
    checkpoint['shared_optimizer'] = tools['optimizer_task']['shared'].state_dict()
    checkpoint['shared_scheduler'] = tools['scheduler_task']['shared'].state_dict()
    
    #update the checkpoint for class_index and related shared model's checkpoints
    checkpoint['task_'+str(class_index)] = tools['net'].ParallelDecoders.decoders[class_index-1].state_dict()
    checkpoint['optimizer_'+str(class_index)] = tools['optimizer_task'][class_index].state_dict()
    checkpoint['scheduler_'+str(class_index)] = tools['scheduler_task'][class_index].state_dict()
    
    torch.save(checkpoint, model_path)



def customlog(filepath, text):
    logfile=open(filepath, "a")
    logfile.write(text)
    logfile.write("\n")
    logfile.close()




def transform_tuple(input_tuple):
    """
    
# Example placeholders for pid values
pid0_0, pid0_1, pid1_0, pid2_0, pid3_0 = "pid0_0", "pid0_1", "pid1_0", "pid2_0", "pid3_0"

# Example usage with a tuple of lists of lists
input_tuple_1 = ([[pid0_0, pid0_1], [pid1_0]], [[pid2_0], [pid3_0]])
print(transform_tuple(input_tuple_1))

# Example usage with a tuple of lists
input_tuple_2 = ([pid0_0, pid0_1], [pid1_0], [[pid2_0], [pid3_0]])
print(transform_tuple(input_tuple_2))

    """
    # Initialize the result list
    result_list = []
    
    # Iterate over each sublist in the input tuple
    for sublist in input_tuple:
        # Check if the sublist itself is a list of lists
        if all(isinstance(item, list) for item in sublist):
            for item in sublist:
                # If the item is a list with more than one element, repeat it
                if len(item) > 1:
                    repeated_items = [item for _ in sublist]
                    result_list.extend(repeated_items)
                else:
                    result_list.append(item)
        else:
            # Handle the case where the sublist is just a list of elements
            if len(sublist) > 1:
                repeated_items = [sublist for _ in input_tuple]
                result_list.extend(repeated_items)
            else:
                result_list.append(sublist)
    
    return result_list

def get_segments(arr):
    """
    Given a 1D binary array arr (0/1), return a list of (start, end) pairs
    for each continuous run of 1's. 'start' and 'end' are inclusive indices.
    Example:
        arr = [0,1,1,1,0,0,1,0]
        -> returns [(1,3), (6,6)]
    """
    segments = []
    in_segment = False
    seg_start = None
    
    for i, val in enumerate(arr):
        if val == 1 and not in_segment:
            in_segment = True
            seg_start = i
        elif val == 0 and in_segment:
            in_segment = False
            segments.append((seg_start, i - 1))
    # If ended in a segment
    if in_segment and seg_start is not None:
        segments.append((seg_start, len(arr) - 1))
    
    return segments

def segment_length(start, end):
    """ Length of a segment with inclusive boundaries. """
    return end - start + 1

def overlap_and_union(seg1, seg2):
    """
    Given two segments seg1=(s1,e1), seg2=(s2,e2), both inclusive:
    - overlap_len = number of positions in their intersection
    - union_len   = number of positions in their union
    If they don't overlap, overlap_len = 0, union_len = sum of lengths.
    """
    s1, e1 = seg1
    s2, e2 = seg2

    overlap_start = max(s1, s2)
    overlap_end   = min(e1, e2)
    if overlap_start > overlap_end:
        # no overlap
        ov_len = 0
    else:
        ov_len = overlap_end - overlap_start + 1

    union_start = min(s1, s2)
    union_end   = max(e1, e2)
    union_len   = union_end - union_start + 1
    
    return ov_len, union_len

def sov_score(y_true, y_pred):
    """
    Compute a simplified SOV score for the predicted segments vs. ground truth.
    
    SOV formula (roughly):
      SOV = 100 * sum_over_i( Delta_i ) / sum_over_i( length_of_groundtruth_segment_i )
    
    where
      Delta_i = min( OV(i), alpha_i * UNION(i) )
      OV(i)   = overlap length between ground-truth seg i and its predicted overlap
      UNION(i)= union length between ground-truth seg i and predicted overlap
      alpha_i = 1 - (|len(gt_seg_i) - len(predicted_overlap)| / UNION(i))
    
    We combine all predicted segments that overlap the i-th ground-truth segment
    into a single "predicted overlap segment" by taking their union, for simplicity.
    """
    # 1) Identify ground-truth segments and predicted segments
    gt_segments = get_segments(y_true)
    pred_segments = get_segments(y_pred)
    
    # 2) For each ground-truth segment, find the union of all predicted segments that overlap it
    total_length_gt = 0.0  # sum of lengths of all ground-truth segments
    total_delta = 0.0      # sum of Delta_i across ground-truth segments
    
    for gt_seg in gt_segments:
        gt_start, gt_end = gt_seg
        gt_len = segment_length(gt_start, gt_end)
        total_length_gt += gt_len
        
        # Collect all predicted segments that have any overlap with gt_seg
        overlapping_pred_segments = []
        for ps in pred_segments:
            # Check if there's any overlap
            ov_len, _ = overlap_and_union(gt_seg, ps)
            if ov_len > 0:
                overlapping_pred_segments.append(ps)
        
        if not overlapping_pred_segments:
            # No overlap at all with this ground-truth segment => Delta_i = 0
            continue
        
        # Merge overlapping predicted segments into one "combined" segment
        # to compare against the single ground-truth segment.
        # This is a simplified approach that merges all predicted segments that overlap gt_seg.
        merged_pred_start = min(ps[0] for ps in overlapping_pred_segments)
        merged_pred_end   = max(ps[1] for ps in overlapping_pred_segments)
        merged_pred_seg   = (merged_pred_start, merged_pred_end)
        pred_len = segment_length(merged_pred_start, merged_pred_end)
        
        # 3) Compute overlap (OV), union, alpha, Delta
        ov_len, uni_len = overlap_and_union(gt_seg, merged_pred_seg)
        
        if uni_len == 0:
            # No union => no overlap => Delta=0
            continue
        
        length_diff = abs(gt_len - pred_len)
        alpha = 1.0 - float(length_diff) / float(uni_len)
        
        # Delta_i = min( OV(i), alpha_i * UNION(i) )
        delta_i = min(ov_len, alpha * uni_len)
        total_delta += delta_i
    
    if total_length_gt == 0:
        return 100.0  # or 0.0, depending on convention. If no ground-truth segments exist, define a default.
    
    # Final SOV
    return 100.0 * (total_delta / total_length_gt)