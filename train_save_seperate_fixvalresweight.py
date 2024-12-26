import torch
torch.manual_seed(0)
from torch.cuda.amp import GradScaler, autocast
import argparse
import os
import yaml
import numpy as np
# import torchmetrics
from time import time
from model import *
from utils import *
from sklearn.metrics import matthews_corrcoef,recall_score,precision_score,f1_score
import pandas as pd
import sys
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
import shutil
from utils import prepare_tensorboard,binary2label,create_mask_tensor
from data_batchsample import prepare_dataloaders as prepare_dataloader_batchsample

# torch.autograd.set_detect_anomaly(True)
from collections import OrderedDict

torch.autograd.set_detect_anomaly(True)

label2idx = OrderedDict([
    ("Other", 0),
    ("ER", 1),
    ("Peroxisome", 2),
    ("Mitochondrion", 3),
    ("SIGNAL", 4),
    ("Nucleus", 5),
    ("Nucleus_export", 6),
    ("chloroplast", 7),
    ("Thylakoid", 8)
])
filter_list = [
        'Q9LPZ4', 'P15330', 'P35869', 'P70278', 'Q80UP3',
        'Q8LH59', 'P19484', 'P35123', 'Q6NVF4', 'Q8NG08', 'Q9BVS4', 'Q9NRA0', 'Q9NUL5', 'Q9UBP0', 'P78953',
        'A8MR65', 'Q8S4Q6', 'Q3U0V2', 'Q96D46', 'Q9NYA1', 'Q9ULX6', 'Q9WTL8',
        'P35922', 'P46934', 'P81299', 'Q13148', 'Q6ICB0', 'Q7TPV4', 'Q8N884', 'Q99LG4', 'Q9Z207',
        'O00571', 'P52306', 'Q13015', 'Q13568', 'Q5TAQ9', 'Q8NAG6', 'Q9BZ23', 'Q9BZS1',
    ]

def loss_fix(id_frag, motif_logits, target_frag, tools):
    #id_frag [batch]
    #motif_logits [batch, num_clas, seq]
    #target_frag [batch, num_clas, seq]
    fixed_loss = 0
    for i in range(len(id_frag)):
        frag_ind = id_frag[i].split('@')[1]
        target_thylakoid = target_frag[i, label2idx['Thylakoid']]  # -1 for Thylakoid, [seq]; -2 for chloroplast
        # label_first = target_thylakoid[0] # 1 or 0
        target_chlo = target_frag[i, label2idx['chloroplast']]
        #for the case "thylakoid" but no known signal for "chloroplast"
        if frag_ind == '0' and torch.max(target_chlo) == 0 and torch.max(target_thylakoid) == 1:
            # print("case2")
            l = torch.where(target_thylakoid == 1)[0][0]
            #true_chlo = target_frag[i, label2idx['chloroplast'], :(l-1)] == 1
            false_chlo = target_chlo[:(l-1)] == 0
            motif_logits[i, label2idx['chloroplast'], :(l-1)][false_chlo] = -10000
    # return fixed_loss
    # return target_frag
    return motif_logits, target_frag



def make_buffer(id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple):
    id_frags_list = []
    seq_frag_list = []
    target_frag_list = []
    for i in range(len(id_frag_list_tuple)):
        id_frags_list.extend(id_frag_list_tuple[i])
        seq_frag_list.extend(seq_frag_list_tuple[i])
        target_frag_list.extend(target_frag_nplist_tuple[i])
    seq_frag_tuple = tuple(seq_frag_list)
    target_frag_pt = torch.from_numpy(np.stack(target_frag_list, axis=0))
    type_protein_pt = torch.stack(list(type_protein_pt_tuple), axis=0)
    return id_frags_list, seq_frag_tuple, target_frag_pt, type_protein_pt


def train_loop(tools, configs, train_writer, stop_task):
    global global_step
    tools["optimizer_task"]['shared'].zero_grad()
    for class_index in range(1,configs.encoder.num_classes):
            tools["optimizer_task"][class_index].zero_grad()
    
    scaler = GradScaler()
    size = len(tools['train_loader'].dataset)
    num_batches = len(tools['train_loader'])
    print("size="+str(size)+" num_batches="+str(num_batches))
    train_loss = 0
    # cs_num=np.zeros(9)
    # cs_correct=np.zeros(9)
    # type_num=np.zeros(10)
    # type_correct=np.zeros(10)
    # cutoff=0.5
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.train().cuda()
    tools['net'].train().to(tools['train_device'])
    for batch, (id_tuple, id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple, sample_weight_tuple,residue_class_weights) in enumerate(tools['train_loader']):
        b_size = len(id_tuple)
        flag_batch_extension = False
        
        id_frags_list, seq_frag_tuple, target_frag_pt, type_protein_pt = make_buffer(id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple)
        print("num of frag: "+str(len(id_frags_list)))
        mask_seq = create_mask_tensor(seq_frag_tuple,configs.encoder.max_len-2).to(tools['train_device'])
        with autocast():
            # Compute prediction and loss
            encoded_seq=tokenize(tools, seq_frag_tuple)

            if type(encoded_seq)==dict:
                for k in encoded_seq.keys():
                    encoded_seq[k]=encoded_seq[k].to(tools['train_device'])
            else:
                encoded_seq=encoded_seq.to(tools['train_device'])
            
            classification_head, motif_logits = tools['net'](
                                 encoded_seq, 
                                 id_tuple, 
                                 id_frags_list, 
                                 seq_frag_tuple)
            weighted_loss_sum = 0
            class_loss = -1
            position_loss=-1
            motif_logits, target_frag = loss_fix(id_frags_list, motif_logits, target_frag_pt, tools)
            sample_weight_pt = torch.from_numpy(np.array(sample_weight_tuple)).to(tools['train_device']).unsqueeze(1)

            true_target_frag = binary2label(target_frag) #[batchsize,sequence_len]
            if not configs.train_settings.add_sample_weight_to_position_loss:
               position_loss = torch.mean(tools['loss_function'](motif_logits.permute(0,2,1)[mask_seq], true_target_frag.to(tools['train_device'])[mask_seq]))
            
            # yichuan: 因为BCEloss已经修改成了none, 所以这里必须要加mean
            #residue_class_weights is a batch of duplicated residue_class_weights, so the residue_class_weights[0] is the real residue_class_weights in class LocalizationDataset
            true_target_frag_int = true_target_frag.long()
            position_weight_pt = residue_class_weights[0][true_target_frag_int].to(tools['train_device']) #torch.Size([18, 1022]) 18 > batchsize, if a protein is longer than 1022
            if configs.train_settings.add_sample_weight_to_position_loss:
                position_loss = torch.mean(tools['loss_function'](motif_logits.permute(0,2,1)[mask_seq], 
                                true_target_frag.to(tools['train_device'])[mask_seq])*position_weight_pt[mask_seq])
            
            if torch.isnan(classification_head).any(): 
                      print(classification_head)
                      print("NaN detected in classification_head")
            if torch.isinf(classification_head).any():
                      print("Inf detected in classification_head")
            if torch.isnan(type_protein_pt).any() or torch.isinf(type_protein_pt).any():
                      print("NaN or Inf detected in type_protein_pt")
            
            if configs.train_settings.data_aug.enable:
                # class_loss = torch.mean(tools['loss_function_pro'](classification_head, type_protein_pt.to(tools['train_device']))) #remove sample_weight_pt
                # class_loss = torch.mean(tools['loss_function_pro'](classification_head, type_protein_pt.to(tools['train_device'])) * sample_weight_pt)  # - yichuan 0526
                #classification_head #batch,9 # keep other!
                #print(sample_weight_pt)
                if configs.train_settings.add_sample_weight_to_class_loss_when_data_aug:
                    if configs.train_settings.train_9_classes.enable:
                        class_loss = torch.mean(tools['binary_loss'](classification_head, type_protein_pt.to(
                                     tools['valid_device'])) * sample_weight_pt)
                    else:
                        class_loss = torch.mean(tools['binary_loss'](classification_head[:,1:], type_protein_pt[:,1:].to(
                                     tools['valid_device'])) * sample_weight_pt)
                else:
                    if configs.train_settings.train_9_classes.enable:
                        class_loss = torch.mean(
                                     tools['binary_loss'](classification_head, type_protein_pt.to(tools['valid_device'])))
                    else:
                        class_loss = torch.mean(
                                     tools['binary_loss'](classification_head[:,1:], type_protein_pt[:,1:].to(tools['valid_device'])))
                        
            else:
                if configs.train_settings.add_sample_weight_to_class_loss_when_data_aug:
                    if configs.train_settings.train_9_classes.enable:
                        class_loss = torch.mean(tools['binary_loss'](classification_head, type_protein_pt.to(
                                     tools['valid_device'])) * sample_weight_pt)
                    else:
                        class_loss = torch.mean(tools['binary_loss'](classification_head[:,1:], type_protein_pt[:,1:].to(
                                     tools['valid_device'])) * sample_weight_pt)
                else:
                    if configs.train_settings.train_9_classes.enable:
                        class_loss = torch.mean(
                                     tools['binary_loss'](classification_head, type_protein_pt.to(tools['valid_device'])))
                    else:
                        class_loss = torch.mean(
                                     tools['binary_loss'](classification_head[:,1:], type_protein_pt[:,1:].to(tools['valid_device'])))

            train_writer.add_scalar('step class_loss', class_loss.item(), global_step=global_step)
            # train_writer.add_scalar('step position_loss', position_loss.item(), global_step=global_step)
            print(f"{global_step} class_loss:{class_loss.item()}  position_loss:{position_loss.item()}")
            # weighted_loss_sum = class_loss + position_loss
            # if epoch >= configs.train_settings.weighted_loss_sum_start_epoch:  # yichuan 0529
            #     weighted_loss_sum = class_loss * configs.train_settings.loss_sum_weights[0] + position_loss * configs.train_settings.loss_sum_weights[1]
            # else:
            # Simplified code for weighted loss calculation
            # position_loss_weighted = position_loss / configs.train_settings.position_loss_T if configs.train_settings.position_loss_T != 1 else position_loss
            # if configs.train_settings.only_use_position_loss:
            #     weighted_loss_sum = position_loss_weighted  # yichuan updated on 0610 and 0601
            # else:
            #     weighted_loss_sum = class_loss + position_loss_weighted  # yichuan updated on 0612
            # Determine the weighted position loss based on a configurable threshold.
            if configs.train_settings.position_loss_T != 1:
                position_loss_weighted = position_loss / configs.train_settings.position_loss_T
            else:
                position_loss_weighted = position_loss
            # Determine the weighted class loss based on a configurable threshold.
            if configs.train_settings.class_loss_T != 1:
                class_loss_weighted = class_loss / configs.train_settings.class_loss_T
            else:
                class_loss_weighted = class_loss
            # Calculate the weighted sum of losses.
            if configs.train_settings.only_use_position_loss:
                # If configured to only use position loss, ignore class loss.
                weighted_loss_sum = position_loss_weighted
            else:
                # Otherwise, sum class loss and weighted position loss.
                weighted_loss_sum = class_loss_weighted + position_loss_weighted


            train_loss += weighted_loss_sum.item()
        
        # Backpropagation
        torch.autograd.set_detect_anomaly(True)
        scaler.scale(weighted_loss_sum).backward()
        # Gradient clipping
        if configs.train_settings.clip_grad_norm !=0:
             #print("clip_grad_norm")
             torch.nn.utils.clip_grad_norm_(tools['net'].parameters(), max_norm=configs.train_settings.clip_grad_norm)
        
        for class_index in range(1,configs.encoder.num_classes): #skip 0 others
            if not stop_task[class_index]:
               scaler.step(tools['optimizer_task'][class_index])
        
        # Only update shared model if no task has stopped
        #if np.sum(stop_task)==0:
        #   scaler.step(tools['optimizer_task']['shared'])
        
        scaler.update()
        for class_index in range(1,configs.encoder.num_classes):
            if not stop_task[class_index]:
                    tools['scheduler_task'][class_index].step()
        
        if np.sum(stop_task)==0:
              tools['scheduler_task']['shared'].step()
              
        print(f"{global_step} loss:{weighted_loss_sum.item()}\n")
        train_writer.add_scalar('step loss', weighted_loss_sum.item(), global_step=global_step)
        train_writer.add_scalar('learning_rate', tools['scheduler_task']['shared'].get_lr()[0], global_step=global_step)
        #if global_step % configs.train_settings.log_every == 0: #30 before changed into 0
        if batch % configs.train_settings.log_every == 0: #for comparison with original codes
            loss, current = weighted_loss_sum.item(), (batch + 1) * b_size  # len(id_tuple)
            customlog(tools["logfilepath"], f"{global_step} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")
            if class_loss !=-1:
                if configs.train_settings.additional_pos_weights:  # yichuan 0529
                    # customlog(tools["logfilepath"],
                    #           f"{global_step} class loss: {class_loss.item():>7f} position_loss:{position_loss_sum.item():>7f}\n")
                    pass
                else:
                    customlog(tools["logfilepath"], f"{global_step} class loss: {class_loss.item():>7f} position_loss:{position_loss.item():>7f}\n")

        
        global_step+=1
    
    epoch_loss = train_loss/num_batches
    return epoch_loss



def test_loop(tools, dataloader,train_writer,valid_writer,configs):
    customlog(tools["logfilepath"], f'number of test steps per epoch: {len(dataloader)}\n')
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.eval().cuda()
    tools['net'].eval().to(tools["valid_device"])
    num_batches = len(dataloader)
    test_loss= torch.zeros(configs.encoder.num_classes,device = tools['valid_device'])
    test_class_loss=torch.zeros(configs.encoder.num_classes,device = tools['valid_device'])
    test_position_loss=torch.zeros(configs.encoder.num_classes,device = tools['valid_device'])
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    #print("in test loop")
    with torch.no_grad():
        for batch, (id_tuple, id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple, sample_weight_tuple, residue_class_weights) in enumerate(dataloader):
            id_frags_list, seq_frag_tuple, target_frag_pt, type_protein_pt = make_buffer(id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple)
            encoded_seq=tokenize(tools, seq_frag_tuple)
            if type(encoded_seq)==dict:
                for k in encoded_seq.keys():
                    encoded_seq[k]=encoded_seq[k].to(tools['valid_device'])
            else:
                encoded_seq=encoded_seq.to(tools['valid_device'])
            #print("ok1")
            mask_seq = create_mask_tensor(seq_frag_tuple,configs.encoder.max_len-2).to(tools['train_device'])
            classification_head, motif_logits= tools['net'](
                       encoded_seq,
                       id_tuple,id_frags_list,seq_frag_tuple) #for test_loop always used None and False!
            #print("ok2")
            weighted_loss_sum = 0
            #if not warm_starting:
            class_loss = 0
            position_loss=0
            motif_logits, target_frag = loss_fix(id_frags_list, motif_logits, target_frag_pt, tools)
            sample_weight_pt = torch.from_numpy(np.array(sample_weight_tuple)).to(tools['valid_device']).unsqueeze(1)
            
            #true_target_frag = binary2label(target_frag)
            true_target_frag = target_frag.permute(0,2,1) #this time used binary_loss,target_frag [B,9,l] so true_target_frag [B,L,9]
            if not configs.train_settings.add_sample_weight_to_position_loss:
                  #for seperate loss should be binary_loss
                  position_loss = torch.mean(tools['binary_loss'](motif_logits.permute(0,2,1)[mask_seq], true_target_frag.to(tools['train_device'])[mask_seq]),dim = 0) #[9]
            #
            true_target_frag_int = true_target_frag* torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
            true_target_frag_int = true_target_frag_int.long() #class 0 is not used.
            position_weight_pt = residue_class_weights[0][true_target_frag_int].to(tools['train_device'])#[residue_class_weights[x] for x in true_target_frag]
            #print(position_weight_pt.shape) #[19,1022]
            #print(position_weight_pt[mask_seq].shape) #[7957]
            #print("true_target_frag.shape")
            #print(true_target_frag.to(tools['train_device'])[mask_seq].shape) #torch.Size([7957, 9])
            # yichuan: 因为BCEloss已经修改成了none, 所以这里必须要加mean
            #print(tools['binary_loss'](motif_logits.permute(0,2,1)[mask_seq], 
            #                    true_target_frag.to(tools['train_device'])[mask_seq]).shape)
            if configs.train_settings.add_sample_weight_to_position_loss:
                position_loss = torch.mean(tools['binary_loss'](motif_logits.permute(0,2,1)[mask_seq], 
                                true_target_frag.to(tools['train_device'])[mask_seq])*position_weight_pt[mask_seq],dim = 0) #should be 8


            if configs.train_settings.data_aug.enable:
                # class_loss = torch.mean(tools['loss_function_pro'](classification_head, type_protein_pt.to(tools['valid_device'])))
                if configs.train_settings.add_sample_weight_to_class_loss_when_data_aug:
                    class_loss = torch.mean(tools['binary_loss'](classification_head, type_protein_pt.to(
                            tools['valid_device'])) * sample_weight_pt,dim = 0) #[9]
                else:
                    #class_loss = torch.mean(tools['binary_loss'](classification_head[:,1:], type_protein_pt[:,1:].to(
                    #        tools['valid_device'])))
                    class_loss = torch.mean(tools['binary_loss'](classification_head, type_protein_pt.to(
                                   tools['valid_device'])),dim = 0) #keep the shape of all heads different from train_loop
            else:
                if configs.train_settings.add_sample_weight_to_class_loss_when_data_aug:
                    class_loss = torch.mean(tools['binary_loss'](classification_head, type_protein_pt.to(
                            tools['valid_device'])) * sample_weight_pt,dim = 0) #[9]
                else:
                    #class_loss = torch.mean(tools['binary_loss'](classification_head[:,1:], type_protein_pt[:,1:].to(
                    #        tools['valid_device'])))
                    class_loss = torch.mean(tools['binary_loss'](classification_head, type_protein_pt.to(
                                   tools['valid_device'])),dim = 0) #keep the shape of all heads different from train_loop

            # Simplified code for weighted loss calculation
            # position_loss_weighted = position_loss / configs.train_settings.position_loss_T if configs.train_settings.position_loss_T != 1 else position_loss
            # if configs.train_settings.only_use_position_loss:
            #     weighted_loss_sum = position_loss_weighted  # yichuan updated on 0610 and 0601
            # else:
            #     weighted_loss_sum = class_loss + position_loss_weighted  # yichuan updated on 0612
            # Determine the weighted position loss based on a configurable threshold.
            if configs.train_settings.position_loss_T != 1:
                position_loss_weighted = position_loss / configs.train_settings.position_loss_T
            else:
                position_loss_weighted = position_loss

            # Determine the weighted class loss based on a configurable threshold.
            if configs.train_settings.class_loss_T != 1:
                class_loss_weighted = class_loss / configs.train_settings.class_loss_T
            else:
                class_loss_weighted = class_loss

            # Calculate the weighted sum of losses.
            if configs.train_settings.only_use_position_loss:
                # If configured to only use position loss, ignore class loss.
                weighted_loss_sum = position_loss_weighted
            else:
                # Otherwise, sum class loss and weighted position loss.
                weighted_loss_sum = class_loss_weighted + position_loss_weighted

            test_loss += weighted_loss_sum #.item()
            test_position_loss += position_loss #.item()
            test_class_loss += class_loss #.item()
            # label = torch.argmax(label_1hot, dim=1)
            # type_pred = torch.argmax(type_probab, dim=1)
            # accuracy.update(type_pred.detach(), label.detach().to(tools['valid_device']))
            # macro_f1_score.update(type_pred.detach(), label.detach().to(tools['valid_device']))
            # f1_score.update(type_pred.detach(), label.detach().to(tools['valid_device']))

        test_loss = test_loss / num_batches
        test_position_loss = test_position_loss / num_batches
        test_class_loss = test_class_loss / num_batches
        # epoch_acc = np.array(accuracy.compute().cpu())
        # epoch_macro_f1 = macro_f1_score.compute().cpu().item()
        # epoch_f1 = np.array(f1_score.compute().cpu())
        # acc_cs = cs_correct / cs_num
    return test_loss,test_class_loss,test_position_loss #changed to tensor [8]


def frag2protein(data_dict, tools):
    overlap=tools['frag_overlap']
    # no_overlap=tools['max_len']-2-overlap
    for id_protein in data_dict.keys():
        id_frag_list = data_dict[id_protein]['id_frag']
        seq_protein=""
        motif_logits_protein=np.array([])
        motif_target_protein=np.array([])
        for i in range(len(id_frag_list)):
            id_frag = id_protein+"@"+str(i)
            ind = id_frag_list.index(id_frag)
            seq_frag = data_dict[id_protein]['seq_frag'][ind]
            target_frag = data_dict[id_protein]['target_frag'][ind]
            motif_logits_frag = data_dict[id_protein]['motif_logits'][ind]
            l=len(seq_frag)
            if i==0:
                seq_protein=seq_frag
                motif_logits_protein=motif_logits_frag[:,:l]
                motif_target_protein=target_frag[:,:l]
            else:
                seq_protein = seq_protein + seq_frag[overlap:]
                # x_overlap = np.maximum(motif_logits_protein[:,-overlap:], motif_logits_frag[:,:overlap])
                x_overlap = (motif_logits_protein[:,-overlap:] + motif_logits_frag[:,:overlap])/2
                motif_logits_protein = np.concatenate((motif_logits_protein[:,:-overlap], x_overlap, motif_logits_frag[:,overlap:l]),axis=1)
                motif_target_protein = np.concatenate((motif_target_protein, target_frag[:,overlap:l]), axis=1)
        data_dict[id_protein]['seq_protein']=seq_protein
        data_dict[id_protein]['motif_logits_protein']=motif_logits_protein
        data_dict[id_protein]['motif_target_protein']=motif_target_protein
    return data_dict


def get_data_dict(args, dataloader, tools,load_best):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.eval().cuda()
    if load_best:
        model_path = os.path.join(tools['checkpoint_path'], f'best_model.pth')
        if args.predict == 1 and os.path.exists(args.resume_path):
           model_path = args.resume_path
        
        customlog(tools['logfilepath'], f"Loading checkpoint from {model_path}\n")
        print(f"Loading checkpoint from {model_path}\n")
        load_checkpoints_only(tools['net'],model_path,tools['num_classes'])
    else:
        customlog(tools['logfilepath'], f"Use on-line model\n")
    
    tools['net'].eval().to(tools["valid_device"])
    n = tools['num_classes']

    # cutoff = tools['cutoff']
    data_dict = {}
    with torch.no_grad():
        # for batch, (id, id_frags, seq_frag, target_frag, type_protein) in enumerate(dataloader):
        for batch, (id_tuple, id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple, sample_weight_tuple,_,) in enumerate(dataloader):
            # id_frags_list, seq_frag_tuple, target_frag_tuple = make_buffer(id_frags, seq_frag, target_frag)
            id_frags_list, seq_frag_tuple, target_frag_pt, type_protein_pt = make_buffer(id_frag_list_tuple,
                                                                                         seq_frag_list_tuple,
                                                                                         target_frag_nplist_tuple,
                                                                                         type_protein_pt_tuple)
            encoded_seq = tokenize(tools, seq_frag_tuple)
            if type(encoded_seq) == dict:
                for k in encoded_seq.keys():
                    encoded_seq[k] = encoded_seq[k].to(tools['valid_device'])
            else:
                encoded_seq = encoded_seq.to(tools['valid_device'])
            classification_head, motif_logits= tools['net'](encoded_seq, id_tuple, id_frags_list, seq_frag_tuple)
            m = torch.nn.Softmax(dim=1)  #torch.nn.Sigmoid()
            motif_logits = m(motif_logits) #batch, 9, len
            classification_head =  torch.nn.Sigmoid()(classification_head) #batch,9
            
            x_frag = np.array(motif_logits.cpu())  # [batch, head, seq]
            y_frag = np.array(target_frag_pt.cpu())  # [batch, head, seq]
            x_pro = np.array(classification_head.cpu())  # [sample, 9]
            y_pro = np.array(type_protein_pt.cpu())  # [sample, 9]
            for i in range(len(id_frags_list)):
                id_protein = id_frags_list[i].split('@')[0]
                j = id_tuple.index(id_protein)
                if id_protein in data_dict.keys():
                    data_dict[id_protein]['id_frag'].append(id_frags_list[i])
                    data_dict[id_protein]['seq_frag'].append(seq_frag_tuple[i])
                    data_dict[id_protein]['target_frag'].append(y_frag[i])  # [[head, seq], ...]
                    data_dict[id_protein]['motif_logits'].append(x_frag[i])  # [[head, seq], ...]
                else:
                    data_dict[id_protein] = {}
                    data_dict[id_protein]['id_frag'] = [id_frags_list[i]]
                    data_dict[id_protein]['seq_frag'] = [seq_frag_tuple[i]]
                    data_dict[id_protein]['target_frag'] = [y_frag[i]]
                    data_dict[id_protein]['motif_logits'] = [x_frag[i]]
                    
                    data_dict[id_protein]['type_pred'] = x_pro[j] #sample, 9
                    data_dict[id_protein]['type_target'] = y_pro[j]
                    

        data_dict = frag2protein(data_dict, tools)
    return data_dict

def maxByVar(numbers, rownum, threshold=2.0):
    max_values = np.max(numbers, axis=0)
    mean_values = np.mean(numbers, axis=0)
    std_devs = np.std(numbers, axis=0)
    result=np.zeros(numbers.shape[1])
    for i in range(numbers.shape[1]):
        if numbers[rownum,i] == max_values[i] and numbers[rownum, i] >= mean_values[i] + std_devs[i] * threshold:
            result[i] = 1
    return result

def evaluate_protein(data_dict, tools, constrain):
    n = tools['num_classes']
    classname = list(label2idx.keys())
    criteria = ["matthews_corrcoef","recall_score", "precision_score", "f1_score"]
    #["roc_auc_score", "average_precision_score", "matthews_corrcoef",
    #            "recall_score", "precision_score", "f1_score"]
    
    IoU_pro_difcut = np.zeros([n])  # just for nuc and nuc_export
    # result_pro_difcut=np.zeros([n,6,9])
    result_pro_difcut = np.zeros([n, 4])
    # cs_acc_difcut=np.zeros([n, 9])
    cs_acc_difcut = np.zeros([n])
    scores = get_scores(tools, n, data_dict, constrain)
    IoU_pro_difcut = scores['IoU_pro']
    TPR_FPR_FNR_difcut = scores['TPR_FPR_FNR']
    cs_acc_difcut = scores['cs_acc']
    result_pro_difcut = scores['result_pro']

    customlog(tools["logfilepath"], f"===================Evaluate protein results========================\n")

    customlog(tools["logfilepath"], f"===========================================\n")
    customlog(tools["logfilepath"], f" Jaccard Index (protein): \n")
    IoU_pro_difcut = pd.DataFrame(IoU_pro_difcut, index=classname)
    IoU_pro_difcut_selected_rows = IoU_pro_difcut.iloc[[label2idx['Nucleus'], label2idx['Nucleus_export']]]
    customlog(tools["logfilepath"], IoU_pro_difcut_selected_rows.__repr__())

    customlog(tools["logfilepath"], f"===========================================\n")
    customlog(tools["logfilepath"], f" TPR, FPR, FNR: \n")
    TPR_FPR_FNR_difcut = pd.DataFrame(TPR_FPR_FNR_difcut, index=classname)
    TPR_FPR_FNR_difcut_selected_rows = TPR_FPR_FNR_difcut.iloc[[label2idx['Nucleus'], label2idx['Nucleus_export']]]
    customlog(tools["logfilepath"], TPR_FPR_FNR_difcut_selected_rows.__repr__())

    customlog(tools["logfilepath"], f"===========================================\n")
    customlog(tools["logfilepath"], f" cs acc: \n")
    cs_acc_difcut = pd.DataFrame(cs_acc_difcut, index=classname)
    rows_to_exclude = [label2idx['Nucleus'], label2idx['Nucleus_export']]
    filtered_df = cs_acc_difcut.drop(cs_acc_difcut.index[rows_to_exclude])
    customlog(tools["logfilepath"], filtered_df.__repr__())

    customlog(tools["logfilepath"], f"===========================================\n")
    customlog(tools["logfilepath"], f" Class prediction performance: \n")
    tem = pd.DataFrame(result_pro_difcut, columns=criteria, index=classname)
    customlog(tools["logfilepath"], tem.__repr__())


def get_scores(tools, n, data_dict, constrain):
    cs_num = np.zeros(n)
    cs_correct = np.zeros(n)
    cs_acc = np.zeros(n)
    # Yichuan 0612
    TPR_pro_avg = np.zeros(n)
    FPR_pro_avg = np.zeros(n)
    FNR_pro_avg = np.zeros(n)
    TPR_FPR_FNR_pro_avg = [None] * n

    IoU_pro = np.zeros(n)
    condition1=condition2=condition3=0
    result_pro = np.zeros([n, 4]) #mcc,precision, recall,f1_score
    for head in range(1,n):
        x_list = []
        y_list = []
        for id_protein in data_dict.keys():
            # x_frag_predict = np.argmax(data_dict[id_protein]['motif_logits_protein'],axis=0)
            # if head in x_frag_predict:
            if 1 in maxByVar(data_dict[id_protein]['motif_logits_protein'], head):
                x_pro = True
            else:
                x_pro = False
            
            #x_pro = head == np.argmax(data_dict[id_protein]['type_pred'], axis=0)
            y_pro = data_dict[id_protein]['type_target'][head]
            
            x_list.append(x_pro)
            y_list.append(y_pro)
            if constrain:
                condition = x_pro
            else:
                condition = True
            if y_pro == 1 and condition:
                #data_dict[id_protein]['motif_logits_protein'] is 9 x seq_len
                y_frag = data_dict[id_protein]['motif_target_protein'][head]
                x_frag = data_dict[id_protein]['motif_logits_protein'][head]
                # x_frag_mask = np.argmax(data_dict[id_protein]['motif_logits_protein'],axis=0)==head #postion max signal is head
                x_frag_mask = maxByVar(data_dict[id_protein]['motif_logits_protein'], head)==1
                #"""
                
                TPR_pro = np.sum((x_frag_mask ==1) * (y_frag == 1)) / np.sum(y_frag == 1)
                FPR_pro = np.sum((x_frag_mask ==1) * (y_frag == 0)) / np.sum(y_frag == 0)
                FNR_pro =  np.sum((x_frag_mask ==0) * (y_frag == 1)) / np.sum(y_frag == 1)
                # IoU_pro[head] += TPR_pro / (TPR_pro + FPR_pro + FNR_pro)
                IoU_pro[head] += sov_score(y_frag, x_frag_mask)
                TPR_pro_avg[head] += TPR_pro
                FPR_pro_avg[head] += FPR_pro
                FNR_pro_avg[head] += FNR_pro

                cs_num[head] += 1 #np.sum(y_frag == 1) > 0 because y_pro == 1 
                #if np.sum(y_frag == 1) > 0: #because y_pro == 1 
                cs_correct[head] += (np.argmax(x_frag) == np.argmax(y_frag))

        TPR_pro_avg[head] = TPR_pro_avg[head] / sum(y_list)
        FPR_pro_avg[head] = FPR_pro_avg[head] / sum(y_list)
        FNR_pro_avg[head] = FNR_pro_avg[head] / sum(y_list)
        TPR_FPR_FNR_pro_avg[head] = (TPR_pro_avg[head], FPR_pro_avg[head], FNR_pro_avg[head])

        IoU_pro[head] = IoU_pro[head] / sum(y_list)
        cs_acc[head] = cs_correct[head] / cs_num[head]

        pred = np.array(x_list)
        target = np.array(y_list)

        #try:
        #    result_pro[head, 0] = roc_auc_score(target, pred)
        #except ValueError:
        #    result_pro[head, 0] = np.nan
        #try:
        #    result_pro[head, 1] = average_precision_score(target, pred)
        #except ValueError:
        #    result_pro[head, 1] = np.nan
        try:
            result_pro[head, 0] = matthews_corrcoef(target, pred)
        except ValueError:
            result_pro[head, 0] = np.nan
        try:
            result_pro[head, 1] = recall_score(target, pred)
        except ValueError:
            result_pro[head, 1] = np.nan
        try:
            result_pro[head, 2] = precision_score(target, pred)
        except ValueError:
            result_pro[head, 2] = np.nan
        try:
            result_pro[head, 3] = f1_score(target, pred)
        except ValueError:
            result_pro[head, 3] = np.nan

    scores = {"IoU_pro": IoU_pro,  # [n]
              "result_pro": result_pro,  # [n, 6]
              "cs_acc": cs_acc,  # [n]
              "TPR_FPR_FNR": TPR_FPR_FNR_pro_avg}
    return scores




def evaluate_protein_targetP(data_dict, tools, constrain):
    n = tools['num_classes']
    classname = list(label2idx.keys())
    criteria = ["matthews_corrcoef","recall_score", "precision_score", "f1_score"]
    #["roc_auc_score", "average_precision_score", "matthews_corrcoef",
    #            "recall_score", "precision_score", "f1_score"]
    
    #IoU_pro_difcut = np.zeros([n])  # just for nuc and nuc_export
    # result_pro_difcut=np.zeros([n,6,9])
    result_pro_difcut = np.zeros([n, 4])
    # cs_acc_difcut=np.zeros([n, 9])
    cs_acc_difcut = np.zeros([n])
    scores = get_scores_targetP(tools, n, data_dict, constrain)
    #IoU_pro_difcut = scores['IoU_pro']
    cs_acc_difcut = scores['cs_acc']
    result_pro_difcut = scores['result_pro']

    customlog(tools["logfilepath"], f"===================Evaluate targetP protein results========================\n")
    customlog(tools["logfilepath"], f"===========================================\n")
    customlog(tools["logfilepath"], f" cs acc: \n")
    cs_acc_difcut = pd.DataFrame(cs_acc_difcut, index=classname)
    rows_to_exclude = [label2idx['Nucleus'], label2idx['Nucleus_export']]
    filtered_df = cs_acc_difcut.drop(cs_acc_difcut.index[rows_to_exclude])
    customlog(tools["logfilepath"], filtered_df.__repr__())
    customlog(tools["logfilepath"], f"===========================================\n")
    customlog(tools["logfilepath"], f" Class prediction performance: \n")
    tem = pd.DataFrame(result_pro_difcut, columns=criteria, index=classname)
    customlog(tools["logfilepath"], tem.__repr__())


def get_scores_targetP(tools, n, data_dict, constrain):
    cs_num = np.zeros(n)
    cs_correct = np.zeros(n)
    cs_acc = np.zeros(n)
    # Yichuan 0612
    TPR_pro_avg = np.zeros(n)
    FPR_pro_avg = np.zeros(n)
    FNR_pro_avg = np.zeros(n)
    TPR_FPR_FNR_pro_avg = [None] * n

    IoU_pro = np.zeros(n)
    condition1=condition2=condition3=0
    result_pro = np.zeros([n, 4]) #mcc,precision, recall,f1_score
    x_list=[]
    y_list=[]
    for head in [0,1,2,5,6]: #no cs only other scores
        for id_protein in data_dict.keys():
            # x_frag_predict = np.argmax(data_dict[id_protein]['motif_logits_protein'],axis=0)
            # if head in x_frag_predict:
            if 1 in maxByVar(data_dict[id_protein]['motif_logits_protein'], head):
                x_pro = True
            else:
                x_pro = False
            
            #x_pro = head == np.argmax(data_dict[id_protein]['type_pred'], axis=0)
            y_pro = data_dict[id_protein]['type_target'][head]
            
            x_list.append(x_pro)
            y_list.append(y_pro)
    
    pred = np.array(x_list)
    target = np.array(y_list)
    try:
        result_pro[0, 0] = matthews_corrcoef(target, pred)
    except ValueError:
        result_pro[0, 0] = np.nan
    try:
        result_pro[0, 1] = recall_score(target, pred)
    except ValueError:
        result_pro[0, 1] = np.nan
    try:
        result_pro[0, 2] = precision_score(target, pred)
    except ValueError:
        result_pro[0, 2] = np.nan
    try:
        result_pro[0, 3] = f1_score(target, pred)
    except ValueError:
        result_pro[0, 3] = np.nan
    
    cs_acc[0]=0
    for head in [3,4,7,8]:
        x_list = []
        y_list = []
        for id_protein in data_dict.keys():
            # x_frag_predict = np.argmax(data_dict[id_protein]['motif_logits_protein'],axis=0)
            # if head in x_frag_predict:
            if 1 in maxByVar(data_dict[id_protein]['motif_logits_protein'], head):
                x_pro = True
            else:
                x_pro = False
            
            #x_pro = head == np.argmax(data_dict[id_protein]['type_pred'], axis=0)
            y_pro = data_dict[id_protein]['type_target'][head]
            
            x_list.append(x_pro)
            y_list.append(y_pro)
            if constrain:
                condition = x_pro
            else:
                condition = True
            if y_pro == 1 and condition:
                #data_dict[id_protein]['motif_logits_protein'] is 9 x seq_len
                y_frag = data_dict[id_protein]['motif_target_protein'][head]
                x_frag = data_dict[id_protein]['motif_logits_protein'][head]
                # x_frag_mask = np.argmax(data_dict[id_protein]['motif_logits_protein'],axis=0)==head #postion max signal is head
                x_frag_mask = maxByVar(data_dict[id_protein]['motif_logits_protein'], head)==1

                cs_num[head] += 1 #np.sum(y_frag == 1) > 0 because y_pro == 1 
                #if np.sum(y_frag == 1) > 0: #because y_pro == 1 
                cs_correct[head] += (np.argmax(x_frag) == np.argmax(y_frag))

        cs_acc[head] = cs_correct[head] / cs_num[head]

        pred = np.array(x_list)
        target = np.array(y_list)

        try:
            result_pro[head, 0] = matthews_corrcoef(target, pred)
        except ValueError:
            result_pro[head, 0] = np.nan
        try:
            result_pro[head, 1] = recall_score(target, pred)
        except ValueError:
            result_pro[head, 1] = np.nan
        try:
            result_pro[head, 2] = precision_score(target, pred)
        except ValueError:
            result_pro[head, 2] = np.nan
        try:
            result_pro[head, 3] = f1_score(target, pred)
        except ValueError:
            result_pro[head, 3] = np.nan

    scores = {#"IoU_pro": IoU_pro,  # [n]
              "result_pro": result_pro,  # [n, 6]
              "cs_acc": cs_acc,  # [n]
              #"TPR_FPR_FNR": TPR_FPR_FNR_pro_avg
              }
    return scores

def main(config_dict, args,valid_batch_number, test_batch_number):
    configs = load_configs(config_dict,args)
    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)
    
    torch.cuda.empty_cache()
    curdir_path, result_path, checkpoint_path, logfilepath = prepare_saving_dir(configs,args.config_path)
    
    train_writer, valid_writer = prepare_tensorboard(result_path)    
    customlog(logfilepath, f'use k-fold index: {valid_batch_number}\n')
    
    if configs.train_settings.dataloader=="batchsample":
        dataloaders_dict = prepare_dataloader_batchsample(configs, valid_batch_number, test_batch_number)
    # elif configs.train_settings.dataloader=="clean":
    #       dataloaders_dict = prepare_dataloader_clean(configs, valid_batch_number, test_batch_number)
    # elif configs.train_settings.dataloader == "batchbalance":
    #       dataloaders_dict = prepare_dataloader_batchbalance(configs, valid_batch_number, test_batch_number)
    
    #debug_dataloader(dataloaders_dict["train"]) #981
    customlog(logfilepath, "Done Loading data\n")
    customlog(logfilepath, f'number of steps for training data: {len(dataloaders_dict["train"])}\n')
    customlog(logfilepath, f'number of steps for valid data: {len(dataloaders_dict["valid"])}\n')
    customlog(logfilepath, f'number of steps for test data: {len(dataloaders_dict["test"])}\n')
    print(f'number of steps for training data: {len(dataloaders_dict["train"])}\n')
    print(f'number of steps for valid data: {len(dataloaders_dict["valid"])}\n')
    print(f'number of steps for test data: {len(dataloaders_dict["test"])}\n')
    """adapter 0611 begin"""
    if configs.encoder.composition=="official_esm_v2":
        encoder=prepare_models(configs,logfilepath, curdir_path)
        customlog(logfilepath, "Done initialize model\n")
        print("Done initialize model\n")

        alphabet = encoder.model.alphabet
        tokenizer = alphabet.get_batch_converter(
            truncation_seq_length=configs.encoder.max_len
        )
        customlog(logfilepath, "Done initialize tokenizer\n")
        print("Done initialize tokenizer\n")
    else:
        tokenizer=prepare_tokenizer(configs, curdir_path)
        customlog(logfilepath, "Done initialize tokenizer\n")
        print("Done initialize tokenizer\n")

        encoder=prepare_models(configs,logfilepath, curdir_path)
        customlog(logfilepath, "Done initialize model\n")
        print("Done initialize model\n")
    """adapter 0611 end"""
    
    #optimizer for shared model (esm2 part)
    optimizer_task = {}
    scheduler_task = {}
    optimizer, scheduler = prepare_optimizer(encoder.model, configs, len(dataloaders_dict["train"]), logfilepath)
    optimizer_task['shared'] = optimizer
    scheduler_task['shared'] = scheduler
    for task_i in range(1,configs.encoder.num_classes): #8 classes
        optimizer, scheduler = prepare_optimizer(encoder.ParallelDecoders.decoders[task_i-1], configs, len(dataloaders_dict["train"]), logfilepath)
        optimizer_task[task_i] = optimizer
        scheduler_task[task_i] = scheduler
    
    if configs.optimizer.mode == 'skip':
        scheduler = optimizer
    customlog(logfilepath, 'preparing optimizer is done\n')
    if args.predict !=1:
        #initialize all the envrionment variables
        best_valid_loss = np.full(configs.encoder.num_classes, np.inf).tolist()
        #best_valid_position_loss = np.full(configs.encoder.num_classes, np.inf).tolist()
        stop_task = np.full(configs.encoder.num_classes,False).tolist() #8 or 9?
        counter_task = np.full(configs.encoder.num_classes,0).tolist()
        start_epoch = 1
        if configs.resume.resume:
                #resume will not set stop_task,if a previouse task stopped it can still get update if a better valid_loss achieved.
                #to do should also save the stop_task, can save but can load_all_checkpoints has stop_task?? best_checkpoint will not record the stop_task
                # best model里面存的stop_task 是patience以前的，stop_task后面改变了 但是current best model不会存的，应该再多一个函数让它存才可以。
                # resume_allseperate_checkpoints 最好可以返回上次停止时候真实的 stop_task,但是 load_all_checkpoints 就不要,因为load_all_checkpoints 只是要全部weights匹配，但是stop_task已经在后面更新了
                
                encoder, optimizer_task, scheduler_task,start_epoch,best_valid_loss, counter_task,stop_task = resume_allseperate_checkpoints(configs, optimizer_task, scheduler_task, logfilepath, checkpoint_path,encoder)
                if np.sum(stop_task)!=0: #有一个task在存这个checkpoint之前stop了
                    for class_index in range(1,len(stop_task)):
                        if stop_task[class_index]:
                            for param in tools['net'].ParallelDecoders.decoders[class_index-1].parameters():
                                      param.requires_grad = False
                    
                    #to do freeze all encoder parameters() for the firs stop
                    for param in tools['net'].model.parameters():
                         param.requires_grad = False
                
    # print('start epoch', start_epoch)
    # exit(0)
    # w=(torch.ones([9,1,1])*5).to(configs.train_settings.device)
    #w = torch.tensor(configs.train_settings.loss_pos_weight, dtype=torch.float32).to(configs.train_settings.device)
    #w_nucleus = torch.tensor(configs.train_settings.loss_pos_weight_nucleus, dtype=torch.float32).to(configs.train_settings.device)
    #w_nucleus_export = torch.tensor(configs.train_settings.loss_pos_weight_nucleus_export, dtype=torch.float32).to(configs.train_settings.device)
    #debug_dataloader(dataloaders_dict["train"]) #936 after call dataloaders_dict['train']
    
    tools = {
        'frag_overlap': configs.encoder.frag_overlap,
        'cutoffs': configs.predict_settings.cutoffs,
        'composition': configs.encoder.composition, 
        'max_len': configs.encoder.max_len,
        'tokenizer': tokenizer,
        'prm4prmpro': configs.encoder.prm4prmpro,
        'net': encoder,
        'train_loader': dataloaders_dict["train"],
        'valid_loader': dataloaders_dict["valid"],
        'test_loader': dataloaders_dict["test"],
        'train_device': configs.train_settings.device,
        'valid_device': configs.valid_settings.device,
        'train_batch_size': configs.train_settings.batch_size,
        'valid_batch_size': configs.valid_settings.batch_size,
        #'optimizer': optimizer,
        'optimizer_task': optimizer_task,
        # 'loss_function': torch.nn.CrossEntropyLoss(reduction="none"),
        # 'loss_function': torch.nn.BCEWithLogitsLoss(pos_weight=w, reduction="mean"),
        'loss_function': torch.nn.CrossEntropyLoss(reduction="none"),
        'binary_loss': torch.nn.BCEWithLogitsLoss(reduction="none"),
        # yichuan: loss_function的BCEWithLogitsLoss原来是mean, 被我改成none, 我在代码中加入了torch.mean
        #'loss_function_6': torch.nn.BCEWithLogitsLoss(pos_weight=w, reduction="sum"),
        #'loss_function_nucleus': torch.nn.BCEWithLogitsLoss(pos_weight=w_nucleus, reduction="sum"),
        #'loss_function_nucleus_export': torch.nn.BCEWithLogitsLoss(pos_weight=w_nucleus_export, reduction="sum"),
        #'pos_weight': w,
        #'loss_function': torch.nn.BCELoss(reduction="none"),
        #'loss_function_pro': torch.nn.BCELoss(reduction="none"),
        #'loss_function_pro': torch.nn.CrossEntropyLoss(reduction="none"),
        # 'loss_function_supcon': SupConHardLoss,  # Yichuan
        'checkpoints_every': configs.checkpoints_every,
        #'scheduler': scheduler,
        'scheduler_task': scheduler_task,
        'result_path': result_path,
        'checkpoint_path': checkpoint_path,
        'logfilepath': logfilepath,
        'num_classes': configs.encoder.num_classes,
        'predict':args.predict
        # 'masked_lm_data_collator': masked_lm_data_collator,
    }

    if args.predict !=1:
        customlog(logfilepath, f'number of train steps per epoch: {len(tools["train_loader"])}\n')
        patience = configs.train_settings.patience
        
        global global_step
        global_step=0
        if configs.train_settings.dataloader=="clean":
           total_steps_per_epoch = int(len(tools["train_loader"].dataset.samples)/configs.train_settings.batch_size)
        
        finish_training = False
        epoch = start_epoch
        customlog(logfilepath, f"Start training...at {epoch}\n")
        #for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):
        while(epoch <= configs.train_settings.num_epochs and not finish_training):
            tools['epoch'] = epoch
            
            start_time = time()

            train_loss = train_loop(tools, configs, train_writer, stop_task)
            if configs.train_settings.dataloader != "clean":
                if configs.train_settings.data_aug.enable and epoch >=configs.train_settings.data_aug.warmup and epoch < configs.train_settings.num_epochs-configs.train_settings.data_aug.cooldown:
                   tools['train_loader'].dataset.samples = tools['train_loader'].dataset.data_aug_train(tools['train_loader'].dataset.original_samples,configs,tools['train_loader'].dataset.class_weights)
            
            if configs.train_settings.data_aug.enable and epoch ==  configs.train_settings.num_epochs-configs.train_settings.data_aug.cooldown:
                tools['train_loader'].dataset.samples = tools['train_loader'].dataset.original_samples
                configs.train_settings.data_aug.enable = False
                print("data length back to original", str(len(tools['train_loader'].dataset.samples)))
                customlog(logfilepath, f"Fold {valid_batch_number} Epoch {epoch} data length back to original {len(tools['train_loader'].dataset.samples)}")
            
            train_writer.add_scalar('epoch loss',train_loss,global_step=epoch)
            end_time = time()
        
        
            if epoch % configs.valid_settings.do_every == 0 and epoch != 0:
                customlog(logfilepath, f'Epoch {epoch}: train loss: {train_loss:>5f}\n')
                print(f'Epoch {epoch}: train loss: {train_loss:>5f}\n')
                print(f"Fold {valid_batch_number} Epoch {epoch} validation...\n-------------------------------\n")
                customlog(logfilepath, f"Fold {valid_batch_number} Epoch {epoch} validation...\n-------------------------------\n")
                start_time = time()
                dataloader = tools["valid_loader"]
                customlog(logfilepath,f'Epoch {epoch}: stop_task status {stop_task}\n')
                valid_loss,valid_class_loss,valid_position_loss = test_loop(tools, dataloader,train_writer,valid_writer,configs) #In test loop, never test supcon loss
                
                valid_writer.add_scalar('epoch loss',torch.mean(valid_loss).item(),global_step=epoch)
                valid_writer.add_scalar('epoch class_loss',torch.mean(valid_class_loss).item(),global_step=epoch)
                valid_writer.add_scalar('epoch position_loss',torch.mean(valid_position_loss).item(),global_step=epoch)
                for class_index in range(0,len(valid_loss)): #don't need to skip other task
                    customlog(logfilepath,f'Epoch {epoch}: valid loss({class_index}):{valid_loss[class_index].item():>5f}\n')
                    customlog(logfilepath,f'Epoch {epoch}: {class_index} valid_class_loss:{valid_class_loss[class_index].item():>5f}\tvalid_position_loss:{valid_position_loss[class_index].item():>5f}\n')
                    print(f'Epoch {epoch}: valid loss({class_index}):{valid_loss[class_index].item():>5f}\n')
                    print(f'Epoch {epoch}: valid_class_loss({class_index}):{valid_class_loss[class_index].item():>5f}\tvalid_position_loss:{valid_position_loss[class_index].item():>5f}\n')
                
                end_time = time()
                #"""
                for class_index in range(1,len(valid_loss)):#skip other task
                  if not stop_task[class_index]:
                      if valid_loss[class_index] < best_valid_loss[class_index]:
                        counter_task[class_index] = 0
                        customlog(logfilepath, f"Epoch {epoch}: valid loss({class_index}) {valid_loss[class_index]} smaller than best loss {best_valid_loss[class_index]}\n-------------------------------\n")
                        best_valid_loss[class_index] = valid_loss[class_index]
                        model_path_task = os.path.join(tools['checkpoint_path'], f'best_model_{class_index}.pth')
                        customlog(logfilepath, f"Epoch {epoch}: A better checkpoint is saved into {model_path_task} \n-------------------------------\n")
                        #if any task updates, need to save all checkpoints into a file to keep the envrionment the same
                        save_all_checkpoint(epoch,model_path_task,tools,best_valid_loss,counter_task,stop_task) 
                        #for each update save all checkpoints into a file of class_index
                        #save_seperate_checkpoint(epoch, model_path, tools,class_index,save_shared= True) 
                        #save class_index checkpoints with its shared model into model_path, for the first stop_task save the shared model
                        #save_shared for every best model, because if trainning stopped accidently can be resummed later.
                        #customlog(logfilepath, f"Epoch {epoch}: checkpoint of Other is saved into {model_path_task} \n")
                        #save_seperate_checkpoint(epoch, model_path, tools,0,save_shared= False) #save other class every time a task updates
                      else:
                          counter_task[class_index] +=1
                          if counter_task[class_index] >= patience:
                                print(f"Stopping Task {class_index} early at epoch {epoch}")
                                customlog(logfilepath,f"Stopping Task {class_index} early at epoch {epoch}\n")
                                # mv the stopping task best checkpoint to the shared best model path, the task best checkpoint will be removed
                                model_path = os.path.join(tools['checkpoint_path'], f'best_model.pth')
                                model_path_task = os.path.join(tools['checkpoint_path'], f'best_model_{class_index}.pth')
                                customlog(logfilepath,f"mv the {model_path_task} to {model_path}\n")
                                shutil.move(model_path_task, model_path)
                                #because resume to model_path_task, all the other best checkpoins should be from model_path
                                for other_index in range(1,tools['num_classes']):
                                      model_path_othertasks = os.path.join(tools['checkpoint_path'], f'best_model_{other_index}.pth')
                                      if os.path.exists(model_path_othertasks):
                                            shutil.copy(model_path, model_path_othertasks)
                                            customlog(logfilepath,f"copy the {model_path} to {model_path_othertasks}\n")
                                
                                
                                # Should stop_task[class_index], but need to load the best checkpoint for other training
                                print(f"Loading bestmodel for all")
                                customlog(logfilepath,f"Loading bestmodel for all\n")
                                #one task stop, load its best checkpoints with the checkpoints for all the other tasks 
                                #load_seperate_checkpoints(configs,optimizer_task,scheduler_task,logfilepath,tools['net'],class_index,model_path,load_shared = np.sum(stop_task) == 0,restart_optimizer=False) #load task checkpoint with shared part, only load shared part if no stop_task == True
                                tools['net'],tools['optimizer_task'],tools['scheduler_task'],epoch,best_valid_loss,counter_task,stop_task_cannotuse = load_all_checkpoints(configs,optimizer_task,scheduler_task,logfilepath,tools['net'],model_path)
                                
                                #epoch go back to the best checkpoint for the task class_index and for all the other tasks.
                                # Freeze the task1_head parameters
                                for param in tools['net'].ParallelDecoders.decoders[class_index-1].parameters():
                                    param.requires_grad = False
                                #to do freeze all encoder parameters() for the firs stop
                                for param in tools['net'].model.parameters():
                                    param.requires_grad = False
                                
                                stop_task[class_index] = True
                                stop_task[0] = False #if no others' data, alwasys keep it to False. no need to do this because, class_index range from 1, but still have it.
                                #check if all the 1-8 tasks has stopped
                                if np.sum(stop_task[1:]) == configs.encoder.num_classes-1:
                                     #all tasks stopped, stop the other task and save the checkpoing
                                     #model_path = os.path.join(tools['checkpoint_path'], f'best_model.pth')
                                     #customlog(logfilepath, f"Epoch {epoch}: checkpoint of Other is saved into {model_path} \n-------------------------------\n")
                                     #save_seperate_checkpoint(epoch, model_path, tools,0,save_shared= False) #save class_index 
                                     #changed, already save other head when saving each best model. 
                                     finish_training = True
            
            epoch+=1
    
   
    #model_checkpoint = torch.load(model_path, map_location='cpu')
    #tools['net'].load_state_dict(model_checkpoint['model_state_dict'])
    start_time = time()
    if args.predict!=1:
       customlog(logfilepath, f"Fold {valid_batch_number} test\n-------------------------------\n")
       dataloader = tools["valid_loader"]
       data_dict = get_data_dict(args, dataloader, tools,load_best=True)
       customlog(logfilepath, f"===================Valid protein results constrain: False========================")
       evaluate_protein(data_dict, tools, False)
    
    #"""
    customlog(logfilepath, f"===================Evaluate on test data========================")
    dataloader = tools["test_loader"]
    data_dict = get_data_dict(args, dataloader, tools,load_best=True)
    customlog(logfilepath, f"===================Test protein results constrain: False========================\n================================================================================================\n")
    evaluate_protein(data_dict, tools, False)
    customlog(logfilepath, f"===================Test protein results constrain: True========================")
    evaluate_protein(data_dict, tools, True)


    # filter_list = ['Q9LPZ4', 'P15330', 'P35869', 'P51587', 'P70278', 'Q80UP3', 'Q8BMI0', 'Q9HC62', 'Q9UK80',
    #                'Q8LH59', 'P19484', 'P25963', 'P35123', 'Q6NVF4', 'Q8NG08', 'Q9BVS4', 'Q9NRA0', 'Q9NUL5', 'Q9UBP0', 'P78953',
    #                'A8MR65', 'Q8S4Q6', 'Q3U0V2', 'Q96D46', 'Q9NYA1', 'Q9ULX6', 'Q9WTL8', 'Q9WTZ9',
    #                'P35922', 'P46934', 'P81299', 'Q13148', 'Q6ICB0', 'Q7TPV4', 'Q8N884', 'Q99LG4', 'Q9Z207',
    #                'O00571', 'O35973', 'O54943', 'P52306', 'Q13015', 'Q13568', 'Q5TAQ9', 'Q8NAG6', 'Q9BZ23', 'Q9BZS1', 'Q9GZX7',
    #                ]  # 1, 2, 3, 4, 0


    #dataloader = tools["test_loader"]
    #data_dict = get_data_dict(args, dataloader, tools)
    testdata = {key for key, value in data_dict.items()}
    print('testdata')
    print(testdata, len(testdata))
    filtered_data_dict = {key: value for key, value in data_dict.items() if key in filter_list}
    print("len(filtered_data_dict)")
    print(len(filtered_data_dict))
    customlog(logfilepath, f"\n\n\n\nComparable Test Data\n-------------------------------\n")
    customlog(logfilepath, f"===================Test protein results constrain: False========================")
    evaluate_protein(filtered_data_dict, tools, False)
    customlog(logfilepath, f"===================Test protein results constrain: True========================")
    evaluate_protein(filtered_data_dict, tools, True)
    #"""
    #"""test targetP data
    dataloader = dataloaders_dict['test_targetp']
    data_dict = get_data_dict(args, dataloader, tools,load_best=True)
    customlog(logfilepath, f"\n===================Test targetP protein results constrain: False========================\n================================================================================================\n")
    evaluate_protein_targetP(data_dict, tools, False)
    customlog(logfilepath, f"===================Test targetP protein results constrain: True========================")
    evaluate_protein_targetP(data_dict, tools, True)
    #"""
    train_writer.close()
    valid_writer.close()
    end_time = time()
    
    del tools, encoder, dataloaders_dict, optimizer, scheduler
    torch.cuda.empty_cache()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CPM')
    parser.add_argument("--config_path", help="The location of config file", default='./config.yaml')
    parser.add_argument("--predict", type=int, help="predict:1 no training, call evaluate_protein; predict:0 call training loop", default=0)
    parser.add_argument("--result_path", default=None,
                        help="result_path, if setted by command line, overwrite the one in config.yaml, "
                             "by default is None")
    parser.add_argument("--resume_path", default=None,
                        help="if set, overwrite the one in config.yaml, by default is None")
    parser.add_argument("--resume", default=0,type=int,
                        help="if resume from resume_path?")
    parser.add_argument("--restart_optimizer",default=0,type=int,
                        help="restart optimizer? default: False")
    
    parser.add_argument("--fold_num", default=1, help="")
    
    args = parser.parse_args()

    config_path = args.config_path

    # print(args.fold_num)
    # exit(0)

    with open(config_path) as file:
        config_dict = yaml.full_load(file)

    for i in range(int(args.fold_num), int(args.fold_num)+1):
        valid_num = i
        if valid_num == 4:
            test_num = 0
        else:
            test_num = valid_num+1
        main(config_dict, args, valid_num, test_num)

    # main(config_dict, args, 0, 1)
    # main(config_dict, args, 1, 2)
    # main(config_dict, args, 2, 3)
    # main(config_dict, args, 3, 4)
    # main(config_dict, args, 4, 0)

"""
python train_save_seperate_fixvalresweight.py --config_path ./configs/config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --result_path ./outputs/train_predict_fold0/ --resume_path ../final_code_v1/results/residue_weight_savesepearte/5fold/fold1/checkpoints/best_model.pth --predict 1

python train_save_seperate_fixvalresweight.py --config_path ./configs/config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --result_path ./outputs/train_predict_fold0/ --resume_path ./results/models/5fold/fold1/checkpoints/best_model.pth --predict 1 --fold_num 0
python train_save_seperate_fixvalresweight.py --config_path ./configs/config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --result_path ./outputs/train_predict_fold1/ --resume_path ./results/models/5fold/fold2/checkpoints/best_model.pth --predict 1 --fold_num 1


python3 train.py --config_path ./configs/config0722_cnnlinear-Epoch15_M0.3_M0.1.yaml --result_path results/cnnlinear_E15_M0.3_M0.1/
python3 train.py --config_path ./configs/config0722_cnnlinear-Epoch15_M0.3_M0.1.yaml --result_path results/cnnlinear_E15_M0.3_M0.1_twoloss/

"""