import torch
torch.manual_seed(0)
import argparse
import os
import yaml
import numpy as np
from model import *
from utils import *
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import random
from time import time
import json
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 10)
from scipy.ndimage import gaussian_filter
from data_batchsample import *
from data_batchsample import prepare_dataloaders as prepare_dataloader_batchsample
from train_save_seperate_fixvalresweight import *
import json


# Function to convert the dictionaries to JSON-compatible format
def convert_to_json(id2head2sig, id2seq):
    # Convert the NumPy arrays to lists
    id2head2sig_json = {
        protein_id: {loc: indices.tolist() for loc, indices in loc_dict.items()}
        for protein_id, loc_dict in id2head2sig.items()
    }
    
    # Combine the dictionaries into one JSON-compatible structure
    combined_dict = {
        "id2head2sig": id2head2sig_json,
        "id2seq": id2seq
    }
    
    # Convert to JSON string
    json_output = json.dumps(combined_dict, indent=4)
    return json_output


def frag2protein_pred(data_dict, tools):
    overlap=tools['frag_overlap']
    # no_overlap=tools['max_len']-2-overlap
    for id_protein in data_dict.keys():
        id_frag_list = data_dict[id_protein]['id_frag']
        seq_protein=""
        motif_logits_protein=np.array([])
        for i in range(len(id_frag_list)):
            id_frag = id_protein+"@"+str(i)
            ind = id_frag_list.index(id_frag)
            seq_frag = data_dict[id_protein]['seq_frag'][ind]
            motif_logits_frag = data_dict[id_protein]['motif_logits'][ind]
            l=len(seq_frag)
            if i==0:
                seq_protein=seq_frag
                motif_logits_protein=motif_logits_frag[:,:l]
            else:
                seq_protein = seq_protein + seq_frag[overlap:]
                x_overlap = (motif_logits_protein[:,-overlap:] + motif_logits_frag[:,:overlap])/2                
                motif_logits_protein = np.concatenate((motif_logits_protein[:,:-overlap], x_overlap, motif_logits_frag[:,overlap:l]),axis=1)
        data_dict[id_protein]['seq_protein']=seq_protein
        # motif_logits_protein[0] = gaussian_filter(motif_logits_protein[0], sigma=2, truncate=1,mode="nearest") # nucleus
        # motif_logits_protein[4] = gaussian_filter(motif_logits_protein[4], sigma=2, truncate=1,mode="nearest") # nucleus_export
        data_dict[id_protein]['motif_logits_protein']=motif_logits_protein
    return data_dict


def present_single(tools, data_dict):
    n=tools['num_classes']
    classname = list(label2idx.keys())
    id2head={}
    
    id2seq={}
    for head in range(1,n):
        name=classname[head]
        for id_protein in data_dict.keys():
            seq = data_dict[id_protein]['seq_protein']
            id2seq[id_protein]=seq
            seq_len = len(seq)
            # x_frag_predict = np.argmax(data_dict[id_protein]['motif_logits_protein'],axis=0)
            # if head in x_frag_predict:
            if 1 in maxByVar(data_dict[id_protein]['motif_logits_protein'], head):
                x_pro = True
            else:
                x_pro = False
            if x_pro:
                x_frag = data_dict[id_protein]['motif_logits_protein'][head]  #[seq]
                # x_frag_mask = np.argmax(data_dict[id_protein]['motif_logits_protein'],axis=0)==head
                x_frag_mask = maxByVar(data_dict[id_protein]['motif_logits_protein'], head)==1

                cs = np.argmax(x_frag)
                motif=""
                if name in ["Mitochondrion","SIGNAL","chloroplast","Thylakoid"]:
                    motif=np.arange(0,cs+1)
                elif name =="ER":
                    motif=np.arange(cs,seq_len)
                elif name =="Peroxisome":
                    # if seq_len-cs>cs:
                    if seq_len-cs>4:
                        motif = np.arange(0,cs+1)
                    else:
                        motif = np.arange(cs,seq_len)
                elif name in ["Nucleus","Nucleus_export"]:
                    motif=np.where(x_frag_mask==True)[0]

                if id_protein in id2head.keys():
                    # head2sig[name]=motif
                    id2head[id_protein][name]=motif
                else:
                    id2head[id_protein]={}
                    id2head[id_protein][name]=motif

            else:
                if id_protein in id2head.keys():
                    # head2sig[name]=motif
                    id2head[id_protein][name]=np.array([])
                else:
                    id2head[id_protein]={}
                    id2head[id_protein][name]=np.array([])
            

    return id2head, id2seq


def make_buffer_pred(id_frag_list_tuple, seq_frag_list_tuple):
    id_frags_list = []
    seq_frag_list = []
    for i in range(len(id_frag_list_tuple)):
        id_frags_list.extend(id_frag_list_tuple[i])
        seq_frag_list.extend(seq_frag_list_tuple[i])
    seq_frag_tuple = tuple(seq_frag_list)
    return id_frags_list, seq_frag_tuple

def predict(dataloader, tools):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.eval().cuda()
    tools['net'].eval().to(tools["pred_device"])
    n=tools['num_classes']

    # cutoff = tools['cutoff']
    data_dict={}
    with torch.no_grad():
        for batch, (id_tuple, id_frag_list_tuple, seq_frag_list_tuple) in enumerate(dataloader):
            start_time = time()
            id_frags_list, seq_frag_tuple = make_buffer_pred(id_frag_list_tuple, seq_frag_list_tuple)
            encoded_seq=tokenize(tools, seq_frag_tuple)
            if type(encoded_seq)==dict:
                for k in encoded_seq.keys():
                    encoded_seq[k]=encoded_seq[k].to(tools['pred_device'])
            else:
                encoded_seq=encoded_seq.to(tools['pred_device'])

            classification_head, motif_logits= tools['net'](encoded_seq, id_tuple, 
                                                             id_frags_list, seq_frag_tuple)

            m = torch.nn.Softmax(dim=1)  #torch.nn.Sigmoid()
            motif_logits = m(motif_logits) #batch, 9, len
            classification_head =  torch.nn.Sigmoid()(classification_head) #batch,9

            x_frag = np.array(motif_logits.cpu())   #[batch, head, seq]
            x_pro = np.array(classification_head.cpu()) #[sample, n]
            for i in range(len(id_frags_list)):
                id_protein=id_frags_list[i].split('@')[0]
                j= id_tuple.index(id_protein)
                if id_protein in data_dict.keys():
                    data_dict[id_protein]['id_frag'].append(id_frags_list[i])
                    data_dict[id_protein]['seq_frag'].append(seq_frag_tuple[i])
                    data_dict[id_protein]['motif_logits'].append(x_frag[i])    #[[head, seq], ...]
                else:
                    data_dict[id_protein]={}
                    data_dict[id_protein]['id_frag']=[id_frags_list[i]]
                    data_dict[id_protein]['seq_frag']=[seq_frag_tuple[i]]
                    data_dict[id_protein]['motif_logits']=[x_frag[i]]
                    data_dict[id_protein]['type_pred']=x_pro[j]
            end_time = time()
            print("One batch time: "+ str(end_time-start_time))

        start_time = time()
        data_dict = frag2protein_pred(data_dict, tools)
        end_time = time()
        print("frag ensemble time: "+ str(end_time-start_time))

        start_time = time()
        # result_pro, motif_pred, result_id = get_prediction(n, data_dict)  # result_pro = [sample_size, class_num], sample order same as result_id  
                                                                                         # motif_pred = class_num dictionaries with protein id as keys
                                                                                         # result_id = [sample_size], protein ids
        end_time = time()
        print("get prediction time: "+ str(end_time-start_time))
        # present(tools, result_pro, motif_pred, result_id)

        start_time = time()
        # present3(tools, result_pro, motif_pred, result_id, data_dict)
        end_time = time()
        print("present time: "+ str(end_time-start_time))
        return data_dict

def get_scores_pred(tools, n, data_dict, constrain):
    cs_num = np.zeros(n)
    cs_correct = np.zeros(n)
    cs_acc = np.zeros(n)

    TPR_pro_avg = np.zeros(n)
    FPR_pro_avg = np.zeros(n)
    FNR_pro_avg = np.zeros(n)

    # TP_frag=np.zeros(n)
    # FP_frag=np.zeros(n)
    # FN_frag=np.zeros(n)
    # #Intersection over Union (IoU) or Jaccard Index
    # IoU = np.zeros(n)
    # Negtive_detect_num=0
    # Negtive_num=0
    # prot_cutoffs = list(tools["prot_cutoffs"])
    # AA_cutoffs = list(tools["AA_cutoffs"])

    # TPR_pro=np.zeros(n)
    # FPR_pro=np.zeros(n)
    # FNR_pro=np.zeros(n)
    IoU_pro = np.zeros(n)
    # Negtive_detect_pro=0
    # Negtive_pro=0
    condition1=condition2=condition3=0
    result_pro=np.zeros([n,4])
    for head in range(1,n):
        x_list=[]
        y_list=[]
        for id_protein in data_dict.keys():
            # x_pro = data_dict[id_protein]['type_pred'][head]  #[1]
            # y_pro = data_dict[id_protein]['type_target'][head]  #[1]   
            # x_list.append(x_pro)  
            # y_list.append(y_pro)
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
                x_frag = data_dict[id_protein]['motif_logits_protein'][head]  #[seq]
                y_frag = data_dict[id_protein]['motif_target_protein'][head]
                # x_frag_mask = np.argmax(data_dict[id_protein]['motif_logits_protein'],axis=0)==head #postion max signal is head
                x_frag_mask = maxByVar(data_dict[id_protein]['motif_logits_protein'], head)==1
                # Negtive_pro += np.sum(np.max(y)==0)
                # Negtive_detect_pro += np.sum((np.max(y)==0) * (np.max(x>=cutoff)==1))
                TPR_pro = np.sum((x_frag_mask ==1) * (y_frag == 1)) / np.sum(y_frag == 1)
                FPR_pro = np.sum((x_frag_mask ==1) * (y_frag == 0)) / np.sum(y_frag == 0)
                FNR_pro =  np.sum((x_frag_mask ==0) * (y_frag == 1)) / np.sum(y_frag == 1)
                # IoU_pro[head] += TPR_pro / (TPR_pro + FPR_pro + FNR_pro)
                IoU_pro[head] += sov_score(y_frag, x_frag_mask)
    
                cs_num[head] += 1 #np.sum(y_frag == 1) > 0 because y_pro == 1 
                #if np.sum(y_frag == 1) > 0: #because y_pro == 1 
                cs_correct[head] += (np.argmax(x_frag) == np.argmax(y_frag))

        IoU_pro[head] = IoU_pro[head] / sum(y_list)
        cs_acc[head] = cs_correct[head] / cs_num[head]

        pred=np.array(x_list)
        target=np.array(y_list)
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
        
    # for head in range(n):
    #     # IoU[head] = TP_frag[head] / (TP_frag[head] + FP_frag[head] + FN_frag[head])
    #     IoU_pro[head] = TPR_pro[head] / (TPR_pro[head] + FPR_pro[head] + FNR_pro[head])
    #     cs_acc[head] = cs_correct[head] / cs_num[head]
    # FDR_frag = Negtive_detect_num / Negtive_num
    # FDR_pro = Negtive_detect_pro / Negtive_pro
    
    scores = {"IoU_pro": IoU_pro,  # [n]
              "result_pro": result_pro,  # [n, 6]
              "cs_acc": cs_acc}  # [n]
    return scores

def maxByVar(numbers, rownum, threshold=2.0):
    max_values = np.max(numbers, axis=0)
    mean_values = np.mean(numbers, axis=0)
    std_devs = np.std(numbers, axis=0)
    result=np.zeros(numbers.shape[1])
    for i in range(numbers.shape[1]):
        if numbers[rownum,i] == max_values[i] and numbers[rownum, i] >= mean_values[i] + std_devs[i] * threshold:
            result[i] = 1
    return result

def get_evaluation(tools, data_dict, constrain):
    n=tools['num_classes']
    # IoU_pro_difcut=np.zeros([n, 9])  #just for nuc and nuc_export
    IoU_pro_difcut=np.zeros([n])  #just for nuc and nuc_export
    # result_pro_difcut=np.zeros([n,6,9])
    result_pro_difcut=np.zeros([n,4])
    # cs_acc_difcut=np.zeros([n, 9]) 
    cs_acc_difcut=np.zeros([n]) 
    classname = list(label2idx.keys())
    criteria = ["matthews_corrcoef","recall_score", "precision_score", "f1_score"]
    # cutoffs=[x / 10 for x in range(1, 10)]
    # cut_dim=0
    # for cutoff in cutoffs:
    scores=get_scores_pred(tools, n, data_dict, constrain)
    IoU_pro_difcut=scores['IoU_pro']
    result_pro_difcut=scores['result_pro']
    cs_acc_difcut=scores['cs_acc'] 
        # cut_dim+=1
    evaluation_file = os.path.join(tools['result_path'],"evaluation.txt")
    customlog(evaluation_file, f"===========================================\n")
    customlog(evaluation_file, f" Jaccard Index (protein): \n")
    IoU_pro_difcut = pd.DataFrame(IoU_pro_difcut, index=classname)
    IoU_pro_difcut_selected_rows = IoU_pro_difcut.iloc[[label2idx['Nucleus'], label2idx['Nucleus_export']]]
    customlog(evaluation_file, IoU_pro_difcut_selected_rows.__repr__())

    customlog(evaluation_file, f"===========================================\n")
    customlog(evaluation_file, f" cs acc: \n")
    cs_acc_difcut = pd.DataFrame(cs_acc_difcut, index=classname)
    rows_to_exclude = [label2idx['Nucleus'], label2idx['Nucleus_export']]
    filtered_df = cs_acc_difcut.drop(cs_acc_difcut.index[rows_to_exclude])
    customlog(evaluation_file, filtered_df.__repr__())

    customlog(evaluation_file, f"===========================================\n")
    customlog(evaluation_file, f" Class prediction performance: \n")
    tem = pd.DataFrame(result_pro_difcut, columns=criteria, index=classname)
    customlog(evaluation_file, tem.__repr__())

def get_individual_scores(tools, data_dict, constrain):
    n=tools['num_classes']
    # cutoffs = list(tools["cutoffs"])
    # AA_cutoffs = list(tools["AA_cutoffs"])
    # prot_cutoffs = list(tools["prot_cutoffs"])
    classname = list(label2idx.keys())
    evaluation_file = os.path.join(tools['result_path'],"evaluation_individual.txt")
    customlog(evaluation_file, f"ID\tlabel\tpredict\tIoU\tcs_correct\tmotif")
    for head in range(1,n):
        name=classname[head]
        x_list=[]
        y_list=[]
        # AA_cutoff = AA_cutoffs[head]
        # prot_cutoff = prot_cutoffs[head]
        for id_protein in data_dict.keys():
            seq = data_dict[id_protein]['seq_protein']
            seq_len = len(seq)
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
                x_frag = data_dict[id_protein]['motif_logits_protein'][head]  #[seq]
                y_frag = data_dict[id_protein]['motif_target_protein'][head]
                # x_frag_mask = np.argmax(data_dict[id_protein]['motif_logits_protein'],axis=0)==head #postion max signal is head
                x_frag_mask = maxByVar(data_dict[id_protein]['motif_logits_protein'], head)==1
                # Negtive_pro += np.sum(np.max(y)==0)
                # Negtive_detect_pro += np.sum((np.max(y)==0) * (np.max(x>=cutoff)==1))
                # TPR_pro = np.sum((x_frag>=AA_cutoff) * (y_frag==1))/np.sum(y_frag==1)
                # FPR_pro = np.sum((x_frag>=AA_cutoff) * (y_frag==0))/np.sum(y_frag==0)
                # FNR_pro = np.sum((x_frag<AA_cutoff) * (y_frag==1))/np.sum(y_frag==1)
                # # x_list.append(np.max(x))
                # # y_list.append(np.max(y))
                # IoU_pro = TPR_pro / (TPR_pro + FPR_pro + FNR_pro)

                TPR_pro = np.sum((x_frag_mask ==1) * (y_frag == 1)) / np.sum(y_frag == 1)
                FPR_pro = np.sum((x_frag_mask ==1) * (y_frag == 0)) / np.sum(y_frag == 0)
                FNR_pro =  np.sum((x_frag_mask ==0) * (y_frag == 1)) / np.sum(y_frag == 1)
                # IoU_pro = TPR_pro / (TPR_pro + FPR_pro + FNR_pro)
                IoU_pro = sov_score(y_frag, x_frag_mask)

                cs_correct = (np.argmax(x_frag) == np.argmax(y_frag))

                motif=""
                if name in ["Mitochondrion","SIGNAL","chloroplast","Thylakoid"]:
                    cs = np.argmax(x_frag)
                    motif=seq[0:cs+1]
                elif name =="ER":
                    cs = np.argmax(x_frag)
                    motif=seq[cs:]
                elif name =="Peroxisome":
                    cs = np.argmax(x_frag)
                    # if seq_len-cs>cs:
                    if seq_len-cs>4:
                        motif = seq[0:cs+1]
                    else:
                        motif = seq[cs:]
                elif name in ["Nucleus","Nucleus_export"]:
                    # sites = np.where(x_frag>AA_cutoff)[0]
                    sites = x_frag_mask ==1
                    if np.sum(sites)==0:
                        site = np.argmax(x_frag)
                        sites[site]=True
                        if site-2>=0:
                            sites[site-2:site]=True
                        if site+2<=seq_len-1:
                            sites[site+1:site+3]=True
                        # sites = [site]
                        # if site-2>=0:
                        #     sites = [site-2, site-1, site]
                        # if site+2<=seq_len-1:
                        #     sites.extend([site+1, site+2])
                    sites = np.array(sites)
                    motif = ''.join([seq[i] for i in range(len(sites)) if sites[i]])
                customlog(evaluation_file, f"{id_protein}\t{name}\t{x_pro}\t{IoU_pro}\t{cs_correct}\t{motif}\n")
                


def predict_withlabel(dataloader, tools, constrain):
    tools['net'].eval().to(tools["pred_device"])
    n=tools['num_classes']

    # cutoff = tools['cutoff']
    data_dict={}
    with torch.no_grad():
        for batch, (id_tuple, id_frag_list_tuple, seq_frag_list_tuple, target_frag_nplist_tuple, type_protein_pt_tuple, sample_weight_tuple,_,) in enumerate(dataloader):
            start_time = time()
            id_frags_list, seq_frag_tuple, target_frag_pt, type_protein_pt = make_buffer(id_frag_list_tuple, 
                                                                                         seq_frag_list_tuple, 
                                                                                         target_frag_nplist_tuple, 
                                                                                         type_protein_pt_tuple)
            encoded_seq=tokenize(tools, seq_frag_tuple)
            if type(encoded_seq)==dict:
                for k in encoded_seq.keys():
                    encoded_seq[k]=encoded_seq[k].to(tools['pred_device'])
            else:
                encoded_seq=encoded_seq.to(tools['pred_device'])

            # classification_head, motif_logits = tools['net'](encoded_seq, id_tuple, id_frags_list, seq_frag_tuple)
            classification_head, motif_logits = tools['net'](
                       encoded_seq,
                       id_tuple,id_frags_list,seq_frag_tuple) #for test_loop always used None and False!

            m = torch.nn.Softmax(dim=1)  #torch.nn.Sigmoid()
            motif_logits = m(motif_logits) #batch, 9, len
            classification_head =  torch.nn.Sigmoid()(classification_head) #batch,9

            x_frag = np.array(motif_logits.cpu())   #[batch, head, seq]
            x_pro = np.array(classification_head.cpu()) #[sample, n]
            y_frag = np.array(target_frag_pt.cpu())    #[batch, head, seq]
            y_pro = np.array(type_protein_pt.cpu()) #[sample, n]
            for i in range(len(id_frags_list)):
                id_protein=id_frags_list[i].split('@')[0]
                j= id_tuple.index(id_protein)
                if id_protein in data_dict.keys():
                    data_dict[id_protein]['id_frag'].append(id_frags_list[i])
                    data_dict[id_protein]['seq_frag'].append(seq_frag_tuple[i])
                    data_dict[id_protein]['target_frag'].append(y_frag[i])     #[[head, seq], ...]
                    data_dict[id_protein]['motif_logits'].append(x_frag[i])    #[[head, seq], ...]
                else:
                    data_dict[id_protein]={}
                    data_dict[id_protein]['id_frag']=[id_frags_list[i]]
                    data_dict[id_protein]['seq_frag']=[seq_frag_tuple[i]]
                    data_dict[id_protein]['target_frag']=[y_frag[i]]
                    data_dict[id_protein]['motif_logits']=[x_frag[i]]
                    data_dict[id_protein]['type_pred']=x_pro[j]
                    data_dict[id_protein]['type_target']=y_pro[j]
            end_time = time()
            print("One batch time: "+ str(end_time-start_time))
        print("!@# data_dict size "+str(len(data_dict)))

        start_time = time()
        data_dict = frag2protein(data_dict, tools)
        end_time = time()
        print("frag ensemble time: "+ str(end_time-start_time))

        start_time = time()
        # result_pro, motif_pred, result_id = get_prediction(n, data_dict)  # result_pro = [sample_size, class_num], sample order same as result_id  
        #                                                                                  # motif_pred = class_num dictionaries with protein id as keys
        #                                                                                  # result_id = [sample_size], protein ids
        # print("!@# result id size "+str(len(result_id)))

        end_time = time()
        print("get prediction time: "+ str(end_time-start_time))
        # present(tools, result_pro, motif_pred, result_id)

        # get_evaluation(tools, data_dict, constrain)
        # print("!@# data_dict size "+str(len(data_dict)))
        # get_individual_scores(tools, data_dict, constrain)

        start_time = time()
        # present3(tools, result_pro, motif_pred, result_id, data_dict)
        end_time = time()
        print("present time: "+ str(end_time-start_time))
        return data_dict
        



        
class LocalizationDataset_pred(Dataset):
    def __init__(self, samples, configs):
        # self.label_to_index = {"Other": 0, "SP": 1, "MT": 2, "CH": 3, "TH": 4}
        # self.index_to_label = {0: "Other", 1: "SP", 2: "MT", 3: "CH", 4: "TH"}
        # self.transform = transform
        # self.target_transform = target_transform
        # self.cs_transform = cs_transform
        self.samples = samples
        self.n = configs.encoder.num_classes
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        id, id_frag_list, seq_frag_list = self.samples[idx]
        return id, id_frag_list, seq_frag_list

def custom_collate_pred(batch):
    id, id_frags, fragments = zip(*batch)
    return id, id_frags, fragments

def split_protein_sequence_pred(prot_id, sequence, configs):
    fragment_length = configs.encoder.max_len - 2
    overlap = configs.encoder.frag_overlap
    fragments = []
    id_frags = []
    sequence_length = len(sequence)
    start = 0
    ind=0

    while start < sequence_length:
        end = start + fragment_length
        if end > sequence_length:
            end = sequence_length
        fragment = sequence[start:end]
        fragments.append(fragment)
        id_frags.append(prot_id+"@"+str(ind))
        ind+=1
        if start + fragment_length > sequence_length:
            break
        start += fragment_length - overlap

    return id_frags, fragments

def prepare_samples_pred(fasta_file, configs):
    # label2idx = {"Other":0,
    #          "ER": 1, 
    #          "Peroxisome": 2, 
    #          "Mitochondrion": 3, 
    #          "SIGNAL": 4, 
    #          "Nucleus": 5,
    #          "Nucleus_export": 6, 
    #          "chloroplast": 7, 
    #          "Thylakoid": 8}
    samples = []
    with open(fasta_file) as f:
        line=f.readline()
        while line!="":
            content=line.strip()
            if content[0]==">":
                prot_id = content[1:]
            else:
                seq = content
                id_frag_list, seq_frag_list = split_protein_sequence_pred(prot_id, seq, configs)
                samples.append((prot_id, id_frag_list, seq_frag_list))
            line=f.readline()
        
    return samples






def prepare_dataloaders_pred(configs, input_file, with_label):
    # id_to_seq = prot_id_to_seq(seq_file)
    random.seed(configs.fix_seed)
    if with_label:
        samples = prepare_samples(input_file,configs)
        random.shuffle(samples)
        dataset = LocalizationDataset(samples, configs=configs,mode = "test")
        pred_dataloader = DataLoader(dataset, batch_size=configs.predict_settings.batch_size, shuffle=True, collate_fn=custom_collate)
    else:
        samples = prepare_samples_pred(input_file,configs)
        random.shuffle(samples)
        dataset = LocalizationDataset_pred(samples, configs=configs)
        pred_dataloader = DataLoader(dataset, batch_size=configs.predict_settings.batch_size, shuffle=False, collate_fn=custom_collate_pred)
    
    # Shuffle the list
    # print(train_dataset)
    return pred_dataloader


def prepare_pred_dir(configs, output_dir, model_dir):
    curdir_path=os.getcwd()
    checkpoint_file = os.path.abspath(model_dir)
    result_path = os.path.abspath(output_dir)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    return result_path, checkpoint_file, curdir_path

def prepare_ensemble_dir(configs, output_dir, model_dir, foldnum):
    curdir_path=os.getcwd()
    checkpoint_file = os.path.abspath(os.path.join(model_dir, "fold"+str(int(foldnum)),"checkpoints","best_model.pth"))
    result_path = os.path.abspath(output_dir)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    return result_path, checkpoint_file, curdir_path


def main(config_dict, input_file, output_dir, model_dir, with_label, constrain, ensemble):
    configs = load_configs(config_dict, None)
    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()

    start_time = time()
    dataloader = prepare_dataloaders_pred(configs, input_file, with_label)
    end_time = time()
    print("dataloader time: "+ str(end_time-start_time))
    print("!@# dataloader size  "+str(len(dataloader.dataset)))

    if not ensemble:
        result_path, checkpoint_file, curdir_path = prepare_pred_dir(configs, output_dir, model_dir)
        start_time = time()
        tokenizer=prepare_tokenizer(configs, curdir_path)
        encoder=prepare_models(configs, '', curdir_path)
        model_checkpoint = torch.load(checkpoint_file, map_location='cpu')
        encoder.model.load_state_dict(model_checkpoint['shared_model'])
        for class_index in range(1,9):    
            encoder.ParallelDecoders.decoders[class_index-1].load_state_dict(model_checkpoint['task_'+str(class_index)])
        
        # encoder.load_state_dict(model_checkpoint['model_state_dict'])
        end_time = time()
        print("model loading time: "+ str(end_time-start_time))
    
        tools = {
            'frag_overlap': configs.encoder.frag_overlap,
            # 'AA_cutoffs': configs.predict_settings.AA_cutoffs,
            # 'prot_cutoffs': configs.predict_settings.prot_cutoffs,
            'composition': configs.encoder.composition, 
            'max_len': configs.encoder.max_len,
            'tokenizer': tokenizer,
            'prm4prmpro': configs.encoder.prm4prmpro,
            'net': encoder,
            'pred_device': configs.predict_settings.device,
            'pred_batch_size': configs.predict_settings.batch_size,
            'result_path': result_path,
            'num_classes': configs.encoder.num_classes
        }
        
        start_time = time()
        if with_label:
            data_dict=predict_withlabel(dataloader, tools, constrain)

            get_evaluation(tools, data_dict, constrain)

            get_individual_scores(tools, data_dict, constrain)
        else:
            data_dict=predict(dataloader, tools)
            # print(data_dict)
            id2pred2sig, id2seq = present_single(tools, data_dict)
            json_data = convert_to_json(id2pred2sig, id2seq)
            output_file = os.path.join(tools['result_path'],"pred_results.json")
            # print(json_data)
            with open(output_file, "w") as file:
                file.write(json_data)


        end_time = time()
        print("predicting time: "+ str(end_time-start_time))

    
        torch.cuda.empty_cache()
    else:
        data_dict_list=[]
        for foldnum in range(1,6):
            result_path, checkpoint_file, curdir_path = prepare_ensemble_dir(configs, output_dir, model_dir,foldnum)
            start_time = time()
            tokenizer=prepare_tokenizer(configs, curdir_path)
            encoder=prepare_models(configs, '', curdir_path)
            model_checkpoint = torch.load(checkpoint_file, map_location='cpu')
            encoder.model.load_state_dict(model_checkpoint['shared_model'])
            for class_index in range(1,9):    
                encoder.ParallelDecoders.decoders[class_index-1].load_state_dict(model_checkpoint['task_'+str(class_index)])
            
            # encoder.load_state_dict(model_checkpoint['model_state_dict'])
            end_time = time()
            print("model loading time: "+ str(end_time-start_time))
        
            tools = {
                'frag_overlap': configs.encoder.frag_overlap,
                # 'AA_cutoffs': configs.predict_settings.AA_cutoffs,
                # 'prot_cutoffs': configs.predict_settings.prot_cutoffs,
                'composition': configs.encoder.composition, 
                'max_len': configs.encoder.max_len,
                'tokenizer': tokenizer,
                'prm4prmpro': configs.encoder.prm4prmpro,
                'net': encoder,
                'pred_device': configs.predict_settings.device,
                'pred_batch_size': configs.predict_settings.batch_size,
                'result_path': result_path,
                'num_classes': configs.encoder.num_classes
            }
            
            start_time = time()
            if with_label:
                data_dict = predict_withlabel(dataloader, tools, constrain)
                data_dict_list.append(data_dict)
            else:
                data_dict = predict(dataloader, tools)
                data_dict_list.append(data_dict)
            end_time = time()
            print("predicting time: "+ str(end_time-start_time))
        
            torch.cuda.empty_cache()
        
        data_dict={}
        for id_protein in data_dict_list[0].keys():
            data_dict[id_protein] = {}
            data_dict[id_protein]['seq_protein']=data_dict_list[0][id_protein]['seq_protein']

            list_of_arrays = [data_dict_list[i][id_protein]['motif_logits_protein'] for i in range(5)]
            stacked_array = np.stack(list_of_arrays, axis=0)
            average_array = np.mean(stacked_array, axis=0)
            data_dict[id_protein]['motif_logits_protein'] = average_array

            list_of_arrays = [data_dict_list[i][id_protein]['type_pred'] for i in range(5)]
            stacked_array = np.stack(list_of_arrays, axis=0)
            average_array = np.mean(stacked_array, axis=0)
            data_dict[id_protein]['type_pred'] = average_array

            if with_label:
                data_dict[id_protein]['motif_target_protein']=data_dict_list[0][id_protein]['motif_target_protein']
                data_dict[id_protein]['type_target']=data_dict_list[0][id_protein]['type_target']
            
        if with_label:
            get_evaluation(tools, data_dict, constrain)
            get_individual_scores(tools, data_dict, constrain)
        else:
            id2pred2sig, id2seq = present_single(tools, data_dict)
            json_data = convert_to_json(id2pred2sig, id2seq)
            output_file = os.path.join(tools['result_path'],"pred_results.json")
            # print(json_data)
            with open(output_file, "w") as file:
                file.write(json_data)





            




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CPM')
    parser.add_argument("--config_path", help="The location of config file", default='./configs/config_new_0wt2.yaml')
    parser.add_argument("--input_file", help="The location of input fasta file")
    parser.add_argument("--output_dir", help="The dir location of output")
    parser.add_argument("--model_dir", help="The dir location of trained model", default='./results/models/5fold')
    # parser.add_argument('--with_label', action='store_true', help='Set the flag to true')
    # parser.add_argument('--constrain', action='store_true', help='If use classification cutoff as constrain')
    # parser.add_argument('--ensemble', action='store_true', help='If use ensemble model')
    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path) as file:
        config_dict = yaml.full_load(file)
    
    input_file = args.input_file
    output_dir = args.output_dir
    model_dir = args.model_dir
    # with_label = args.with_label
    # constrain = args.constrain
    # ensemble = args.ensemble
    with_label = False
    constrain = True
    ensemble = True
    

    main(config_dict, input_file, output_dir, model_dir, with_label, constrain, ensemble)

#use case

#python predict.py --config_path ./results_log/D0614_2T2loss/P02_5/config0614_E15_T02_T5.yaml --input_file ./test_data_EC269_fold1.csv --output_dir ./results_log/D0614_2T2loss/P02_5/results_fold0 --model_dir ./results_log/D0614_2T2loss/P02_5/best_model_02_5.pth --with_label --constrain
#python predict.py --config_path ./results_log/D0614_2T2loss/P02_5/config0614_E15_T02_T5.yaml --input_file ./test_data_1_ECO269_nodupliTargetp.csv --output_dir ./results_log/D0614_2T2loss/P02_5/results_fold0 --model_dir ./results_log/D0614_2T2loss/P02_5/best_model_02_5.pth --with_label --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_targetp\test_targetp_rmtrain_fold1.csv --output_dir ./outputs/test_targetp/fold0 --model_dir ./results\residue_weight_savesepearte\5fold\fold1\checkpoints\best_model.pth --with_label --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_1_ECO269_INSPfilter.csv --output_dir ./outputs/INSP/fold0 --model_dir ./results\residue_weight_savesepearte\5fold\fold1\checkpoints\best_model.pth --with_label --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_2_ECO269_INSPfilter.csv --output_dir ./outputs/INSP/fold1 --model_dir ./results\residue_weight_savesepearte\5fold\fold2\checkpoints\best_model.pth --with_label --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_3_ECO269_INSPfilter.csv --output_dir ./outputs/INSP/fold2 --model_dir ./results\residue_weight_savesepearte\5fold\fold3\checkpoints\best_model.pth --with_label --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_4_ECO269_INSPfilter.csv --output_dir ./outputs/INSP/fold3 --model_dir ./results\residue_weight_savesepearte\5fold\fold4\checkpoints\best_model.pth --with_label --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_0_ECO269_INSPfilter.csv --output_dir ./outputs/INSP/fold4 --model_dir ./results\residue_weight_savesepearte\5fold\fold5\checkpoints\best_model.pth --with_label --constrain

#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_1_ECO269_INSPfilter.csv --output_dir ./outputs/newINSP/fold0 --model_dir ./results\residue_weight_savesepearte\5fold\fold1\checkpoints\best_model.pth --with_label --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_1_ECO269_INSPfilter.csv --output_dir ./outputs/newINSP/ensemblefold0 --model_dir ./results\residue_weight_savesepearte\5fold --with_label --constrain --ensemble

#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_fasta.fasta --output_dir ./outputs/test/fold4 --model_dir ./results\residue_weight_savesepearte\5fold\fold5\checkpoints\best_model.pth --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_fasta.fasta --output_dir ./outputs/test/pred_ensemble --model_dir ./results\residue_weight_savesepearte\5fold --constrain --ensemble

#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_ECO269_fold1.csv --output_dir ./outputs/test_269/fold0 --model_dir ./results\residue_weight_savesepearte\5fold\fold1\checkpoints\best_model.pth --with_label --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_ECO269_fold2.csv --output_dir ./outputs/test_269/fold1 --model_dir ./results\residue_weight_savesepearte\5fold\fold2\checkpoints\best_model.pth --with_label --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_ECO269_fold3.csv --output_dir ./outputs/test_269/fold2 --model_dir ./results\residue_weight_savesepearte\5fold\fold3\checkpoints\best_model.pth --with_label --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_ECO269_fold4.csv --output_dir ./outputs/test_269/fold3 --model_dir ./results\residue_weight_savesepearte\5fold\fold4\checkpoints\best_model.pth --with_label --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_ECO269_fold0.csv --output_dir ./outputs/test_269/fold4 --model_dir ./results\residue_weight_savesepearte\5fold\fold5\checkpoints\best_model.pth --with_label --constrain

#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_deeploc2_rmtrain_fold1.csv --output_dir ./outputs/deeploc2/fold0 --model_dir ./results\residue_weight_savesepearte\5fold\fold1\checkpoints\best_model.pth --with_label --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_deeploc2_rmtrain_fold2.csv --output_dir ./outputs/deeploc2/fold1 --model_dir ./results\residue_weight_savesepearte\5fold\fold2\checkpoints\best_model.pth --with_label --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_deeploc2_rmtrain_fold3.csv --output_dir ./outputs/deeploc2/fold2 --model_dir ./results\residue_weight_savesepearte\5fold\fold3\checkpoints\best_model.pth --with_label --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_deeploc2_rmtrain_fold4.csv --output_dir ./outputs/deeploc2/fold3 --model_dir ./results\residue_weight_savesepearte\5fold\fold4\checkpoints\best_model.pth --with_label --constrain
#python predict.py --config_path ./results\residue_weight_savesepearte\5fold\fold1\config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_deeploc2_rmtrain_fold0.csv --output_dir ./outputs/deeploc2/fold4 --model_dir ./results\residue_weight_savesepearte\5fold\fold5\checkpoints\best_model.pth --with_label --constrain

#python predict.py --config_path ./configs/config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_Peroxisome_fold1.csv --output_dir ./outputs/peroxisome/fold0 --model_dir ./results/models/5fold/fold1/checkpoints/best_model.pth --with_label --constrain
#python predict.py --config_path ./configs/config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_Peroxisome_fold2.csv --output_dir ./outputs/peroxisome/fold1 --model_dir ./results/models/5fold/fold2/checkpoints/best_model.pth --with_label --constrain
#python predict.py --config_path ./configs/config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_Peroxisome_fold3.csv --output_dir ./outputs/peroxisome/fold2 --model_dir ./results/models/5fold/fold3/checkpoints/best_model.pth --with_label --constrain
#python predict.py --config_path ./configs/config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_Peroxisome_fold4.csv --output_dir ./outputs/peroxisome/fold3 --model_dir ./results/models/5fold/fold4/checkpoints/best_model.pth --with_label --constrain
#python predict.py --config_path ./configs/config0824_multiCNN-linear_M0.3_M0.1_activation.yaml --input_file ./test_data_Peroxisome_fold0.csv --output_dir ./outputs/peroxisome/fold4 --model_dir ./results/models/5fold/fold5/checkpoints/best_model.pth --with_label --constrain

#python predict.py --config_path ./configs/config_new_0wt2.yaml --input_file ./test_deeploc2_signalData_rmtrain_fold3.csv --output_dir ./outputs/pred_deeploc_sig_testfold3 --model_dir ./results/residue_weight_savesepearte/5fold/fold3/checkpoints/best_model.pth --with_label --constrain