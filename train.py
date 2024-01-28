import torch
torch.manual_seed(0)
from torch.cuda.amp import GradScaler, autocast
import argparse
import os
import yaml
import numpy as np
import torchmetrics
from time import time
from data import *
from model import *
from utils import *
from sklearn.metrics import roc_auc_score,average_precision_score,matthews_corrcoef,recall_score,precision_score,f1_score
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# TODO: change the path to esm models.
torch.hub.set_dir("/home/zengs/zengs_data/torch_hub")


def train_loop(tools):
    # accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=tools['num_classes'], average='macro')
    # f1_score = torchmetrics.F1Score(num_classes=tools['num_classes'], average='macro', task="multiclass")
    # accuracy.to(tools['train_device'])
    # f1_score.to(tools["train_device"])
    tools["optimizer"].zero_grad()
    scaler = GradScaler()
    size = len(tools['train_loader'].dataset)
    num_batches = len(tools['train_loader'])
    train_loss=0
    # cs_num=np.zeros(9)
    # cs_correct=np.zeros(9)
    # type_num=np.zeros(10)
    # type_correct=np.zeros(10)
    # cutoff=0.5
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.train().cuda()
    tools['net'].train().to(tools['train_device'])
    for batch, (id, seq_frag, target_frag, sample_weight) in enumerate(tools['train_loader']):
        with autocast():
            # Compute prediction and loss
            encoded_seq=tokenize(tools, seq_frag)
            if type(encoded_seq)==dict:
                for k in encoded_seq.keys():
                    encoded_seq[k]=encoded_seq[k].to(tools['train_device'])
            else:
                encoded_seq=encoded_seq.to(tools['train_device'])
            motif_logits = tools['net'](encoded_seq)
            # if cs_probab.size()[1]<200:
            #     zero_pad=200-cs_probab.size()[1]
            #     additional_elements = torch.zeros([cs_probab.size()[0],zero_pad]).to(tools['train_device'])
            #     cs_probab = torch.cat((cs_probab, additional_elements), dim=1)
            # elif cs_probab.size()[1]>200:
            #     cs_probab = cs_probab[:,:200]

            # w=torch.ones([9,1,1])*5
            # print(seq_frag)
            # print(motif_logits.shape, target_frag.shape, encoded_seq.shape)
            weighted_loss_sum = tools['loss_function'](motif_logits, target_frag.to(tools['train_device']))
            # losses=[]
            # for head in range(motif_logits.size()[1]):
            #     loss = tools['loss_function'](motif_logits[:, head, :], target_frag[:,head].to(tools['train_device']))
            #     weighted_loss = loss * sample_weight.unsqueeze(1).to(tools['train_device'])
            #     losses.append(torch.mean(weighted_loss))
            
            # weighted_loss_sum = sum(losses)
            train_loss += weighted_loss_sum.item()

        # Backpropagation
        scaler.scale(weighted_loss_sum).backward()
        scaler.step(tools['optimizer'])
        scaler.update()
        tools['scheduler'].step()
        if batch % 30 == 0:
            loss, current = weighted_loss_sum.item(), (batch + 1) * len(seq_frag)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            customlog(tools["logfilepath"], f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")
    # epoch_acc = accuracy.compute().cpu().item()
    # epoch_f1 = f1_score.compute().cpu().item()
    epoch_loss = train_loss/num_batches
    # acc_cs = cs_correct / cs_num
    customlog(tools["logfilepath"], f" loss: {epoch_loss:>5f}\n")
    # customlog(tools["logfilepath"], f" accuracy_macro: {epoch_acc:>5f}\n")
    # customlog(tools["logfilepath"], f" f1_macro: {epoch_f1:>5f}\n")
    # customlog(tools["logfilepath"], f" acc_cs: {acc_cs:>5f}\n")
    # Reset metrics at the end of epoch
    # accuracy.reset()
    # f1_score.reset()
    return epoch_loss


# def train_on_batch(model, seq, label, cs):
#     with autocast():
#         # Compute prediction and loss
#         type_probab, cs_probab = model(seq)
#         if cs_probab.size()[1]<200:
#           zero_pad=200-cs_probab.size()[1]
#           additional_elements = torch.zeros([cs_probab.size()[0],zero_pad]).to(device)
#           cs_probab = torch.cat((cs_probab, additional_elements), dim=1)
#         # loss1 = loss_fn(type_probab, label.cuda())
#         loss1 = loss_fn(type_probab, label.to(device))
#         mask =  cs[:, 0] != 1
#         # loss2 = loss_fn(cs_probab[mask], cs[mask].cuda())
#         loss2 = loss_fn(cs_probab[mask], cs[mask].to(device))
#         loss = loss1 + loss2
#         return loss



def test_loop(tools, dataloader):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.eval().cuda()
    tools['net'].eval().to(tools["valid_device"])
    # accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=tools['num_classes'], average=None)
    # macro_f1_score = torchmetrics.F1Score(num_classes=tools['num_classes'], average='macro', task="multiclass")
    # f1_score = torchmetrics.F1Score(num_classes=tools['num_classes'], average=None, task="multiclass")
    # accuracy.to(tools["valid_device"])
    # macro_f1_score.to(tools["valid_device"])
    # f1_score.to(tools['valid_device'])
    num_batches = len(dataloader)
    test_loss=0
    # cs_num=0
    # cs_correct=0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch, (id, seq_frag, target_frag, sample_weight) in enumerate(dataloader):
            encoded_seq=tokenize(tools, seq_frag)
            if type(encoded_seq)==dict:
                for k in encoded_seq.keys():
                    encoded_seq[k]=encoded_seq[k].to(tools['valid_device'])
            else:
                encoded_seq=encoded_seq.to(tools['valid_device'])
            motif_logits = tools["net"](encoded_seq)
            # if cs_probab.size()[1]<200:
            #     zero_pad=200-cs_probab.size()[1]
            #     additional_elements = torch.zeros([cs_probab.size()[0],zero_pad]).to(tools['train_device'])
            #     cs_probab = torch.cat((cs_probab, additional_elements), dim=1)
            # elif cs_probab.size()[1]>200:
            #     cs_probab = cs_probab[:,:200]

            weighted_loss_sum = tools['loss_function'](motif_logits, target_frag.to(tools['train_device']))
            # losses=[]
            # for head in range(motif_logits.size()[1]):
            #     loss = tools['loss_function'](motif_logits[:, head, :], target_frag[:,head].to(tools['valid_device']))
            #     weighted_loss = loss * sample_weight.unsqueeze(1).to(tools['valid_device'])
            #     losses.append(torch.mean(weighted_loss))
            # weighted_loss_sum = sum(losses)

            test_loss += weighted_loss_sum.item()
            # label = torch.argmax(label_1hot, dim=1)
            # type_pred = torch.argmax(type_probab, dim=1)
            # accuracy.update(type_pred.detach(), label.detach().to(tools['valid_device']))
            # macro_f1_score.update(type_pred.detach(), label.detach().to(tools['valid_device']))
            # f1_score.update(type_pred.detach(), label.detach().to(tools['valid_device']))

        test_loss = test_loss / num_batches
        # epoch_acc = np.array(accuracy.compute().cpu())
        # epoch_macro_f1 = macro_f1_score.compute().cpu().item()
        # epoch_f1 = np.array(f1_score.compute().cpu())
        # acc_cs = cs_correct / cs_num
        customlog(tools["logfilepath"], f" loss: {test_loss:>5f}\n")
        # customlog(tools["logfilepath"], f" accuracy: "+str(epoch_acc)+"\n")
        # customlog(tools["logfilepath"], f" f1: "+str(epoch_f1)+"\n")
        # customlog(tools["logfilepath"], f" f1_macro: {epoch_macro_f1:>5f}\n")
        # customlog(tools["logfilepath"], f" acc_cs: {acc_cs:>5f}\n")
        # Reset metrics at the end of epoch
        # accuracy.reset()
        # macro_f1_score.reset()
        # f1_score.reset()
    return test_loss

def evaluate(tools, dataloader):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.eval().cuda()
    model_path = os.path.join(tools['checkpoint_path'], f'best_model.pth')
    model_checkpoint = torch.load(model_path, map_location='cpu')
    tools['net'].load_state_dict(model_checkpoint['model_state_dict'])
    tools['net'].eval().to(tools["valid_device"])
    n=tools['num_classes']

    # accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=tools['num_classes'], average=None)
    # macro_f1_score = torchmetrics.F1Score(num_classes=tools['num_classes'], average='macro', task="multiclass")
    # f1_score = torchmetrics.F1Score(num_classes=tools['num_classes'], average=None, task="multiclass")
    # accuracy.to(tools["valid_device"])
    # macro_f1_score.to(tools["valid_device"])
    # f1_score.to(tools['valid_device'])
    num_batches = len(dataloader)
    TP_num=np.zeros(n)
    FP_num=np.zeros(n)
    FN_num=np.zeros(n)
    #Intersection over Union (IoU) or Jaccard Index
    IoU = np.zeros(n)
    Negtive_detect_num=0
    Negtive_num=0
    size = len(tools['train_loader'].dataset)
    cutoff = tools['cutoff']
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch, (id, seq_frag, target_frag, sample_weight) in enumerate(dataloader):
            encoded_seq=tokenize(tools, seq_frag)
            if type(encoded_seq)==dict:
                for k in encoded_seq.keys():
                    encoded_seq[k]=encoded_seq[k].to(tools['valid_device'])
            else:
                encoded_seq=encoded_seq.to(tools['valid_device'])
            motif_logits = tools["net"](encoded_seq)
            m=torch.nn.Sigmoid()
            motif_logits = m(motif_logits)
            # if cs_probab.size()[1]<200:
            #     zero_pad=200-cs_probab.size()[1]
            #     additional_elements = torch.zeros([cs_probab.size()[0],zero_pad]).to(tools['train_device'])
            #     cs_probab = torch.cat((cs_probab, additional_elements), dim=1)
            # elif cs_probab.size()[1]>200:
            #     cs_probab = cs_probab[:,:200]
            # losses=[]

            for head in range(motif_logits.size()[1]):
                x = np.array(motif_logits[:, head, :].cpu())
                y = np.array(target_frag[:,head].cpu())
                Negtive_num += sum(np.max(y, axis=1)==0)
                Negtive_detect_num += sum((np.max(y, axis=1)==0) * (np.max(x>=cutoff, axis=1)==1))
                TP_num[head] += np.sum((x>=cutoff) * (y==1))
                FP_num[head] += np.sum((x>=cutoff) * (y==0))
                FN_num[head] += np.sum((x<cutoff) * (y==1))

            # label = torch.argmax(label_1hot, dim=1)
            # type_pred = torch.argmax(type_probab, dim=1)
            # accuracy.update(type_pred.detach(), label.detach().to(tools['valid_device']))
            # macro_f1_score.update(type_pred.detach(), label.detach().to(tools['valid_device']))
            # f1_score.update(type_pred.detach(), label.detach().to(tools['valid_device']))

        for head in range(n):
            IoU[head] = TP_num[head] / (TP_num[head] + FP_num[head] + FN_num[head])
        Negtive_detect_ratio = Negtive_detect_num / Negtive_num
        # epoch_acc = np.array(accuracy.compute().cpu())
        # epoch_macro_f1 = macro_f1_score.compute().cpu().item()
        # epoch_f1 = np.array(f1_score.compute().cpu())
        # acc_cs = cs_correct / cs_num
        customlog(tools["logfilepath"], f" Jaccard Index: "+ str(IoU)+"\n")
        customlog(tools["logfilepath"], f" Negtive detect ratio: {Negtive_detect_ratio:>5f}\n")
        # customlog(tools["logfilepath"], f" accuracy: "+str(epoch_acc)+"\n")
        # customlog(tools["logfilepath"], f" f1: "+str(epoch_f1)+"\n")
        # customlog(tools["logfilepath"], f" f1_macro: {epoch_macro_f1:>5f}\n")
        # customlog(tools["logfilepath"], f" acc_cs: {acc_cs:>5f}\n")
        # Reset metrics at the end of epoch
        # accuracy.reset()
        # macro_f1_score.reset()
        # f1_score.reset()
    return 0

def frag2protein(data_dict, tools):
    overlap=tools['frag_overlap']
    # no_overlap=tools['max_len']-2-overlap
    for id_protein in data_dict.keys():
        id_frag_list = data_dict[id_protein]['id_frag']
        seq_protein=""
        motif_logits_protein=np.array([])
        target_protein=np.array([])
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
                target_protein=target_frag[:,:l]
            else:
                seq_protein = seq_protein + seq_frag[overlap:]
                x_overlap = np.maximum(motif_logits_protein[:,-overlap:], motif_logits_frag[:,:overlap])
                motif_logits_protein = np.concatenate((motif_logits_protein[:,:-overlap], x_overlap, motif_logits_frag[:,overlap:l]),axis=1)
                target_protein = np.concatenate((target_protein, target_frag[:,overlap:l]), axis=1)
        data_dict[id_protein]['seq_protein']=seq_protein
        data_dict[id_protein]['motif_logits_protein']=motif_logits_protein
        data_dict[id_protein]['target_protein']=target_protein
    return data_dict

def evaluate_protein(dataloader, tools):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.eval().cuda()
    model_path = os.path.join(tools['checkpoint_path'], f'best_model.pth')
    model_checkpoint = torch.load(model_path, map_location='cpu')
    tools['net'].load_state_dict(model_checkpoint['model_state_dict'])
    tools['net'].eval().to(tools["valid_device"])
    n=tools['num_classes']

    

    cutoff = tools['cutoff']
    data_dict={}
    with torch.no_grad():
        for batch, (id, seq_frag, target_frag, sample_weight) in enumerate(dataloader):
            
            encoded_seq=tokenize(tools, seq_frag)
            if type(encoded_seq)==dict:
                for k in encoded_seq.keys():
                    encoded_seq[k]=encoded_seq[k].to(tools['valid_device'])
            else:
                encoded_seq=encoded_seq.to(tools['valid_device'])
            motif_logits = tools["net"](encoded_seq)
            m=torch.nn.Sigmoid()
            motif_logits = m(motif_logits)

            x = np.array(motif_logits.cpu())   #[batch, head, seq]
            y = np.array(target_frag.cpu())    #[batch, head, seq]
            for i in range(len(id)):
                id_protein=id[i].split('@')[0]
                if id_protein in data_dict.keys():
                    data_dict[id_protein]['id_frag'].append(id[i])
                    data_dict[id_protein]['seq_frag'].append(seq_frag[i])
                    data_dict[id_protein]['target_frag'].append(y[i])     #[[head, seq], ...]
                    data_dict[id_protein]['motif_logits'].append(x[i])    #[[head, seq], ...]
                else:
                    data_dict[id_protein]={}
                    data_dict[id_protein]['id_frag']=[id[i]]
                    data_dict[id_protein]['seq_frag']=[seq_frag[i]]
                    data_dict[id_protein]['target_frag']=[y[i]]
                    data_dict[id_protein]['motif_logits']=[x[i]]

        data_dict = frag2protein(data_dict, tools)

        IoU_difcut=np.zeros([n, 9])
        FDR_frag_difcut=np.zeros([1,9])
        IoU_pro_difcut=np.zeros([n, 9])
        FDR_pro_difcut=np.zeros([1,9])
        result_pro_difcut=np.zeros([n,6,9])
        classname=["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
             "SIGNAL", "chloroplast", "Thylakoid"]
        criteria=["roc_auc_score", "average_precision_score", "matthews_corrcoef",
              "recall_score", "precision_score", "f1_score"]

        cutoffs=[x / 10 for x in range(1, 10)]
        cut_dim=0
        for cutoff in cutoffs:
            scores=get_scores(tools, cutoff, n, data_dict)
            IoU_difcut[:,cut_dim]=scores['IoU']
            # IoU_difcut[:,cut_dim]=np.array([float("{:.3f}".format(i)) for i in scores['IoU']])
            FDR_frag_difcut[:,cut_dim]=scores['FDR_frag']
            # FDR_frag_difcut[:,cut_dim]=float("{:.3f}".format(scores['FDR_frag']))
            IoU_pro_difcut[:,cut_dim]=scores['IoU_pro']
            # IoU_pro_difcut[:,cut_dim]=np.array([float("{:.3f}".format(i)) for i in scores['IoU_pro']])
            FDR_pro_difcut[:,cut_dim]=scores['FDR_pro']
            # FDR_pro_difcut[:,cut_dim]=float("{:.3f}".format(scores['FDR_pro']))
            result_pro_difcut[:,:,cut_dim]=scores['result_pro']
            # result_pro_difcut[:,:,cut_dim]=np.array([float("{:.3f}".format(i)) for i in scores['result_pro'].reshape(-1)]).reshape(scores['result_pro'].shape) 
            cut_dim+=1

        customlog(tools["logfilepath"], f"===========================================\n")
        # classname=["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
        #          "dual", "SIGNAL", "chloroplast", "Thylakoid"]
        
        customlog(tools["logfilepath"], f" Jaccard Index (fragment): \n")
        IoU_difcut=pd.DataFrame(IoU_difcut,columns=cutoffs,index=classname)
        customlog(tools["logfilepath"], IoU_difcut.__repr__())
        # IoU_difcut.to_csv(tools["logfilepath"],mode='a',sep="\t") 
        customlog(tools["logfilepath"], f"===========================================\n")
        customlog(tools["logfilepath"], f" FDR (fragment): \n")
        FDR_frag_difcut=pd.DataFrame(FDR_frag_difcut,columns=cutoffs)
        customlog(tools["logfilepath"], FDR_frag_difcut.__repr__())
        # FDR_frag_difcut.to_csv(tools["logfilepath"],mode='a',sep="\t")
        customlog(tools["logfilepath"], f"===========================================\n")
        customlog(tools["logfilepath"], f" Jaccard Index (protein): \n")
        IoU_pro_difcut=pd.DataFrame(IoU_pro_difcut,columns=cutoffs,index=classname)
        customlog(tools["logfilepath"], IoU_pro_difcut.__repr__())
        # IoU_pro_difcut.to_csv(tools["logfilepath"],mode='a',sep="\t")
        customlog(tools["logfilepath"], f"===========================================\n")
        customlog(tools["logfilepath"], f" FDR (protein): \n")
        FDR_pro_difcut=pd.DataFrame(FDR_pro_difcut,columns=cutoffs)
        customlog(tools["logfilepath"], FDR_pro_difcut.__repr__())
        # FDR_pro_difcut.to_csv(tools["logfilepath"],mode='a',sep="\t")
        customlog(tools["logfilepath"], f"===========================================\n")
        for i in range(len(classname)):
            customlog(tools["logfilepath"], f" Class prediction performance ({classname[i]}): \n")
            tem = pd.DataFrame(result_pro_difcut[i],columns=cutoffs,index=criteria)
            customlog(tools["logfilepath"], tem.__repr__())
            # tem.to_csv(tools["logfilepath"],mode='a',sep="\t")

    

    
            # for head in range(motif_logits.size()[1]):
            #     x = np.array(motif_logits[:, head, :].cpu())
            #     y = np.array(target_frag[:,head].cpu())
            #     Negtive_num += sum(np.max(y, axis=1)==0)
            #     Negtive_detect_num += sum((np.max(y, axis=1)==0) * (np.max(x>=cutoff, axis=1)==1))
            #     TP_frag[head] += np.sum((x>=cutoff) * (y==1))
            #     FP_frag[head] += np.sum((x>=cutoff) * (y==0))
            #     FN_frag[head] += np.sum((x<cutoff) * (y==1))

                
            

        

        # for head in range(n):
        #     x_list=[]
        #     y_list=[]
        #     for id_protein in data_dict.keys():
        #         x = np.array(data_dict[id_protein]['motif_logits'])[:,head]   #[frag_num, seq]
        #         y = np.array(data_dict[id_protein]['target_protein'])[:,head] #[frag_num, seq]
        #         Negtive_num += sum(np.max(y, axis=1)==0)
        #         Negtive_detect_num += sum((np.max(y, axis=1)==0) * (np.max(x>=cutoff, axis=1)==1))
        #         TP_frag[head] += np.sum((x>=cutoff) * (y==1))
        #         FP_frag[head] += np.sum((x>=cutoff) * (y==0))
        #         FN_frag[head] += np.sum((x<cutoff) * (y==1))




        #         x = data_dict[id_protein]['motif_logits_protein'][head]  #[seq]
        #         y = data_dict[id_protein]['target_protein'][head]
        #         Negtive_pro += np.sum(np.max(y)==0)
        #         Negtive_detect_pro += np.sum((np.max(y)==0) * (np.max(x>=cutoff)==1))
        #         TP_pro[head] += np.sum((x>=cutoff) * (y==1))
        #         FP_pro[head] += np.sum((x>=cutoff) * (y==0))
        #         FN_pro[head] += np.sum((x<cutoff) * (y==1))
        #         x_list.append(np.max(x))
        #         y_list.append(np.max(y))
        #     pred=np.array(x_list)
        #     target=np.array(y_list)
        #     result_pro[head,0] = roc_auc_score(target, pred)
        #     result_pro[head,1] = average_precision_score(target, pred)
        #     result_pro[head,2] = matthews_corrcoef(target, pred>=cutoff)
        #     result_pro[head,3] = recall_score(target, pred>=cutoff)
        #     result_pro[head,4] = precision_score(target, pred>=cutoff)
        #     result_pro[head,5] = f1_score(target, pred>=cutoff)
  
    
        # for head in range(n):
        #     IoU[head] = TP_frag[head] / (TP_frag[head] + FP_frag[head] + FN_frag[head])
        #     IoU_pro[head] = TP_pro[head] / (TP_pro[head] + FP_pro[head] + FN_pro[head])
        # FDR_frag = Negtive_detect_num / Negtive_num
        # FDR_pro = Negtive_detect_pro / Negtive_pro

        # customlog(tools["logfilepath"], f"===========================================\n")
        # # classname=["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
        # #          "dual", "SIGNAL", "chloroplast", "Thylakoid"]
        # classname=["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
        #          "SIGNAL", "chloroplast", "Thylakoid"]
        # customlog(tools["logfilepath"], f" Jaccard Index (fragment): \n")
        # for i in range(len(classname)):
        #     customlog(tools["logfilepath"], classname[i]+ f": {IoU[i]:>5f}\n")
        # customlog(tools["logfilepath"], f"===========================================\n")
        # customlog(tools["logfilepath"], f" FDR (fragment): {FDR_frag:>5f}\n")
        # customlog(tools["logfilepath"], f"===========================================\n")
        # customlog(tools["logfilepath"], f" Jaccard Index (protein): \n")
        # for i in range(len(classname)):
        #     customlog(tools["logfilepath"], classname[i]+ f": {IoU_pro[i]:>5f}\n")
        # customlog(tools["logfilepath"], f"===========================================\n")
        # customlog(tools["logfilepath"], f" FDR (protein): {FDR_pro:>5f}\n")
        # customlog(tools["logfilepath"], f"===========================================\n")
        # criteria=["roc_auc_score", "average_precision_score", "matthews_corrcoef",
        #           "recall_score", "precision_score", "f1_score"]
        # for i in range(len(classname)):
        #      customlog(tools["logfilepath"], f" Class prediction performance ({classname[i]}): \n")
        #      for j in range(len(criteria)):
        #          customlog(tools["logfilepath"], criteria[j]+f": {result_pro[i,j]:>5f}\n")
        
        # performance={"IoU":IoU, "FDR_frag":FDR_frag, "IoU_pro":IoU_pro, "FDR_pro":FDR_pro,
        #              "result_pro":result_pro}

def get_scores(tools, cutoff, n, data_dict):
    TP_frag=np.zeros(n)
    FP_frag=np.zeros(n)
    FN_frag=np.zeros(n)
    #Intersection over Union (IoU) or Jaccard Index
    IoU = np.zeros(n)
    Negtive_detect_num=0
    Negtive_num=0

    TP_pro=np.zeros(n)
    FP_pro=np.zeros(n)
    FN_pro=np.zeros(n)
    IoU_pro = np.zeros(n)
    Negtive_detect_pro=0
    Negtive_pro=0
    result_pro=np.zeros([n,6])
    for head in range(n):
        x_list=[]
        y_list=[]
        for id_protein in data_dict.keys():
            x = np.array(data_dict[id_protein]['motif_logits'])[:,head]   #[frag_num, seq]
            y = np.array(data_dict[id_protein]['target_frag'])[:,head] #[frag_num, seq]
            Negtive_num += sum(np.max(y, axis=1)==0)
            Negtive_detect_num += sum((np.max(y, axis=1)==0) * (np.max(x>=cutoff, axis=1)==1))
            TP_frag[head] += np.sum((x>=cutoff) * (y==1))
            FP_frag[head] += np.sum((x>=cutoff) * (y==0))
            FN_frag[head] += np.sum((x<cutoff) * (y==1))

            x = data_dict[id_protein]['motif_logits_protein'][head]  #[seq]
            y = data_dict[id_protein]['target_protein'][head]
            Negtive_pro += np.sum(np.max(y)==0)
            Negtive_detect_pro += np.sum((np.max(y)==0) * (np.max(x>=cutoff)==1))
            TP_pro[head] += np.sum((x>=cutoff) * (y==1))
            FP_pro[head] += np.sum((x>=cutoff) * (y==0))
            FN_pro[head] += np.sum((x<cutoff) * (y==1))
            x_list.append(np.max(x))
            y_list.append(np.max(y))
        pred=np.array(x_list)
        target=np.array(y_list)
        result_pro[head,0] = roc_auc_score(target, pred)
        result_pro[head,1] = average_precision_score(target, pred)
        result_pro[head,2] = matthews_corrcoef(target, pred>=cutoff)
        result_pro[head,3] = recall_score(target, pred>=cutoff)
        result_pro[head,4] = precision_score(target, pred>=cutoff)
        result_pro[head,5] = f1_score(target, pred>=cutoff)


    for head in range(n):
        IoU[head] = TP_frag[head] / (TP_frag[head] + FP_frag[head] + FN_frag[head])
        IoU_pro[head] = TP_pro[head] / (TP_pro[head] + FP_pro[head] + FN_pro[head])
    FDR_frag = Negtive_detect_num / Negtive_num
    FDR_pro = Negtive_detect_pro / Negtive_pro
    # customlog(tools["logfilepath"], f"===========================================\n")
    # # classname=["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
    # #          "dual", "SIGNAL", "chloroplast", "Thylakoid"]
    # classname=["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
    #          "SIGNAL", "chloroplast", "Thylakoid"]
    # customlog(tools["logfilepath"], f" Jaccard Index (fragment): \n")
    # for i in range(len(classname)):
    #     customlog(tools["logfilepath"], classname[i]+ f": {IoU[i]:>5f}\n")
    # customlog(tools["logfilepath"], f"===========================================\n")
    # customlog(tools["logfilepath"], f" FDR (fragment): {FDR_frag:>5f}\n")
    # customlog(tools["logfilepath"], f"===========================================\n")
    # customlog(tools["logfilepath"], f" Jaccard Index (protein): \n")
    # for i in range(len(classname)):
    #     customlog(tools["logfilepath"], classname[i]+ f": {IoU_pro[i]:>5f}\n")
    # customlog(tools["logfilepath"], f"===========================================\n")
    # customlog(tools["logfilepath"], f" FDR (protein): {FDR_pro:>5f}\n")
    # customlog(tools["logfilepath"], f"===========================================\n")
    # criteria=["roc_auc_score", "average_precision_score", "matthews_corrcoef",
    #           "recall_score", "precision_score", "f1_score"]
    # for i in range(len(classname)):
    #      customlog(tools["logfilepath"], f" Class prediction performance ({classname[i]}): \n")
    #      for j in range(len(criteria)):
    #          customlog(tools["logfilepath"], criteria[j]+f": {result_pro[i,j]:>5f}\n")
    
    scores={"IoU":IoU, #[n]
            "FDR_frag":FDR_frag, #[1]
            "IoU_pro":IoU_pro, #[n]
            "FDR_pro":FDR_pro, #[1]
            "result_pro":result_pro} #[n, 6]
    return scores


def main(config_dict, valid_batch_number, test_batch_number):
    configs = load_configs(config_dict)
    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()
    curdir_path, result_path, checkpoint_path, logfilepath = prepare_saving_dir(configs)

    npz_file=os.path.join(curdir_path, "targetp_data.npz")
    seq_file=os.path.join(curdir_path, "idmapping_2023_08_25.tsv")

    customlog(logfilepath, f'use k-fold index: {valid_batch_number}\n')
    # dataloaders_dict = prepare_dataloaders(valid_batch_number, test_batch_number, npz_file, seq_file, configs)
    dataloaders_dict = prepare_dataloaders(configs, valid_batch_number, test_batch_number)
    customlog(logfilepath, "Done Loading data\n")
    
    if configs.encoder.composition=="official_esm_v2":
        encoder=prepare_models(configs,logfilepath, curdir_path)
        customlog(logfilepath, "Done initialize model\n")
        
        alphabet = encoder.model.alphabet
        tokenizer = alphabet.get_batch_converter(
            truncation_seq_length=configs.encoder.max_len
        )
    else:  
        tokenizer=prepare_tokenizer(configs, curdir_path)
        customlog(logfilepath, "Done initialize tokenizer\n")

        encoder=prepare_models(configs,logfilepath, curdir_path)
        customlog(logfilepath, "Done initialize model\n")
    
    
    optimizer, scheduler = prepare_optimizer(encoder, configs, len(dataloaders_dict["train"]), logfilepath)
    customlog(logfilepath, 'preparing optimizer is done\n')

    encoder, start_epoch = load_checkpoints(configs, optimizer, scheduler, logfilepath, encoder)

    # w=(torch.ones([9,1,1])*5).to(configs.train_settings.device)
    w= torch.tensor(configs.train_settings.loss_pos_weight, dtype=torch.float32).to(configs.train_settings.device)

    tools = {
        'frag_overlap': configs.encoder.frag_overlap,
        'cutoff': configs.valid_settings.cutoff,
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
        'optimizer': optimizer,
        # 'loss_function': torch.nn.CrossEntropyLoss(reduction="none"),
        'loss_function': torch.nn.BCEWithLogitsLoss(pos_weight=w, reduction="mean"),
        'checkpoints_every': configs.checkpoints_every,
        'scheduler': scheduler,
        'result_path': result_path,
        'checkpoint_path': checkpoint_path,
        'logfilepath': logfilepath,
        'num_classes': configs.encoder.num_classes
    }

    customlog(logfilepath, f'number of train steps per epoch: {len(tools["train_loader"])}\n')
    customlog(logfilepath, "Start training...\n")

    best_valid_loss=np.inf
    for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):
        tools['epoch'] = epoch
        print(f"Fold {valid_batch_number} Epoch {epoch}\n-------------------------------")
        customlog(logfilepath, f"Fold {valid_batch_number} Epoch {epoch} train...\n-------------------------------\n")
        start_time = time()
        train_loss= train_loop(tools)
        end_time = time()

        if epoch % configs.valid_settings.do_every == 0 and epoch != 0:
            customlog(logfilepath, f"Fold {valid_batch_number} Epoch {epoch} validation...\n-------------------------------\n")
            start_time = time()
            dataloader=tools["valid_loader"]
            valid_loss = test_loop(tools, dataloader)
            end_time = time()


            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # best_valid_macro_f1 = valid_macro_f1
                # best_valid_f1 = valid_f1
                # Set the path to save the model checkpoint.
                model_path = os.path.join(tools['checkpoint_path'], f'best_model.pth')
                save_checkpoint(epoch, model_path, tools)

    customlog(logfilepath, f"Fold {valid_batch_number} test\n-------------------------------\n")
    start_time = time()
    dataloader=tools["test_loader"]
    evaluate(tools, dataloader)
    evaluate_protein(dataloader, tools)
    end_time = time()

    del tools, encoder, dataloaders_dict, optimizer, scheduler
    torch.cuda.empty_cache()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CPM')
    parser.add_argument("--config_path", help="The location of config file", default='./config.yaml')
    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path) as file:
        config_dict = yaml.full_load(file)


    for i in range(5):
        valid_num=i
        if valid_num==4:
            test_num=0
        else:
            test_num=valid_num+1
        main(config_dict, valid_num, test_num)
        break







