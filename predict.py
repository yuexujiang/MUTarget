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



def get_prediction(n, data_dict):
    s=len(data_dict.keys())
    result_pro=np.zeros([s,n])
    motif_pred = [{} for i in range(n)]
    result_id = []
    for head in range(n):
        x_list=[]
        for id_protein in data_dict.keys():
            x = data_dict[id_protein]['motif_logits_protein'][head]  #[seq]
            motif_pred[head][id_protein]=x
            x_list.append(data_dict[id_protein]['type_pred'][head])
            if not id_protein in result_id:
                result_id.append(id_protein)
        
        pred=np.array(x_list)
        result_pro[:,head] = pred
    
    return result_pro, motif_pred, result_id

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
        data_dict[id_protein]['motif_logits_protein']=motif_logits_protein
    return data_dict

def present(tools, result_pro, motif_pred, result_id):
    classname=["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
             "SIGNAL", "chloroplast", "Thylakoid"]
    output_file = os.path.join(tools['result_path'],"prediction_results.txt")
    logfile=open(output_file, "w")

    cutoffs = list(tools["cutoffs"])
    result_bool = np.zeros_like(result_pro)
    for j in range(result_pro.shape[1]):
        result_bool[:,j] = result_pro[:,j]>cutoffs[j]
    for i in range(len(result_id)):
        id = result_id[i]
        logfile.write(id+"\n")
        pro_pred = result_bool[i]
        pred = np.where(pro_pred==1)[0]
        if pred.size == 0:
            logfile.write("Other\n")
        else:
            for j in pred:
                logfile.write(classname[j]+"\n")
                logfile.write(str(motif_pred[j][id]>cutoffs[j])+"\n")
    logfile.close()

def fix_pred(result_pro, motif_pred, result_id, cutoffs):
    ind_thylakoid = -1
    ind_chlo = -2
    for i in range(result_pro.shape[0]):
        id = result_id[i]
        if result_pro[i,ind_thylakoid]>=cutoffs[ind_thylakoid]:
            result_pro[i,ind_chlo]=1
            cs_thy = np.argmax(motif_pred[ind_thylakoid][id])
            motif_pred[ind_chlo][id][cs_thy:]=0
    return result_pro, motif_pred

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def present2(tools, result_pro, motif_pred, result_id):
    result={}
    classname=["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
             "SIGNAL", "chloroplast", "Thylakoid"]
    cutoffs = list(tools["cutoffs"])

    result_pro, motif_pred = fix_pred(result_pro, motif_pred, result_id, cutoffs)

    for i in range(len(result_id)):
        id = result_id[i]
        seq_len = len(motif_pred[0][id])
        result[id]={}
        for j in range(len(classname)):
            name=classname[j]
            if result_pro[i,j]<cutoffs[j]:
                result[id][name]=""
            else:
                if name in ["Mitochondrion","SIGNAL", "chloroplast", "Thylakoid"]:
                    result[id][name]="0-"+ str(np.argmax(motif_pred[j][id]))
                elif name == "ER":
                    result[id][name]=str(np.argmax(motif_pred[j][id])) + "-"+ str(seq_len-1)
                elif name == "Peroxisome":
                    cs = np.argmax(motif_pred[j][id])
                    if seq_len-cs > cs:
                        result[id][name]="0-"+ str(cs)
                    else:
                        result[id][name]=str(cs) + "-"+ str(seq_len-1)
                elif name in ["Nucleus", "Nucleus_export"]:
                    sites= np.where(motif_pred[j][id]>cutoffs[j])[0]
                    if len(sites)==0:
                        site = np.argmax(motif_pred[j][id])
                        sites = [site]
                        if site-2>=0:
                            sites = [site-2, site-1, site]
                        if site+2<=seq_len-1:
                            sites = sites.extend([site+1, site+2])
                    result[id][name]=str(np.array(sites))
    json_object = json.dumps(result, indent=2, cls=NpEncoder)
    output_file = os.path.join(tools['result_path'],"prediction_results.json")
    with open(output_file, "w") as outfile:
        outfile.write(json_object)

def present3(tools, result_pro, motif_pred, result_id, data_dict):
    id2AAscores={}
    id2label={}
    classname=["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
             "SIGNAL", "chloroplast", "Thylakoid"]
    cutoffs = list(tools["cutoffs"])

    result_pro, motif_pred = fix_pred(result_pro, motif_pred, result_id, cutoffs)

    for i in range(len(result_id)):
        id = result_id[i]
        seq = data_dict[id]['seq_protein']
        seq_len = len(seq)
        motif_score = np.zeros(seq_len)
        id2AAscores[id]=motif_score
        label=""
        for j in range(len(classname)):
            name=classname[j]
            if result_pro[i,j]<cutoffs[j]:
                continue
            else:
                label+="\t"+name
                score = j+1
                if name in ["Mitochondrion","SIGNAL", "chloroplast", "Thylakoid"]:
                    # result[id][name]="0-"+ str(np.argmax(motif_pred[j][id]))
                    cs = np.argmax(motif_pred[j][id])
                    for k in range(cs):
                        if motif_score[k]==0:
                            motif_score[k]=score
                elif name == "ER":
                    # result[id][name]=str(np.argmax(motif_pred[j][id])) + "-"+ str(seq_len-1)
                    cs = np.argmax(motif_pred[j][id])
                    motif_score[cs:seq_len]=score
                elif name == "Peroxisome":
                    cs = np.argmax(motif_pred[j][id])
                    if seq_len-cs > cs:
                        # result[id][name]="0-"+ str(cs)
                        motif_score[0:cs]=score
                    else:
                        # result[id][name]=str(cs) + "-"+ str(seq_len-1)
                        motif_score[cs:seq_len]=score
                elif name in ["Nucleus", "Nucleus_export"]:
                    sites= np.where(motif_pred[j][id]>cutoffs[j])[0]
                    if len(sites)==0:
                        site = np.argmax(motif_pred[j][id])
                        sites = [site]
                        if site-2>=0:
                            sites = [site-2, site-1, site]
                        if site+2<=seq_len-1:
                            sites.extend([site+1, site+2])
                    motif_score[sites]=score
                    # result[id][name]=str(np.array(sites))
        id2AAscores[id]=motif_score
        if label=="":
            label="\tOthers"
        id2label[id]=label
    # json_object = json.dumps(result, indent=2, cls=NpEncoder)
    output_file = os.path.join(tools['result_path'],"prediction_results.txt")
    with open(output_file, "w") as outfile:
        # outfile.write(json_object)
        # for k in result.keys():
        #     outfile.write(">"+str(k)+"\n")
        #     outfile.write(str(result[k]))
        #     outfile.write("\n")
        for i in range(len(result_id)):
            id = result_id[i]
            outfile.write(">"+str(id)+id2label[id]+"\n")
            outfile.write(str(result_pro[i])+"\n")
            outfile.write(str(id2AAscores[id]))
            outfile.write("\n")




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
            print(1)
            id_frags_list, seq_frag_tuple = make_buffer_pred(id_frag_list_tuple, seq_frag_list_tuple)
            encoded_seq=tokenize(tools, seq_frag_tuple)
            if type(encoded_seq)==dict:
                for k in encoded_seq.keys():
                    encoded_seq[k]=encoded_seq[k].to(tools['pred_device'])
            else:
                encoded_seq=encoded_seq.to(tools['pred_device'])

            classification_head, motif_logits = tools['net'](encoded_seq, id_tuple, id_frags_list, seq_frag_tuple)
            m=torch.nn.Sigmoid()
            motif_logits = m(motif_logits)
            classification_head = m(classification_head)

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
        print(2)

        data_dict = frag2protein_pred(data_dict, tools)
        print(3)

        result_pro, motif_pred, result_id = get_prediction(n, data_dict)  # result_pro = [sample_size, class_num], sample order same as result_id  
                                                                                         # motif_pred = class_num dictionaries with protein id as keys
                                                                                         # result_id = [sample_size], protein ids
        # present(tools, result_pro, motif_pred, result_id)
        print(4)
        present3(tools, result_pro, motif_pred, result_id, data_dict)




        
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

def prepare_samples_pred(csv_file, configs):
    # label2idx = {"Nucleus":0, "ER":1, "Peroxisome":2, "Mitochondrion":3, "Nucleus_export":4,
    #              "dual":5, "SIGNAL":6, "chloroplast":7, "Thylakoid":8}
    label2idx = {"Nucleus":0, "ER":1, "Peroxisome":2, "Mitochondrion":3, "Nucleus_export":4,
                 "SIGNAL":5, "chloroplast":6, "Thylakoid":7}
    samples = []
    n = configs.encoder.num_classes
    df = pd.read_csv(csv_file)
    row,col=df.shape
    for i in range(row):
        prot_id = df.loc[i,"Entry"]
        seq = df.loc[i,"Sequence"]
  
        id_frag_list, seq_frag_list = split_protein_sequence_pred(prot_id, seq, configs)
        samples.append((prot_id, id_frag_list, seq_frag_list))
        # for j in range(len(seq_frag_list )):
        #     id=prot_id+"@"+str(j)
        #     samples.append((id, fragments[j]))
        
    return samples

def prepare_dataloaders_pred(configs, input_file):
    # id_to_seq = prot_id_to_seq(seq_file)
    samples = prepare_samples_pred(input_file,configs)

    random.seed(configs.fix_seed)
    # Shuffle the list
    random.shuffle(samples)
    

    # print(train_dataset)
    dataset = LocalizationDataset_pred(samples, configs=configs)
 
    pred_dataloader = DataLoader(dataset, batch_size=configs.train_settings.batch_size, shuffle=False, collate_fn=custom_collate_pred)

    return pred_dataloader

def prepare_pred_dir(configs, output_dir, model_dir):
    curdir_path=os.getcwd()
    checkpoint_file = os.path.join(os.path.abspath(model_dir),"best_model.pth")
    result_path = os.path.abspath(output_dir)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    return result_path, checkpoint_file, curdir_path


def main(config_dict, input_file, output_dir, model_dir):
    configs = load_configs(config_dict)
    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()


    dataloader = prepare_dataloaders_pred(configs, input_file)
    result_path, checkpoint_file, curdir_path = prepare_pred_dir(configs, output_dir, model_dir)

    tokenizer=prepare_tokenizer(configs, curdir_path)

    encoder=prepare_models(configs, '', curdir_path)

    model_checkpoint = torch.load(checkpoint_file, map_location='cpu')
    encoder.load_state_dict(model_checkpoint['model_state_dict'])

    tools = {
        'frag_overlap': configs.encoder.frag_overlap,
        'cutoffs': configs.predict_settings.cutoffs,
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


    predict(dataloader, tools)
    end_time = time()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CPM')
    parser.add_argument("--config_path", help="The location of config file", default='./config.yaml')
    parser.add_argument("--input_file", help="The location of input fasta file")
    parser.add_argument("--output_dir", help="The dir location of output")
    parser.add_argument("--model_dir", help="The dir location of trained model")
    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path) as file:
        config_dict = yaml.full_load(file)
    
    input_file = args.input_file
    output_dir = args.output_dir
    model_dir = args.model_dir
    

    main(config_dict, input_file, output_dir, model_dir)
    #use case
    #python predict.py --config_path ./config.yaml --input_file ./test_data_fold1_sub.csv --output_dir ./results_test --model_dir ./test_checkpoint/fold0
    








