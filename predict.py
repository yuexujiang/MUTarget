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
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



def get_prediction(tools, cutoff, n, data_dict):
    s=len(data_dict.keys())
    result_pro=np.zeros([s,n])
    motif_pred = [{} for i in range(n)]
    result_id = []
    for head in range(n):
        x_list=[]
        for id_protein in data_dict.keys():
            x = data_dict[id_protein]['motif_logits_protein'][head]  #[seq]
            motif_pred[head][id_protein]=x
            x_list.append(np.max(x))
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
                x_overlap = np.maximum(motif_logits_protein[:,-overlap:], motif_logits_frag[:,:overlap])
                motif_logits_protein = np.concatenate((motif_logits_protein[:,:-overlap], x_overlap, motif_logits_frag[:,overlap:l]),axis=1)
        data_dict[id_protein]['seq_protein']=seq_protein
        data_dict[id_protein]['motif_logits_protein']=motif_logits_protein
    return data_dict

def predict(dataloader, tools):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.eval().cuda()
    tools['net'].eval().to(tools["pred_device"])
    n=tools['num_classes']

    

    cutoff = tools['cutoff']
    data_dict={}
    with torch.no_grad():
        for batch, (id, seq_frag) in enumerate(dataloader):
            
            encoded_seq=tokenize(tools, seq_frag)
            if type(encoded_seq)==dict:
                for k in encoded_seq.keys():
                    encoded_seq[k]=encoded_seq[k].to(tools['pred_device'])
            else:
                encoded_seq=encoded_seq.to(tools['pred_device'])
            motif_logits = tools["net"](encoded_seq)
            m=torch.nn.Sigmoid()
            motif_logits = m(motif_logits)

            x = np.array(motif_logits.cpu())   #[batch, head, seq]

            for i in range(len(id)):
                id_protein=id[i].split('@')[0]
                if id_protein in data_dict.keys():
                    data_dict[id_protein]['id_frag'].append(id[i])
                    data_dict[id_protein]['seq_frag'].append(seq_frag[i])
                    data_dict[id_protein]['motif_logits'].append(x[i])    #[[head, seq], ...]
                else:
                    data_dict[id_protein]={}
                    data_dict[id_protein]['id_frag']=[id[i]]
                    data_dict[id_protein]['seq_frag']=[seq_frag[i]]
                    data_dict[id_protein]['motif_logits']=[x[i]]

        data_dict = frag2protein_pred(data_dict, tools)

        result_pro, motif_pred, result_id = get_prediction(tools, cutoff, n, data_dict)  # result_pro = [sample_size, class_num], sample order same as result_id  
                                                                                         # motif_pred = class_num dictionaries with protein id as keys
                                                                                         # result_id = [sample_size], protein ids
        
        classname=["Nucleus", "ER", "Peroxisome", "Mitochondrion", "Nucleus_export",
             "SIGNAL", "chloroplast", "Thylakoid"]
        output_file = os.path.join(tools['result_path'],"prediction_results.txt")
        logfile=open(output_file, "w")
        result_bool= result_pro>0.5
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
                    logfile.write(str(motif_pred[j][id]>0.5)+"\n")
        logfile.close()


        
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
        id, seq_frag = self.samples[idx]
        return id, seq_frag

def split_protein_sequence_pred(sequence, configs):
    fragment_length = configs.encoder.max_len - 2
    overlap = configs.encoder.frag_overlap
    fragments = []
    sequence_length = len(sequence)
    start = 0

    while start < sequence_length:
        end = start + fragment_length
        if end > sequence_length:
            end = sequence_length
        fragment = sequence[start:end]
        fragments.append(fragment)
        start += fragment_length - overlap

    return fragments

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
  
        fragments = split_protein_sequence_pred(seq, configs)
        for j in range(len(fragments)):
            id=prot_id+"@"+str(j)
            samples.append((id, fragments[j]))
        
    return samples

def prepare_dataloaders_pred(configs, input_file):
    # id_to_seq = prot_id_to_seq(seq_file)
    samples = prepare_samples_pred(input_file,configs)

    random.seed(configs.fix_seed)
    # Shuffle the list
    random.shuffle(samples)
    

    # print(train_dataset)
    dataset = LocalizationDataset_pred(samples, configs=configs)
 
    pred_dataloader = DataLoader(dataset, batch_size=configs.train_settings.batch_size, shuffle=False)

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
        'cutoff': configs.valid_settings.cutoff,
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
    








