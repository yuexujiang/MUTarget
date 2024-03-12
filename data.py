
import torch
torch.manual_seed(0)
from torch.utils.data import Dataset
# from torchvision.transforms import ToTensor, Lambda
import numpy as np
from torch.utils.data import DataLoader, random_split
from utils import calculate_class_weights
import pandas as pd
import random
import yaml
from utils import *


class LocalizationDataset(Dataset):
    def __init__(self, samples, configs):
        # self.label_to_index = {"Other": 0, "SP": 1, "MT": 2, "CH": 3, "TH": 4}
        # self.index_to_label = {0: "Other", 1: "SP", 2: "MT", 3: "CH", 4: "TH"}
        # self.transform = transform
        # self.target_transform = target_transform
        # self.cs_transform = cs_transform
        self.samples = samples
        self.n = configs.encoder.num_classes
        print(self.count_samples_by_class(self.n, self.samples))
        self.class_weights = calculate_class_weights(self.count_samples_by_class(self.n, self.samples))
        print(self.class_weights)
    @staticmethod
    def count_samples_by_class(n, samples):
        """Count the number of samples for each class."""
        # class_counts = {}
        class_counts = np.zeros(n) # one extra is for samples without motif
        # Iterate over the samples
        for id, id_frag_list, seq_frag_list, target_frag_list, type_protein in samples:
            class_counts += type_protein
        return class_counts
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        id, id_frag_list, seq_frag_list, target_frag_list, type_protein = self.samples[idx]

        labels=np.where(type_protein==1)[0]
        weights=[]
        for label in labels:
            weights.append(self.class_weights[label])
        sample_weight = max(weights)
        # labels=np.where(np.max(target_frags, axis=1)==1)[0]
        # weights=[]
        # for label in labels:
        #     weights.append(self.class_weights[label])
        # if np.max(target_frags)==0:
        #     weights.append(self.class_weights[self.n])
        
        # sample_weight = max(weights)
        # target_frags = torch.from_numpy(np.stack(target_frags, axis=0))
        type_protein = torch.from_numpy(type_protein)
        return id, id_frag_list, seq_frag_list, target_frag_list, type_protein, sample_weight 
        # return id, type_protein
    
def custom_collate(batch):
    id, id_frags, fragments, target_frags, type_protein, sample_weight = zip(*batch)
    return id, id_frags, fragments, target_frags, type_protein, sample_weight


def prot_id_to_seq(seq_file):
    id2seq = {}
    with open(seq_file) as file:
        for line in file:
            id = line.strip().split("\t")[0]
            seq = line.strip().split("\t")[2]
            id2seq[id] = seq
    return id2seq

# def prepare_samples(exclude, fold_num_list, id2seq_dic, npz_file):
#     samples = []
#     data = np.load(npz_file)
#     fold = data['fold']
#     if exclude:
#         index = [all(num != target for target in fold_num_list) for num in fold]
#         # index = fold != fold_num
#     else:
#         index = [any(num == target for target in fold_num_list) for num in fold]
#         # index = fold == fold_num
#     prot_ids = data['ids'][index]
#     y_type = data['y_type'][index]
#     y_cs = data['y_cs'][index]
#     for idx, prot_id in enumerate(prot_ids):
#         seq = id2seq_dic[prot_id]
#         # if len(seq)>200:
#         #   seq=seq[:200]
#         label = y_type[idx]
#         position = np.argmax(y_cs[idx])
#         samples.append((seq, int(label), position))
#     return samples

def split_protein_sequence(prot_id, sequence, targets, configs):
    fragment_length = configs.encoder.max_len - 2
    overlap = configs.encoder.frag_overlap
    fragments = []
    target_frags = []
    id_frags = []
    sequence_length = len(sequence)
    start = 0
    ind=0

    while start < sequence_length:
        end = start + fragment_length
        if end > sequence_length:
            end = sequence_length
        fragment = sequence[start:end]
        target_frag = targets[:,start:end]
        if target_frag.shape[1]<fragment_length:
            pad=np.zeros([targets.shape[0],fragment_length-target_frag.shape[1]])
            target_frag =  np.concatenate((target_frag, pad),axis=1)
        target_frags.append(target_frag)
        fragments.append(fragment)
        id_frags.append(prot_id+"@"+str(ind))
        ind+=1
        if start + fragment_length > sequence_length:
            break
        start += fragment_length - overlap

    return id_frags, fragments, target_frags

#["Nucleus","ER","Peroxisome","Mitochondrion","Nucleus_export","dual","SIGNAL","chloroplast","Thylakoid"]
def fix_sample(motif_left, motif_right, label, label2idx, type_protein, targets):
    if motif_left=="None":
        motif_left=0
    else:
        motif_left = int(motif_left)-1
    motif_right = int(motif_right)
    if label == "Thylakoid" and motif_left != 0:
        index_row = label2idx["chloroplast"]
        type_protein[index_row] = 1
        targets[index_row, motif_left-1] = 1
    return motif_left, motif_right, type_protein, targets



def prepare_samples(csv_file, configs):
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
        targets = np.zeros([n,len(seq)])
        type_protein = np.zeros(n)
        # motifs = df.iloc[i,1:-2]
        motifs = df.loc[i,"MOTIF"].split("|")
        for motif in motifs:
            if not pd.isnull(motif):
                # label = motif.split("|")[0].split(":")[1]
                label = motif.split(":")[1]
                # motif_left = motif.split("|")[0].split(":")[0].split("-")[0]
                motif_left = motif.split(":")[0].split("-")[0]
                motif_right = motif.split(":")[0].split("-")[1]
                
                motif_left, motif_right, type_protein, targets = fix_sample(motif_left, motif_right, label, label2idx, type_protein, targets)
                if label in label2idx:
                    index_row = label2idx[label]
                    type_protein[index_row] = 1
                    if label in ["SIGNAL", "chloroplast", "Thylakoid", "Mitochondrion"]:
                        targets[index_row, motif_right-1] = 1
                    elif label == "Peroxisome" and motif_left == 0:
                        targets[index_row, motif_right-1] = 1
                    elif label == "Peroxisome" and motif_left != 0:
                        targets[index_row, motif_left] = 1
                    elif label == "ER":
                        targets[index_row, motif_left] = 1
                    elif label == "Nucleus" or label == "Nucleus_export":
                        targets[index_row, motif_left:motif_right] = 1
        id_frag_list, seq_frag_list, target_frag_list = split_protein_sequence(prot_id, seq, targets, configs)
        samples.append((prot_id, id_frag_list, seq_frag_list, target_frag_list, type_protein))
        # for j in range(len(fragments)):
        #     id=prot_id+"@"+str(j)
        #     samples.append((id, fragments[j], target_frags[j], type_protein))
        
    return samples


def prepare_dataloaders(configs, valid_batch_number, test_batch_number):
    # id_to_seq = prot_id_to_seq(seq_file)
    if configs.train_settings.dataset == 'v2':
        samples = prepare_samples("./parsed_EC7_v2/PLANTS_uniprot.csv",configs)
        samples.extend(prepare_samples("./parsed_EC7_v2/ANIMALS_uniprot.csv", configs))
        samples.extend(prepare_samples("./parsed_EC7_v2/FUNGI_uniprot.csv", configs))
        cv=pd.read_csv("./parsed_EC7_v2/split/type/partition.csv")
    elif configs.train_settings.dataset == 'v3':
        samples = prepare_samples("./parsed_EC7_v3/PLANTS_uniprot.csv",configs)
        samples.extend(prepare_samples("./parsed_EC7_v3/ANIMALS_uniprot.csv", configs))
        samples.extend(prepare_samples("./parsed_EC7_v3/FUNGI_uniprot.csv", configs))
        cv=pd.read_csv("./parsed_EC7_v3/split/type/partition.csv")
    train_id=[]
    val_id=[]
    test_id=[]
    id=cv.loc[:,'entry']
    # split=cv.loc[:,'split']
    # fold=cv.loc[:,'fold']
    partition=cv.loc[:,'partition']
    for i in range(len(id)):
        # f=fold[i]
        # s=split[i]
        p=partition[i]
        d=id[i]
        if p == valid_batch_number:
            val_id.append(d)
        elif p == test_batch_number:
            test_id.append(d)
        else:
            train_id.append(d)


    # print(train_id)

    
    train_sample=[]
    valid_sample=[]
    test_sample=[]

    for i in samples:
        # id=i[0].split("@")[0]
        id=i[0]
        # print(id)
        if id in train_id:
            train_sample.append(i)
        elif id in val_id:
            valid_sample.append(i)
        elif id in test_id:
            test_sample.append(i)

    # train_samples = prepare_samples(exclude=True, fold_num_list=[valid_batch_number, test_batch_number], id2seq_dic=id_to_seq, npz_file=npz_file)
    # valid_samples = prepare_samples(exclude=False, fold_num_list=[valid_batch_number], id2seq_dic=id_to_seq, npz_file=npz_file)
    # test_samples = prepare_samples(exclude=False, fold_num_list=[test_batch_number], id2seq_dic=id_to_seq, npz_file=npz_file)
    random.seed(configs.fix_seed)
    # Shuffle the list
    random.shuffle(samples)
    # train_dataset = LocalizationDataset(samples, configs=configs)
    # dataset_size = len(train_dataset)
    # train_size = int(0.8 * dataset_size)  # 80% for training, adjust as needed
    # test_size = dataset_size - train_size
    # train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])
    # val_size = train_size - int(0.8 * train_size)
    # train_dataset, valid_dataset = random_split(train_dataset, [int(0.8 * train_size), val_size])



    # print(train_dataset)
    train_dataset = LocalizationDataset(train_sample, configs=configs)
    valid_dataset = LocalizationDataset(valid_sample, configs=configs)
    test_dataset = LocalizationDataset(test_sample, configs=configs)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size, shuffle=True, collate_fn=custom_collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=configs.valid_settings.batch_size, shuffle=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.valid_settings.batch_size, shuffle=True, collate_fn=custom_collate)
    return {'train': train_dataloader, 'test': test_dataloader , 'valid': valid_dataloader}

if __name__ == '__main__':
    config_path = './config.yaml'
    with open(config_path) as file:
        configs_dict = yaml.full_load(file)

    configs_file = load_configs(configs_dict)

    dataloaders_dict = prepare_dataloaders(configs_file, 0, 1)

    for batch in dataloaders_dict['train']:
        # id_batch, fragments_batch, target_frags_batch, weights_batch = batch 
        (prot_id, id_frag_list, seq_frag_list, target_frag_nplist, type_protein_pt, sample_weight) = batch 
        # id, type_protein = batch 
        # print(len(id_batch))
        # print(len(fragments_batch))
        # print(np.array(target_frags_batch).shape)
        # print(len(weights_batch))
        print("==========================")
        print(type(prot_id))
        print(prot_id)
        print(type(id_frag_list))
        print(id_frag_list)
        print(type(seq_frag_list))
        print(seq_frag_list)
        print(type(target_frag_nplist))
        print(target_frag_nplist)
        print(type(type_protein_pt))
        print(type_protein_pt)
        print(type(sample_weight))
        print(sample_weight)
        # print(next(iter(dataloaders_dict['test'])))
        # a=np.array(target_frags_batch)
        # print(np.max(a, axis=2))
        # print(target_frags_batch.size())
        # print(target_frags_batch[1])
        # print(weights_batch)
        break

    print('done')

