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
from Bio.Seq import Seq
np.random.seed(42)
random.seed(42)

label2idx = {"Other":0,
             "ER": 1, 
             "Peroxisome": 2, 
             "Mitochondrion": 3, 
             "SIGNAL": 4, 
             "Nucleus": 5,
             "Nucleus_export": 6, 
             "chloroplast": 7, 
             "Thylakoid": 8}

class LocalizationDataset(Dataset):
    def __init__(self, samples, configs,mode="train"):
        # self.label_to_index = {"Other": 0, "SP": 1, "MT": 2, "CH": 3, "TH": 4}
        # self.index_to_label = {0: "Other", 1: "SP", 2: "MT", 3: "CH", 4: "TH"}
        # self.transform = transform
        # self.target_transform = target_transform
        # self.cs_transform = cs_transform
        self.original_samples = samples
        self.n = configs.encoder.num_classes
        self.class_weights = calculate_class_weights(self.count_samples_by_class(self.n, self.original_samples))
        self.residue_class_weights = calculate_residue_class_weights(self.n, self.original_samples) #dict[classtype]:weight
        if configs.train_settings.other_weight:
             self.residue_class_weights[0] = configs.train_settings.other_weight #before is 0.008
        
        print("class_weights")
        print(self.class_weights)
        print("residue_class_weights")
        print(self.residue_class_weights)
        if mode == "train" and configs.train_settings.data_aug.enable:
           self.data_aug = True
           if hasattr(configs.train_settings.data_aug,"binomial"):
               self.binomial = configs.train_settings.data_aug.binomial
           else:
               self.binomial = False
           if configs.train_settings.data_aug.warmup == 0: #if warmup ==0, data_aug on first epoch
              samples = self.data_aug_train(samples,configs,self.class_weights)

        self.samples = samples
        #print(samples[0:2]) #same as original

        self.mode = mode

    #"""
    def random_mutation(self, sequence, target, pos_mutation_rate, neg_mutation_rate):
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # List of standard amino acids
        seq = Seq(sequence)
        seq_list = list(seq)
        # Get the mutable positions
        # print(target)
        pos_mutable_positions = [i for i, label in enumerate(target) if label == 1]
        if len(pos_mutable_positions) == 1:
            num_pos_mutations = 1 if random.random() < pos_mutation_rate else 0
        else:
            num_pos_mutations = int(pos_mutation_rate * len(pos_mutable_positions))

        neg_mutable_positions = [i for i, label in enumerate(target) if label == 0]
        num_neg_mutations = int(neg_mutation_rate * len(neg_mutable_positions))
        if num_pos_mutations > 0 or num_neg_mutations > 0:
            if num_pos_mutations > 0:
                num_pos_mutations = min(num_pos_mutations, len(pos_mutable_positions))
                mutation_positions = random.sample(pos_mutable_positions, num_pos_mutations)
                for pos in mutation_positions:
                    # Ensure the mutated amino acid is different from the original
                    new_aa = random.choice([aa for aa in amino_acids if aa != seq_list[pos]])
                    seq_list[pos] = new_aa

            if num_neg_mutations > 0:
                num_neg_mutations = min(num_neg_mutations, len(neg_mutable_positions))
                mutation_positions = random.sample(neg_mutable_positions, num_neg_mutations)
                for pos in mutation_positions:
                    # Ensure the mutated amino acid is different from the original
                    new_aa = random.choice([aa for aa in amino_acids if aa != seq_list[pos]])
                    seq_list[pos] = new_aa

            # Join the mutated amino acids back into a sequence
            mutated_sequence = ''.join(seq_list)
            # print(sequence)
            # print(mutated_sequence)
            return mutated_sequence
        else:
            return sequence
    #"""
    def random_mutation_binomial(self, sequence, target, pos_mutation_rate, neg_mutation_rate):
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # List of standard amino acids
        seq = Seq(sequence)
        seq_list = list(seq)
        # Get the mutable positions
        #print("".join([str(x) for x in target]))
        pos_mutable_positions = [i for i, label in enumerate(target) if label == 1]
        #if len(pos_mutable_positions) == 1:
        #    num_pos_mutations = 1 if random.random() < pos_mutation_rate else 0
        #else:
        #    num_pos_mutations = int(pos_mutation_rate * len(pos_mutable_positions))
        neg_mutable_positions = [i for i, label in enumerate(target) if label == 0]
        #num_neg_mutations = int(neg_mutation_rate * len(neg_mutable_positions))
        if len(pos_mutable_positions) > 0 or len(neg_mutable_positions) > 0:
            probability = np.asarray(target)*pos_mutation_rate+(1-np.asarray(target))*neg_mutation_rate
            #print(probability)
            indices_replaced = np.random.binomial(1, probability).astype(bool)
            #print(indices_replaced)
            for i_rep,value in enumerate(indices_replaced):
                if value:
                     seq_list[i_rep] = random.choice(list(amino_acids))# if aa != seq_list[i_rep]]) remove original
            
            # Join the mutated amino acids back into a sequence
            mutated_sequence = ''.join(seq_list)
            return mutated_sequence
        else:
            return sequence
    #"""
    
    def data_aug_train(self, samples, configs, class_weights):
        print("data aug on len of " + str(len(samples)))
        aug_samples = []
        pos_mutation_rate, neg_mutation_rate = configs.train_settings.data_aug.pos_mutation_rate, configs.train_settings.data_aug.neg_mutation_rate

        for id, id_frag_list, seq_frag_list, target_frag_list, type_protein in samples:
            class_positions = np.where(type_protein == 1)[0]

            # 这里我改了, 为了测试
            per_times = np.max([1, int(np.ceil(
                configs.train_settings.data_aug.per_times * np.max([class_weights[x] for x in class_positions])))])
            # per_times = 1


            temp_target_frag_list = target_frag_list.copy()
            for aug_i in range(per_times):
                aug_id = id + "_" + str(aug_i)
                aug_id_frag_list = [aug_id + "@" + id_frag.split("@")[1] for id_frag in id_frag_list]

                if aug_i == 0:
                    flattened_aug_target_frag_list = np.hstack(temp_target_frag_list)
                    #之前的错误，要把others类设为全0 
                    flattened_aug_target_frag_list[0,:] = [0] * flattened_aug_target_frag_list.shape[1]
                    if 1 in flattened_aug_target_frag_list[label2idx['Nucleus']] or 1 in flattened_aug_target_frag_list[label2idx['Nucleus_export']]:
                        pass
                    idx = label2idx['ER']
                    if 1 in flattened_aug_target_frag_list[idx]:
                        # pass
                        stop_left = flattened_aug_target_frag_list[idx].tolist().index(1)
                        flattened_aug_target_frag_list[idx][stop_left + 1:] = [1] * \
                                                                      (len(flattened_aug_target_frag_list[idx]) - stop_left - 1)
                    idx = label2idx['Peroxisome']
                    if 1 in flattened_aug_target_frag_list[idx]:
                        # pass
                        stop_left = flattened_aug_target_frag_list[idx].tolist().index(1)
                        stop_right = len(flattened_aug_target_frag_list[idx]) - 1 - \
                                     flattened_aug_target_frag_list[idx].tolist()[::-1].index(1)
                        stop = (stop_left+stop_right)/2
                        if stop < len(flattened_aug_target_frag_list[idx]) / 2:
                            flattened_aug_target_frag_list[idx][:stop_right] = [1] * stop_right
                        else:
                            flattened_aug_target_frag_list[idx][stop_left + 1:] = [1] * (
                                        len(flattened_aug_target_frag_list[idx]) - stop_left - 1)

                    N_side = [label2idx["Mitochondrion"], label2idx['SIGNAL'], label2idx['chloroplast'], label2idx['Thylakoid']]
                    for idx in N_side:
                        # pass
                        if 1 in flattened_aug_target_frag_list[idx]:
                            stop_right = len(flattened_aug_target_frag_list[idx]) - 1 - \
                                         flattened_aug_target_frag_list[idx].tolist()[::-1].index(1)
                            flattened_aug_target_frag_list[idx][:stop_right] = [1] * stop_right

                    shapes = [arr.shape for arr in temp_target_frag_list]

                    split_indices = np.cumsum([shape[1] for shape in shapes])[:-1]

                    temp_target_frag_list = np.split(flattened_aug_target_frag_list, split_indices, axis=1)
                
                #之前的缩进错误也改了
                if self.binomial:
                    aug_seq_frag_list = [
                                self.random_mutation_binomial(sequence, [int(max(set(column))) for column in zip(*target)][:len(sequence)],
                                pos_mutation_rate, neg_mutation_rate) for sequence, target in
                                zip(seq_frag_list, temp_target_frag_list)]

                else:
                    aug_seq_frag_list = [
                                self.random_mutation(sequence, [int(max(set(column))) for column in zip(*target)][:len(sequence)],
                                pos_mutation_rate, neg_mutation_rate) for sequence, target in
                                zip(seq_frag_list, temp_target_frag_list)]


                aug_target_frag_list = target_frag_list
                aug_type_protein = type_protein
                aug_samples.append(
                    (aug_id, aug_id_frag_list, aug_seq_frag_list, aug_target_frag_list, aug_type_protein))

        print("data length after augmentation is ", str(len(aug_samples)))
        # exit(0)
        return aug_samples

    @staticmethod
    def count_samples_by_class(n, samples):
        """Count the number of samples for each class."""
        # class_counts = {}
        class_counts = np.zeros(n)  # one extra is for samples without motif
        # Iterate over the samples
        for id, id_frag_list, seq_frag_list, target_frag_list, type_protein in samples:
            class_counts += type_protein
        return class_counts

    def __len__(self):
           return len(self.samples)

    def __getitem__(self, idx):
        id, id_frag_list, seq_frag_list, target_frag_list, type_protein = self.samples[idx]
        labels = np.where(type_protein == 1)[0]
        weights = []
        for label in labels:
            weights.append(self.class_weights[label])
        sample_weight = max(weights)
        residue_class_weights = self.residue_class_weights
        # print(self.class_weights)
        # exit(0) 0613 check
        # labels=np.where(np.max(target_frags, axis=1)==1)[0]
        # weights=[]
        # for label in labels:
        #     weights.append(self.class_weights[label])
        # if np.max(target_frags)==0:
        #     weights.append(self.class_weights[self.n])

        # sample_weight = max(weights)
        # target_frags = torch.from_numpy(np.stack(target_frags, axis=0))
        type_protein = torch.from_numpy(type_protein)
        return id, id_frag_list, seq_frag_list, target_frag_list, type_protein, sample_weight, residue_class_weights
        # return id, type_protein


def custom_collate(batch):
    id, id_frags, fragments, target_frags, type_protein, sample_weight,residue_class_weights= zip(*batch)
    return id, id_frags, fragments, target_frags, type_protein, sample_weight, residue_class_weights


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
    ind = 0

    while start < sequence_length:
        end = start + fragment_length
        if end > sequence_length:
            end = sequence_length
        fragment = sequence[start:end]
        target_frag = targets[:, start:end]
        if target_frag.shape[1] < fragment_length:
            pad = np.zeros([targets.shape[0], fragment_length-target_frag.shape[1]])
            target_frag = np.concatenate((target_frag, pad), axis=1)
        target_frags.append(target_frag)
        fragments.append(fragment)
        id_frags.append(prot_id+"@"+str(ind))
        ind += 1
        if start + fragment_length > sequence_length:
            break
        start += fragment_length - overlap

    return id_frags, fragments, target_frags


def fix_sample(motif_left, motif_right, label, label2idx, type_protein, targets):
    if motif_left == "None":
        motif_left = 0
    else:
        motif_left = int(motif_left)-1
    motif_right = int(motif_right)
    if label == "Thylakoid": #and motif_left != 0: #duolin modified on 7/21/2024 if type_protein is "Thylakoid" it must also be "chloroplast"
        index_row = label2idx["chloroplast"]
        type_protein[index_row] = 1
    if label == "Thylakoid" and motif_left != 0:
       targets[index_row, motif_left-1] = 1
    
    return motif_left, motif_right, type_protein, targets


def prepare_samples(csv_file, configs):
    samples = []
    n = configs.encoder.num_classes
    df = pd.read_csv(csv_file)
    row, col = df.shape
    for i in range(row):
        prot_id = df.loc[i, "Entry"]
        seq = df.loc[i, "Sequence"]
        targets = np.zeros([n, len(seq)])
        type_protein = np.zeros(n)
        # motifs = df.iloc[i,1:-2]
        if pd.isnull(df.loc[i,"MOTIF"]):
            type_protein[0]=1
        else:
            motifs = df.loc[i, "MOTIF"].split("|")
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
        
        targets[0, :] = np.all(targets[1:, :] == 0, axis=0).astype(int)
        
        id_frag_list, seq_frag_list, target_frag_list = split_protein_sequence(prot_id, seq, targets, configs)
        samples.append((prot_id, id_frag_list, seq_frag_list, target_frag_list, type_protein))
        # for j in range(len(fragments)):
        #     id=prot_id+"@"+str(j)
        #     samples.append((id, fragments[j], target_frags[j], type_protein))
    return samples


def prepare_dataloaders(configs, valid_batch_number, test_batch_number):
    # id_to_seq = prot_id_to_seq(seq_file)
    if configs.train_settings.dataset == 'v2':
        samples = prepare_samples("./parsed_EC7_v2/PLANTS_uniprot.csv", configs)
        samples.extend(prepare_samples("./parsed_EC7_v2/ANIMALS_uniprot.csv", configs))
        samples.extend(prepare_samples("./parsed_EC7_v2/FUNGI_uniprot.csv", configs))
        cv = pd.read_csv("./parsed_EC7_v2/split/type/partition.csv")
    if configs.train_settings.dataset == 'v2_cdhit0.9':
        print("v2_cdhit0.9")
        samples = prepare_samples("./parsed_EC7_v2_cdhit0.9/PLANTS_uniprot_cdhit.csv", configs)
        samples.extend(prepare_samples("./parsed_EC7_v2_cdhit0.9/ANIMALS_uniprot_cdhit.csv", configs))
        samples.extend(prepare_samples("./parsed_EC7_v2_cdhit0.9/FUNGI_uniprot_cdhit.csv", configs))
        cv = pd.read_csv("./parsed_EC7_v2/split/type/partition.csv")
    elif configs.train_settings.dataset == 'v3':
        samples = prepare_samples("./parsed_EC7_v3/PLANTS_uniprot.csv", configs)
        samples.extend(prepare_samples("./parsed_EC7_v3/ANIMALS_uniprot.csv", configs))
        samples.extend(prepare_samples("./parsed_EC7_v3/FUNGI_uniprot.csv", configs))
        cv = pd.read_csv("./parsed_EC7_v3/split/type/partition.csv")
    elif configs.train_settings.dataset == 'v4':
        samples = prepare_samples("./parsed_v4/PLANTS_uniprot.csv", configs)
        samples.extend(prepare_samples("./parsed_v4/ANIMALS_uniprot.csv", configs))
        samples.extend(prepare_samples("./parsed_v4/FUNGI_uniprot.csv", configs))
        cv = pd.read_csv("./parsed_v4/partition.csv")

    train_id = []
    val_id = []
    test_id = []
    id = cv.loc[:, 'entry']
    # split=cv.loc[:,'split']
    # fold=cv.loc[:,'fold']
    partition = cv.loc[:, 'partition']
    for i in range(len(id)):
        # f=fold[i]
        # s=split[i]
        p = partition[i]
        d = id[i]
        if p == valid_batch_number:
            val_id.append(d)
        elif p == test_batch_number:
            test_id.append(d)
        else:
            train_id.append(d)


    # print(train_id)


    train_sample = []
    valid_sample = []
    test_sample = []
    for i in samples:
        # id=i[0].split("@")[0]
        id = i[0]
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
    #print("first train[0]") #same!
    #print(train_sample[0])
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
    #print(train_sample[0])
    targetp_test = prepare_samples("./test_targetp/test_targetp_rmtrain_fold"+str(test_batch_number)+".csv", configs)
    
    train_dataset = LocalizationDataset(train_sample, configs=configs,mode = "train")
    valid_dataset = LocalizationDataset(valid_sample, configs=configs,mode = "valid")
    test_dataset = LocalizationDataset(test_sample, configs=configs,mode = "test")
    test_targetp_dataset = LocalizationDataset(targetp_test, configs=configs,mode = "test")
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size, shuffle=True, collate_fn=custom_collate,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=configs.valid_settings.batch_size, shuffle=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.valid_settings.batch_size, shuffle=False, collate_fn=custom_collate)
    test_targetp_dataloader = DataLoader(test_targetp_dataset, batch_size=configs.valid_settings.batch_size, shuffle=False, collate_fn=custom_collate)
    
    return {'train': train_dataloader, 'test': test_dataloader, 'valid': valid_dataloader,'test_targetp':test_targetp_dataloader}


