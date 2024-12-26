from peft import get_peft_model
from peft import LoraConfig, PromptTuningConfig, PromptTuningInit
import torch
from torch import nn
import torch.nn.functional as F
torch.manual_seed(0)
import os
from transformers import AutoTokenizer
from transformers import EsmModel
from utils import customlog, prepare_saving_dir

from prompt_tunning import PrefixTuning
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix to hold positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # Reshape for batch and sequence length
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pe[:x.size(0), :]
        return x

# from train import make_buffer



# from train import make_buffer


class LayerNormNet(nn.Module):
    """
    From https://github.com/tttianhao/CLEAN
    """

    def __init__(self, configs, hidden_dim=512, out_dim=256):
        super(LayerNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.device = configs.train_settings.device
        self.dtype = torch.float32
        feature_dim = {"facebook/esm2_t6_8M_UR50D": 320, "facebook/esm2_t33_650M_UR50D": 1280,
                       "facebook/esm2_t30_150M_UR50D": 640, "facebook/esm2_t12_35M_UR50D": 480}
        self.fc1 = nn.Linear(feature_dim[configs.encoder.model_name], hidden_dim, dtype=self.dtype, device=self.device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=self.dtype, device=self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=self.dtype, device=self.device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=self.dtype, device=self.device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=self.dtype, device=self.device)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        # x = self.dropout(self.ln2(self.fc2(x)))
        # x = torch.relu(x)
        x = self.fc3(x)
        return x


def initialize_PromptProtein(pretrain_loc, trainable_layers):
    from PromptProtein.models import openprotein_promptprotein
    from PromptProtein.utils import PromptConverter
    model, dictionary = openprotein_promptprotein(pretrain_loc)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        for lname in trainable_layers:
            if lname in name:
                param.requires_grad = True
    return model


class CustomPromptModel(nn.Module):
    def __init__(self, configs, pretrain_loc, trainable_layers):
        super(CustomPromptModel, self).__init__()
        self.pretrain = initialize_PromptProtein(pretrain_loc, trainable_layers)
        # self.learnable_prompt_emb = nn.Parameter(initialize_learnable_prompts(prompt_tok_vec, self.pretrain, self.converter)).to("cuda")
        # self.learnable_prompt_emb = nn.Parameter(initialize_learnable_prompts(8, self.pretrain, self.converter)).to("cuda")
        # self.learnable_prompt_emb = nn.Parameter(initialize_learnable_prompts(8, self.pretrain, self.converter, self.device)).to(self.device)
        # self.decoder_class = Decoder_linear(input_dim=1280, output_dim=5)
        self.pooling_layer = nn.AdaptiveAvgPool1d(output_size=1)
        self.head = nn.Linear(1280, configs.encoder.num_classes)
        self.cs_head = nn.Linear(1280, 1)
        # if decoder_cs=="linear":
        #     self.decoder_cs = Decoder_linear(input_dim=1280, output_dim=1)

    def forward(self, encoded_seq):
        result = self.pretrain(encoded_seq, with_prompt_num=1)
        # logits size => (B, T+2, E)
        logits = result['logits']

        transposed_feature = logits.transpose(1, 2)
        pooled_features = self.pooling_layer(transposed_feature).squeeze(2)
        type_probab = self.head(pooled_features)
        cs_head = self.cs_head(logits).squeeze(dim=-1)
        cs_pred = cs_head[:, 1:]
        # print("peptide_probab size ="+str(type_probab.size()))
        return type_probab, cs_pred


def prepare_tokenizer(configs, curdir_path):
    if configs.encoder.composition == "esm_v2":
        tokenizer = AutoTokenizer.from_pretrained(configs.encoder.model_name)
    else:
        tokenizer = None  # for adapter
    return tokenizer


def tokenize(tools, seq):
    if tools['composition'] == "esm_v2":
        max_length = tools['max_len']
        encoded_sequence = tools["tokenizer"](seq, max_length=max_length, padding='max_length',
                                              truncation=True,
                                              return_tensors="pt"
                                              )
        # encoded_sequence['input_ids'] = torch.squeeze(encoded_sequence['input_ids'])
        # encoded_sequence['attention_mask'] = torch.squeeze(encoded_sequence['attention_mask'])
    elif tools['composition']=="official_esm_v2":
        # data = [("", one_seq) for one_seq in seq]

        data = []
        for one_seq in seq:
            if len(one_seq) < tools['max_len']:
                one_seq = one_seq + "<pad>" * (tools['max_len'] - len(one_seq) - 2)
            data.append(("", one_seq))

        _, _, encoded_sequence = tools["tokenizer"](data)
    elif tools['composition'] == "promprot":
        if tools['prm4prmpro'] == 'seq':
            prompts = ['<seq>']
            encoded_sequence = tools["tokenizer"](seq, prompt_toks=prompts)
        elif tools['prm4prmpro'] == 'ppi':
            prompts = ['<ppi>']
            encoded_sequence = tools["tokenizer"](seq, prompt_toks=prompts)
    elif tools['composition'] == "both":
        max_length = tools['max_len']
        encoded_sequence_esm2 = tools["tokenizer"]["tokenizer_esm"](seq, max_length=max_length, padding='max_length',
                                                                    truncation=True,
                                                                    return_tensors="pt"
                                                                    )
        if tools['prm4prmpro'] == 'seq':
            prompts = ['<seq>']
            encoded_sequence_promprot = tools["tokenizer"]["tokenizer_promprot"](seq, prompt_toks=prompts)
        elif tools['prm4prmpro'] == 'ppi':
            prompts = ['<ppi>']
            encoded_sequence_promprot = tools["tokenizer"]["tokenizer_promprot"](seq, prompt_toks=prompts)
        encoded_sequence = {"encoded_sequence_esm2": encoded_sequence_esm2,
                            "encoded_sequence_promprot": encoded_sequence_promprot}
    return encoded_sequence


def print_trainable_parameters(model, logfilepath):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    customlog(logfilepath,
              f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}\n")


def prepare_esm_model(model_name, configs):
    if configs.PEFT == "projector": #additional layers after esm2
        model = ESM2WithCNNandTransformer(model_name)
    else:
        model = EsmModel.from_pretrained(model_name)
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        if configs.PEFT == "lora":
            lora_targets = ["attention.self.query", "attention.self.key","attention.self.value"]
            target_modules = []
            if configs.encoder.lora_lr_num > 0:
                start_layer_idx = np.max([model.config.num_hidden_layers - configs.encoder.lora_lr_num, 0])
                for idx in range(start_layer_idx, model.config.num_hidden_layers):
                    for layer_name in lora_targets:
                        target_modules.append(f"layer.{idx}.{layer_name}")
            
            config = LoraConfig(target_modules=target_modules)
            model = get_peft_model(model, config)
        # elif configs.PEFT == "PromT":
        #     config = PromptTuningConfig(task_type="SEQ_CLS", prompt_tuning_init=PromptTuningInit.TEXT, num_virtual_tokens=8,
        #                             prompt_tuning_init_text="Classify what the peptide type of a protein sequence", tokenizer_name_or_path=configs.encoder.model_name)
        #     model = get_peft_model(model, config)
        #     for param in model.encoder.layer[-1].parameters():
        #         param.requires_grad = True
        #     for param in model.pooler.parameters():
        #         param.requires_grad = False
        elif configs.PEFT == "frozen":
            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False
        elif configs.PEFT == "PFT":
            # Allow the parameters of the last transformer block to be updated during fine-tuning
            for param in model.encoder.layer[configs.train_settings.fine_tune_lr:].parameters():
                param.requires_grad = True
            for param in model.pooler.parameters():
                param.requires_grad = False
        elif configs.PEFT == "lora_PFT":
            config = LoraConfig(target_modules=["query", "key"])
            model = get_peft_model(model, config)
            # Allow the parameters of the last transformer block to be updated during fine-tuning
            for param in model.encoder.layer[configs.train_settings.fine_tune_lr:].parameters():
                param.requires_grad = True
            for param in model.pooler.parameters():
                param.requires_grad = False
    
    return model


def initialize_weights(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)  # Initialize biases to zero


class ESM2WithCNNandTransformer(nn.Module):
    def __init__(self, model_name,num_heads=8, num_encoder_layers=1, cnn_out_channels=256,
                 cnn_kernel_size=1):
        super(ESM2WithCNNandTransformer, self).__init__()

        # # ESM model for extracting features
        # self.esm_model = esm_model
        self.esm_model = EsmModel.from_pretrained(model_name)
        # Freeze all layers
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        class Config:
              def __init__(self, hidden_size):
                  self.hidden_size = hidden_size  # Define hidden_size attribute
        
        self.config = Config(cnn_out_channels)
        # 1D Convolutional layer
        self.conv1d = nn.Conv1d(in_channels=self.esm_model.config.hidden_size, out_channels=cnn_out_channels,
                                kernel_size=cnn_kernel_size, padding='same')

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=cnn_out_channels)

        # Transformer Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=cnn_out_channels, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)


    def forward(self, input_ids,attention_mask):
        # # Forward pass through the ESM2 model to get sequence embeddings
        with torch.no_grad():
             outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
        #
        # # Get sequence embeddings from ESM2
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
        #
        # # Apply 1D Convolution (requires input shape: (batch_size, channels, sequence_length))
        sequence_output = sequence_output.permute(0, 2, 1)  # Change to (batch_size, hidden_size, sequence_length)
        cnn_output = self.conv1d(sequence_output)  # (batch_size, cnn_out_channels, sequence_length)
        cnn_output = cnn_output.permute(0, 2, 1)  # Change back to (batch_size, sequence_length, cnn_out_channels)

        # Apply positional encoding
        cnn_output = self.positional_encoding(cnn_output)

        # Transformer encoder
        transformer_output = self.transformer_encoder(cnn_output)

        # Pooling: use mean pooling or CLS token
        # pooled_output = torch.mean(transformer_output, dim=1)  # (batch_size, cnn_out_channels)
        #batch_size,seqlen,features_dim = transformer_output.shape
        #x = transformer_output.reshape(-1,features_dim)
        return transformer_output

class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels, output_dim, kernel_sizes, stride, padding, droprate=0.3, inner_linear_dim=128,num_layers=2):
        super(MultiScaleCNN, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding) for kernel_size in kernel_sizes
        ])
        self.batchnorm1 = nn.BatchNorm1d(out_channels * len(kernel_sizes))  # BatchNorm after the first Conv layer
        self.conv_layers2 = nn.ModuleList([
            nn.Conv1d(out_channels * len(kernel_sizes), out_channels, kernel_size, stride, padding) for kernel_size in kernel_sizes
        ])
        self.batchnorm2 = nn.BatchNorm1d(out_channels * len(kernel_sizes))  # BatchNorm after the second Conv layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        #self.fc_shared = nn.Linear(out_channels * len(kernel_sizes), inner_linear_dim)  # Shared fully connected layer
        #self.fc_multiclass = nn.Linear(inner_linear_dim, output_dim)  # Output layer for multi-class classification
        #use on 7.25.2024 4:19 
        self.fc_multiclass = MoBYMLP(in_dim = out_channels * len(kernel_sizes),inner_dim = inner_linear_dim,out_dim = output_dim,num_layers=num_layers)


    def forward(self, x):
        conv_results = [conv_layer(x) for conv_layer in self.conv_layers]
        x = torch.cat(conv_results, dim=1)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        conv_results2 = [conv_layer(x) for conv_layer in self.conv_layers2]
        x = torch.cat(conv_results2, dim=1)
        x = self.batchnorm2(x)
        x = self.relu(x)
        
        x = x.permute(0, 2, 1)  # batch, length, out_channels
        #x = self.fc_shared(x)
        #print(x.shape)
        x = self.fc_multiclass(x)  # batch, length, output_dim
        return x



class MoBYMLP(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(MoBYMLP, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True)) #relu cannot be used with sigmoid!!! smallest will be 0.5?
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim,
                                    out_dim) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        #print("mlp forward")
        #print(x.shape)  #[128,512,100]
        batch_size,seqlen,features_dim = x.shape
        x = x.reshape(-1,features_dim)
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        x = x.reshape(batch_size,seqlen,-1)
        return x

#this was used 
class ParallelmultiscaleCNN_Linear_Decoders(nn.Module):
    def __init__(self, input_size, inner_linear_dim = 100, num_layers=2,out_channels=8, kernel_size=14, droprate=0.3,output_act=None):
        super(ParallelmultiscaleCNN_Linear_Decoders, self).__init__()
        
        self.num_decoders = 8 
        self.decoders = nn.ModuleList(
                #0-3: dnn
                [MoBYMLP(in_dim = input_size,inner_dim = inner_linear_dim,out_dim = output_size,num_layers=num_layers) for output_size in [1] * 4]+
                #4-5: cnn
                [MultiScaleCNN(in_channels=input_size, out_channels=out_channels,output_dim=output_size,
                kernel_sizes=kernel_size, stride=1,
                          padding='same', droprate=droprate,inner_linear_dim=inner_linear_dim,num_layers=num_layers)
                          for output_size in [1] * 2]+
                #6-7: dnn
                [MoBYMLP(in_dim = input_size,inner_dim = inner_linear_dim,out_dim = output_size,num_layers=num_layers) for output_size in [1] * 2]
                
        
        )
        self.activation = output_act
    
    def forward(self, x):
        #x is [batch,L,D]
        # decoder_outputs = [decoder(x) for decoder in self.linear_decoders]
        decoder_outputs = []
        for motif_index in range(self.num_decoders):
            
            if motif_index <4:
                #outputs = self.linear_decoders[motif_index](x).squeeze(-1)  # should be [batch,maxlen-2,1]
                outputs = self.decoders[motif_index](x).squeeze(-1) # should be [batch,maxlen-2,1]
            elif motif_index >= 4 and motif_index<6:
                #outputs = self.cnn_decoders[motif_index-5](x.permute(0, 2, 1)).squeeze(-1)
                outputs = self.decoders[motif_index](x.permute(0, 2, 1)).squeeze(-1)
            else:
                #outputs = self.linear_decoders[motif_index-2](x).squeeze(-1)  # sould be [batch,maxlen-2,1]
                outputs = self.decoders[motif_index](x).squeeze(-1)  # sould be [batch,maxlen-2,1]
            
            if self.activation == 'tanh':# -1 1
               decoder_outputs.append(torch.tanh(outputs))
            elif self.activation == 'sigmoid': # 0 1
               decoder_outputs.append(torch.sigmoid(outputs))
            else:
               decoder_outputs.append(outputs)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)  # [batch, num_class-1, maxlen-2]
        return decoder_outputs# [batch,8, maxlen-2]
    



def remove_s_e_token(target_tensor, mask_tensor):  # target_tensor [batch, seq+2, ...]  =>  [batch, seq, ...]
    # mask_tensor=inputs['attention_mask']
    # input_tensor=inputs['input_ids']
    result = []
    for i in range(mask_tensor.size()[0]):
        ind = torch.where(mask_tensor[i] == 0)[0]
        if ind.size()[0] == 0:
            result.append(target_tensor[i][1:-1])
        else:
            eos_ind = ind[0].item() - 1
            result.append(torch.concatenate((target_tensor[i][1:eos_ind], target_tensor[i][eos_ind + 1:]), axis=0))

    new_tensor = torch.stack(result, axis=0)
    return new_tensor


class Encoder(nn.Module):
    def __init__(self, configs, model_name='facebook/esm2_t33_650M_UR50D', model_type='esm_v2'):
        super().__init__()
        self.model_type = model_type
        self.PEFT = configs.PEFT
        if model_type == 'esm_v2':
            self.model = prepare_esm_model(model_name, configs)
        # self.pooling_layer = nn.AdaptiveAvgPool2d((None, 1))
        self.combine = configs.decoder.combine
        self.apply_DNN = configs.decoder.apply_DNN
     
        if configs.decoder.type == "multiscalecnn-linear":
            if not hasattr(configs.decoder,"output_act"):
                  configs.decoder.output_act = None
            
            self.ParallelDecoders = ParallelmultiscaleCNN_Linear_Decoders(input_size=self.model.config.hidden_size,
                                                                inner_linear_dim = configs.decoder.inner_linear_dim,
                                                                num_layers = configs.decoder.linear_num_layers,
                                                                out_channels=configs.decoder.cnn_channel,
                                                                kernel_size=configs.decoder.cnn_kernel,
                                                                droprate=configs.decoder.droprate,
                                                                output_act = configs.decoder.output_act,
                                                                )
        

        self.overlap = configs.encoder.frag_overlap

        self.predict_max = configs.train_settings.predict_max


    def get_pro_class(self, predict_max, id, id_frags_list, seq_frag_tuple, motif_logits, overlap):
        # motif_logits_max, _ = torch.max(motif_logits, dim=-1, keepdim=True).squeeze(-1) #should be [batch,num_class]
        # print(motif_logits_max)
        motif_pro_list = []
        motif_pro_list_dnn = []
        for id_protein in id:
            ind_frag = 0
            #print("in get_pro_class model.py")
            #print(id_protein)
            id_frag = str(id_protein) + "@" + str(ind_frag)
            #print("id_frags_list in model.py")
            #print(id_frags_list)
            while id_frag in id_frags_list:
                ind = id_frags_list.index(id_frag)
                motif_logit = motif_logits[ind]  # [num_class,max_len]
                seq_frag = seq_frag_tuple[ind]
                l = len(seq_frag)
                if ind_frag == 0:
                    motif_pro = motif_logit[:, :l]  # [num_class,length]
                else:
                    overlap_motif = (motif_pro[:, -overlap:] + motif_logit[:, :overlap]) / 2
                    motif_pro = torch.concatenate((motif_pro[:, :-overlap], overlap_motif, motif_logit[:, overlap:l]),
                                                  axis=-1)
                ind_frag += 1
                id_frag = id_protein + "@" + str(ind_frag)

            if predict_max:
                # print('-before max', motif_pro.shape)  # should be [num_class,length]
                # min_pool = torch.min(motif_pro.clone()[0, :])
                max_pool = torch.max(motif_pro, dim=1)[0]
                _motif_pro = max_pool
                # print('-after max', motif_pro.shape)  # should be [num_class]
                # print(motif_pro)
            else:
                # print('-before mean', motif_pro.shape)  # should be [num_class,length]
                # motif_pro = torch.mean(motif_pro, dim=-1)
                _motif_pro = torch.mean(motif_pro.clone(), dim=-1)  # yichuan 0605
                # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
                # 只在特定版本pytorch上出现此错误
                # print('-after mean', motif_pro.shape)  # should be [num_class]

            # motif_pro_list_dnn.append(motif_pro)
            motif_pro_list.append(_motif_pro.clone())  # [batch,num_class]

        motif_pro_list = torch.stack(motif_pro_list, dim=0)
        # motif_pro_list_dnn = torch.stack(motif_pro_list_dnn, dim=0)
        #print('motif_pro_list', motif_pro_list.shape) #[16, 9] Batchsize, num_classes
        #print('motif_pro_list_dnn', motif_pro_list_dnn)
        return motif_pro_list

    def forward(self, encoded_sequence, id, id_frags_list, seq_frag_tuple):
        """
        Batch is built before forward(), in train_loop()
        Batch is either (anchor) or (anchor+pos+neg)
        """
        classification_head = None
        classification_head_other = None
        motif_logits = None

        features = self.model(input_ids=encoded_sequence['input_ids'],
                              attention_mask=encoded_sequence['attention_mask'])
        # print(features)
        if self.PEFT == "projector":
            last_hidden_state = remove_s_e_token(features,encoded_sequence['attention_mask'])  # [batch, maxlen-2, dim]
        else:# print(features)
            last_hidden_state = remove_s_e_token(features.last_hidden_state,encoded_sequence['attention_mask'])  # [batch, maxlen-2, dim]

        """CASE D"""
        """这"""
        motif_logits = self.ParallelDecoders(
            last_hidden_state)  # list no shape # last_hidden_state=[batch, maxlen-2, dim] motif_logits [B,8,L]
        motif_logits_other = 1 - torch.max(motif_logits,dim = 1)[0] #[B,1,L]
        motif_logits_all = torch.cat((motif_logits_other.unsqueeze(1), motif_logits),dim = 1)
        #print("motif_logits_all shape")
        #print(motif_logits_all.shape) #should be [B,9,L]
        if self.combine:
            classification_head = self.get_pro_class(self.predict_max, id, id_frags_list, seq_frag_tuple,
                                                     motif_logits, self.overlap)
            classification_head_other = 1-torch.max(classification_head,dim=-1)[0]
            classification_head_all = torch.cat((classification_head_other.unsqueeze(1),classification_head),dim=1)
            
        
        return classification_head_all, motif_logits_all




def prepare_models(configs, logfilepath, curdir_path):
    if configs.encoder.composition == "esm_v2":
        encoder = Encoder(model_name=configs.encoder.model_name,
                          model_type=configs.encoder.model_type,
                          configs=configs
                          )
    if configs.encoder.composition=="official_esm_v2":
        encoder = OfficialEsmEncoder(model_name=configs.encoder.model_name,
                            model_type=configs.encoder.model_type,
                            configs=configs
                            )

    if not logfilepath == "":
        print_trainable_parameters(encoder, logfilepath)
    return encoder


class MaskedLMDataCollator:
    """Data collator for masked language modeling.

    The idea is based on the implementation of DataCollatorForLanguageModeling at
    https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L751C7-L782
    """

    def __init__(self, batch_converter, mlm_probability=0.15):
        """_summary_

        Args:
            mlm_probability (float, optional): The probability with which to (randomly) mask tokens in the input. Defaults to 0.15.
        """
        self.mlm_probability = mlm_probability
        """github ESM2
        self.special_token_indices = [batch_converter.alphabet.cls_idx, 
                                batch_converter.alphabet.padding_idx, 
                                batch_converter.alphabet.eos_idx,
                                #batch_converter.eos_token_id,
                                batch_converter.alphabet.unk_idx, 
                                batch_converter.alphabet.mask_idx,
                                #batch_converter.mask_token_id
                                ]
        """
        self.special_token_indices = batch_converter.all_special_ids
        # self.vocab_size = batch_converter.alphabet.all_toks.__len__()#github esm2
        self.vocab_size = batch_converter.all_tokens.__len__()
        # self.mask_idx = batch_converter.alphabet.mask_idx #github esm2
        self.mask_idx = batch_converter.mask_token_id  # huggingface

    def get_special_tokens_mask(self, tokens):
        return [1 if token in self.special_token_indices else 0 for token in tokens]

    def mask_tokens(self, batch_tokens):
        """make a masked input and label from batch_tokens.

        Args:
            batch_tokens (tensor): tensor of batch tokens
            batch_converter (tensor): batch converter from ESM-2.

        Returns:
            inputs: inputs with masked
            labels: labels for masked tokens
        """
        ## mask tokens
        inputs = batch_tokens.clone().to(batch_tokens.device)  # clone not work for huggingface tensor! don't know why!
        labels = batch_tokens.clone().to(batch_tokens.device)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = [self.get_special_tokens_mask(val) for val in labels]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # must remove this otherwise, inputs will be change as well
        # labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_idx

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size,
                                     labels.shape, dtype=torch.long).to(batch_tokens.device)

        # print(indices_random)
        # print(random_words)
        # print(inputs)

        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

"""
2024 06 11 adapter addition
"""
# from Jiang, ParallelLinearDecoders
class ParallelLinear(nn.Module):
    def __init__(self, input_size, output_sizes):
        super(ParallelLinear, self).__init__()
        self.linear_decoders = nn.ModuleList([
            nn.Linear(input_size, output_size) for output_size in output_sizes
        ])

    def forward(self, x):
        decoder_outputs = [decoder(x) for decoder in self.linear_decoders]
        return decoder_outputs


class ProteinCNN(nn.Module):
    def __init__(self, input_dim, num_filters, filter_size):
        super(ProteinCNN, self).__init__()
        # Convolutional layer
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=filter_size, padding='same')
        # Linear layer to transform the output to [batch, seq_length]
        self.linear = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = x.squeeze(-1)
        return x


import esm_utilities

class OfficialEsmEncoder(nn.Module):
    def __init__(self, configs, model_name='esm2_t33_650M_UR50D', model_type='esm_v2'):
        super().__init__()
        self.model_type = model_type
        if model_type == 'official_esm_v2':
            self.model = esm_utilities.load_model(
                model_architecture=configs.encoder.model_name,
                num_end_adapter_layers=configs.encoder.adapter_lr_num)
            
            if configs.PEFT == "PrefixPrompt":
                # TODO(Yuexu): change the hyperparameter 
                # (prompt_len and prompt_layer_indices) for your case.
                prompt_model = PrefixTuning(self.model, 
                                            prompt_len=configs.encoder.prompt_len, 
                                            prompt_layer_indices=configs.encoder.prompt_layer_indices  #[0,5,10,15,20,25,30]
                                            ) 
                                            # prompt_layer_indices=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]) 
                self.model.prefix_module = prompt_model

            for param in self.model.parameters():
                param.requires_grad = False

            if configs.PEFT == "Adapter":
                for name, param in self.model.named_parameters():
                    if "adapter_layer" in name:
                        param.requires_grad = True

            if configs.PEFT == "Adapter_PFT":
                for p in self.model.layers[configs.train_settings.fine_tune_lr:].parameters():
                    p.requires_grad = True
                for name, param in self.model.named_parameters():
                    if "adapter_layer" in name:
                        param.requires_grad = True

            elif configs.PEFT == "PrefixPrompt":
                for name, param in self.model.named_parameters():
                    if "prefix_module" in name:
                        param.requires_grad = True

        self.combine = configs.decoder.combine
        self.apply_DNN = configs.decoder.apply_DNN
        if configs.decoder.type == "multiscalecnn-linear":
            if not hasattr(configs.decoder,"output_act"):
                  configs.decoder.output_act = None
            
            self.ParallelDecoders = ParallelmultiscaleCNN_Linear_Decoders(input_size=1280,
                                                                inner_linear_dim = configs.decoder.inner_linear_dim,
                                                                num_layers = configs.decoder.linear_num_layers,
                                                                out_channels=configs.decoder.cnn_channel,
                                                                kernel_size=configs.decoder.cnn_kernel,
                                                                droprate=configs.decoder.droprate,
                                                                output_act = configs.decoder.output_act,
                                                                )

        self.overlap = configs.encoder.frag_overlap
        self.predict_max = configs.train_settings.predict_max
        self.num_layers = self.model.num_layers

    def get_pro_class(self, predict_max, id, id_frags_list, seq_frag_tuple, motif_logits, overlap):
        # motif_logits_max, _ = torch.max(motif_logits, dim=-1, keepdim=True).squeeze(-1) #should be [batch,num_class]
        # print(motif_logits_max)
        motif_pro_list = []
        motif_pro_list_dnn = []
        for id_protein in id:
            ind_frag = 0
            #print("in get_pro_class model.py")
            #print(id_protein)
            id_frag = str(id_protein) + "@" + str(ind_frag)
            #print("id_frags_list in model.py")
            #print(id_frags_list)
            while id_frag in id_frags_list:
                ind = id_frags_list.index(id_frag)
                motif_logit = motif_logits[ind]  # [num_class,max_len]
                seq_frag = seq_frag_tuple[ind]
                l = len(seq_frag)
                if ind_frag == 0:
                    motif_pro = motif_logit[:, :l]  # [num_class,length]
                else:
                    overlap_motif = (motif_pro[:, -overlap:] + motif_logit[:, :overlap]) / 2
                    motif_pro = torch.concatenate((motif_pro[:, :-overlap], overlap_motif, motif_logit[:, overlap:l]),
                                                  axis=-1)
                ind_frag += 1
                id_frag = id_protein + "@" + str(ind_frag)

            if predict_max:
                # print('-before max', motif_pro.shape)  # should be [num_class,length]
                # min_pool = torch.min(motif_pro.clone()[0, :])
                max_pool = torch.max(motif_pro, dim=1)[0]
                _motif_pro = max_pool
                # print('-after max', motif_pro.shape)  # should be [num_class]
                # print(motif_pro)
            else:
                # print('-before mean', motif_pro.shape)  # should be [num_class,length]
                # motif_pro = torch.mean(motif_pro, dim=-1)
                _motif_pro = torch.mean(motif_pro.clone(), dim=-1)  # yichuan 0605
                # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
                # 只在特定版本pytorch上出现此错误
                # print('-after mean', motif_pro.shape)  # should be [num_class]

            # motif_pro_list_dnn.append(motif_pro)
            motif_pro_list.append(_motif_pro.clone())  # [batch,num_class]

        motif_pro_list = torch.stack(motif_pro_list, dim=0)
        # motif_pro_list_dnn = torch.stack(motif_pro_list_dnn, dim=0)
        #print('motif_pro_list', motif_pro_list.shape) #[16, 9] Batchsize, num_classes
        #print('motif_pro_list_dnn', motif_pro_list_dnn)
        return motif_pro_list

    def forward(self, encoded_sequence, id, id_frags_list, seq_frag_tuple):
        classification_head = None
        classification_head_other = None
        motif_logits = None
        
        features = self.model(
            tokens=encoded_sequence,
            repr_layers=[self.num_layers])["representations"][self.num_layers]

        last_hidden_state = features[:, 1:-1]  # [batch, maxlen-2, dim]

        """CASE D"""
        """这"""
        motif_logits = self.ParallelDecoders(
            last_hidden_state)  # list no shape # last_hidden_state=[batch, maxlen-2, dim] motif_logits [B,8,L]
        motif_logits_other = 1 - torch.max(motif_logits,dim = 1)[0] #[B,1,L]
        motif_logits_all = torch.cat((motif_logits_other.unsqueeze(1), motif_logits),dim = 1)
        #print("motif_logits_all shape")
        #print(motif_logits_all.shape) #should be [B,9,L]
        if self.combine:
            classification_head = self.get_pro_class(self.predict_max, id, id_frags_list, seq_frag_tuple,
                                                     motif_logits, self.overlap) #[batch, num_class]
            classification_head_other = 1-torch.max(classification_head,dim=-1)[0]
            classification_head_all = torch.cat((classification_head_other.unsqueeze(1),classification_head),dim=1) #[batch, num_class+1]
            
        
        return classification_head_all, motif_logits_all


# if __name__ == '__main__':










