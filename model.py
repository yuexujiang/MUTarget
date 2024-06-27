
from peft import get_peft_model
from peft import LoraConfig, PromptTuningConfig, PromptTuningInit
import torch
from torch import nn
torch.manual_seed(0)
import os
from transformers import AutoTokenizer
from PromptProtein.utils import PromptConverter
from PromptProtein.models import openprotein_promptprotein
from transformers import EsmModel
from utils import customlog

import esm_utilities
from prompt_tunning import PrefixTuning



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
        cs_pred = cs_head[:,1:]
        # print("peptide_probab size ="+str(type_probab.size())) 
        return type_probab, cs_pred
    
def prepare_tokenizer(configs, curdir_path):
    if configs.encoder.composition=="esm_v2":
        tokenizer = AutoTokenizer.from_pretrained(configs.encoder.model_name)
    elif configs.encoder.composition=="promprot":
        model, dictionary = openprotein_promptprotein(os.path.join(curdir_path, "PromptProtein", "PromptProtein.pt"))
        tokenizer = PromptConverter(dictionary)
    elif configs.encoder.composition=="both":
        tokenizer_esm = AutoTokenizer.from_pretrained(configs.encoder.model_name)
        model, dictionary = openprotein_promptprotein(os.path.join(curdir_path, "PromptProtein", "PromptProtein.pt"))
        tokenizer_promprot = PromptConverter(dictionary)
        tokenizer={"tokenizer_esm":tokenizer_esm, "tokenizer_promprot":tokenizer_promprot}
    else:
        tokenizer = None
    return tokenizer

def tokenize(tools, seq):
    if tools['composition']=="esm_v2":
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
    elif tools['composition']=="promprot":
        if tools['prm4prmpro']=='seq':
            prompts = ['<seq>']
            encoded_sequence = tools["tokenizer"](seq, prompt_toks=prompts)
        elif tools['prm4prmpro']=='ppi':
            prompts = ['<ppi>']
            encoded_sequence = tools["tokenizer"](seq, prompt_toks=prompts)
    elif tools['composition']=="both":
        max_length = tools['max_len']
        encoded_sequence_esm2 = tools["tokenizer"]["tokenizer_esm"](seq, max_length=max_length, padding='max_length',
                                                  truncation=True,
                                                  return_tensors="pt"
                                                  )
        if tools['prm4prmpro']=='seq':
            prompts = ['<seq>']
            encoded_sequence_promprot = tools["tokenizer"]["tokenizer_promprot"](seq, prompt_toks=prompts)
        elif tools['prm4prmpro']=='ppi':
            prompts = ['<ppi>']
            encoded_sequence_promprot = tools["tokenizer"]["tokenizer_promprot"](seq, prompt_toks=prompts)
        encoded_sequence={"encoded_sequence_esm2":encoded_sequence_esm2, "encoded_sequence_promprot":encoded_sequence_promprot}
    return encoded_sequence

def print_trainable_parameters(model,logfilepath):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        customlog(logfilepath, f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}\n")
    


def prepare_esm_model(model_name, configs):
    model = EsmModel.from_pretrained(model_name)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    if configs.PEFT == "lora":
        config = LoraConfig(target_modules=["query", "key"])
        model = get_peft_model(model, config)
        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in model.pooler.parameters():
            param.requires_grad = False
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
        for param in model.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in model.pooler.parameters():
            param.requires_grad = False
    return model

class ParallelLinearDecoders(nn.Module):
    def __init__(self, input_size, output_sizes):
        super(ParallelLinearDecoders, self).__init__()
        self.linear_decoders = nn.ModuleList([
            nn.Linear(input_size, output_size) for output_size in output_sizes
        ])

    def forward(self, x):
        decoder_outputs = [decoder(x) for decoder in self.linear_decoders]
        return decoder_outputs

class Encoder(nn.Module):
    def __init__(self, configs, model_name='facebook/esm2_t33_650M_UR50D', model_type='esm_v2'):
        super().__init__()
        self.model_type = model_type
        if model_type == 'esm_v2':
            self.model = prepare_esm_model(model_name, configs)
        # self.pooling_layer = nn.AdaptiveAvgPool2d((None, 1))
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.ParallelLinearDecoders = ParallelLinearDecoders(input_size=self.model.config.hidden_size, 
                                                             output_sizes=[1] * configs.encoder.num_classes)
        # self.mhatt = nn.MultiheadAttention(embed_dim=320, num_heads=10, batch_first=True)
        # self.attheadlist = []
        # self.headlist = []
        # for i in range(9):
            # self.attheadlist.append(nn.MultiheadAttention(embed_dim=320, num_heads=1, batch_first=True))
            # self.headlist.append(nn.Linear(320, 1))
        # self.device = device
        # self.device=configs.train_settings.device
    def forward(self, encoded_sequence):
        features = self.model(input_ids=encoded_sequence['input_ids'], attention_mask=encoded_sequence['attention_mask'],output_attentions=True)
        last_hidden_state = features.last_hidden_state[:,1:-1] #[batch, seq+2, dim]
        # print(last_hidden_state.size())
        # attention_mask = encoded_sequence['attention_mask'].repeat_interleave(encoded_sequence['attention_mask'].size(1), dim=0).reshape(
        #     -1, encoded_sequence['attention_mask'].size(1), encoded_sequence['attention_mask'].size(1))
        #attention_mask size is [batch, seq+2, seq+2]
        # attention_mask=attention_mask.to(dtype=torch.float)
        # motif_logits=[]
        motif_logits = self.ParallelLinearDecoders(last_hidden_state)


        # print(attention_mask.device)
        # for i in range(9):
            # attn_output = self.attheadlist[i](query=last_hidden_state, key=last_hidden_state, value=last_hidden_state, 
            #                                   attn_mask=attention_mask, need_weights=False) #[batch, seq+2, dim]
            # print(last_hidden_state.device)
            # logits = self.headlist[i](last_hidden_state).squeeze(dim=-1) #[batch, seq+2]
            # motif_logits.append(logits[:,1:-1])
            

        # attn_output, attn_output_weights = self.mhatt(query=last_hidden_state, key=last_hidden_state, value=last_hidden_state, 
        #                                                   attn_mask=attention_mask, average_attn_weights=False) 
        # #attn_output_weights = [batch,num_heads,seq+2, seq+2]
        # motif_logits=[]
        # for i in range(9):
        #     head = attn_output_weights[:,i] #[batch,seq+2, seq+2]
        #     logits = self.pooling_layer(head).squeeze(-1) #[batch,seq+2]
        #     print(logits[0])
        #     motif_logits.append(logits[:,1:-1])
            

        motif_logits = torch.stack(motif_logits, dim=1).squeeze(-1)
        # print(motif_logits.size())
        

        # print(encoded_sequence['input_ids'].size())
        # print(encoded_sequence['attention_mask'].size())
        # last_layer_attentions = features.attentions[-1]  # [batch, head, seq, seq]
        # pooled_features = self.pooling_layer(last_layer_attentions).squeeze(-1) # [batch, head, seq]
        # motif_pred = self.cs_head(pooled_features)
        # motif_logits = pooled_features[:, 0:9, 1:-1]
        return motif_logits

class OfficialEsmEncoder(nn.Module):
    def __init__(self, configs, model_name='esm2_t33_650M_UR50D', model_type='esm_v2'):
        super().__init__()
        self.model_type = model_type
        if model_type == 'official_esm_v2':
            self.model = esm_utilities.load_model(
                model_architecture=configs.encoder.model_name,
                num_end_adapter_layers=2)
            
            if configs.PEFT == "PrefixPrompt":
                # TODO(Yuexu): change the hyperparameter 
                # (prompt_len and prompt_layer_indices) for your case.
                prompt_model = PrefixTuning(self.model, 
                                            prompt_len=10, 
                                            prompt_layer_indices=[0]) 
                self.model.prefix_module = prompt_model

            for param in self.model.parameters():
                param.requires_grad = False
            
            if configs.PEFT == "Adapter":
                for name, param in self.model.named_parameters():
                    if "adapter_layer" in name:
                        param.requires_grad = True
                        
            elif configs.PEFT == "PrefixPrompt":
                for name, param in self.model.named_parameters():
                    if "prefix_module" in name:
                        param.requires_grad = True
            
            print("**********************************")
            print("Trainbel parameters:")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"{name}, {str(param.data.shape)}")
            print("**********************************")
            # exit(0)
            
        # self.pooling_layer = nn.AdaptiveAvgPool2d((None, 1))
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.ParallelLinearDecoders = ParallelLinearDecoders(input_size=self.model.embed_dim, 
                                                             output_sizes=[1] * configs.encoder.num_classes)
        self.num_layers = self.model.num_layers

    def forward(self, encoded_sequence):
        features = self.model(
            tokens=encoded_sequence, 
            repr_layers=[self.num_layers])["representations"][self.num_layers]
        
        last_hidden_state = features[:,1:-1] #[batch, seq+2, dim]
        
        motif_logits = self.ParallelLinearDecoders(last_hidden_state)
        motif_logits = torch.stack(motif_logits, dim=1).squeeze(-1)
        return motif_logits


class Bothmodels(nn.Module):
    def __init__(self, configs, pretrain_loc, trainable_layers, model_name='facebook/esm2_t33_650M_UR50D', model_type='esm_v2'):
        super().__init__()
        self.model_esm = prepare_esm_model(model_name, configs)
        self.model_promprot = initialize_PromptProtein(pretrain_loc, trainable_layers)
        self.pooling_layer = nn.AdaptiveAvgPool1d(output_size=1)
        self.head = nn.Linear(self.model_esm.embeddings.position_embeddings.embedding_dim+1280, configs.encoder.num_classes)
        self.cs_head = nn.Linear(self.model_esm.embeddings.position_embeddings.embedding_dim, 1)
    
    def forward(self, encoded_sequence):
        features = self.model_esm(input_ids=encoded_sequence["encoded_sequence_esm2"]['input_ids'], 
                              attention_mask=encoded_sequence["encoded_sequence_esm2"]['attention_mask'])
        transposed_feature = features.last_hidden_state.transpose(1, 2)
        pooled_features_esm2 = self.pooling_layer(transposed_feature).squeeze(2)

        cs_head = self.cs_head(features.last_hidden_state).squeeze(dim=-1)
        cs_pred = cs_head[:,1:201]

        features = self.model_promprot(encoded_sequence["encoded_sequence_promprot"], with_prompt_num=1)['logits']
        transposed_feature = features.transpose(1, 2)
        pooled_features_promprot = self.pooling_layer(transposed_feature).squeeze(2)

        pooled_features = torch.cat((pooled_features_esm2, pooled_features_promprot),dim=1)

        classification_head = self.head(pooled_features)
        
        return classification_head, cs_pred
    
def prepare_models(configs, logfilepath, curdir_path):
    if configs.encoder.composition=="esm_v2":
        encoder = Encoder(model_name=configs.encoder.model_name,
                      model_type=configs.encoder.model_type,
                      configs=configs
                      )
    if configs.encoder.composition=="official_esm_v2":
        encoder = OfficialEsmEncoder(model_name=configs.encoder.model_name,
                            model_type=configs.encoder.model_type,
                            configs=configs
                            )

    elif configs.encoder.composition=="promprot":
        encoder=CustomPromptModel(configs=configs, pretrain_loc=os.path.join(curdir_path, "PromptProtein", "PromptProtein.pt"), trainable_layers=["layers.32", "emb_layer_norm_after"])
    elif configs.encoder.composition=="both":
        encoder=Bothmodels(configs=configs, pretrain_loc=os.path.join(curdir_path, "PromptProtein", "PromptProtein.pt"), 
                           trainable_layers=[], model_name=configs.encoder.model_name, model_type=configs.encoder.model_type)
    if not logfilepath == "":
        print_trainable_parameters(encoder, logfilepath)
    return encoder

# if __name__ == '__main__':










