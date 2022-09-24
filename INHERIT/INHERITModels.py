# Author: Zeheng Bai
##### DNABERT AND INHERIT MODELS #####
from basicsetting import *
from transformers.tokenization_dna import DNATokenizer
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import BertModel
from transformers import BatchEncoding

class Baseline_DNABERT(torch.nn.Module):
    '''DNABERT: without pre-trained models'''
    def __init__(self, config, freeze_bert=True, bert_dir=''):
        super(Baseline_DNABERT, self).__init__()
        if bert_dir == '':
            self.bert = BertForSequenceClassification(config=config)
        else:
            self.bert = BertForSequenceClassification.from_pretrained(bert_dir)
        if freeze_bert:
            self.bert.requires_grad_(False)
        self.regressor = torch.nn.Linear(2, 1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, return_dict=True)
        cls = bert_output['logits']
        out = self.regressor(cls)
        return out

class Baseline_IHT(torch.nn.Module):
    '''INHERIT: with two pre-trained models'''
    def __init__(self, config, bac_bert_dir, pha_bert_dir, freeze_bert=True):
        super(Baseline_IHT, self).__init__()
        self.bacbert = BertForSequenceClassification.from_pretrained(bac_bert_dir)
        self.phabert = BertForSequenceClassification.from_pretrained(pha_bert_dir)
        if freeze_bert:
            self.bacbert.requires_grad_(False)
            self.phabert.requires_grad_(False)
        self.regressor = torch.nn.Linear(4, 1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        bac_bert_output = self.bacbert(input_ids=input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, return_dict=True)
        pha_bert_output = self.phabert(input_ids=input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, return_dict=True)
        bac_cls = bac_bert_output['logits']
        pha_cls = pha_bert_output['logits']
        cls = torch.cat((bac_cls, pha_cls), axis=1)
        out = self.regressor(cls)
        return out


class Baseline_DNABERT_prob(torch.nn.Module):
    '''DNABERT: without pre-trained models'''
    def __init__(self, config, freeze_bert=True, bert_dir=''):
        super(Baseline_DNABERT_prob, self).__init__()
        if bert_dir == '':
            self.bert = BertModel(config=config, output_hidden_states=True)
        else:
            self.bert = BertModel.from_pretrained(bert_dir, output_hidden_states=True)
        if freeze_bert:
            self.bert.requires_grad_(False)
        self.bn = torch.nn.BatchNorm1d(768, affine=False)
        self.pooler = torch.nn.Linear(768, 768) 
        self.act = torch.nn.Tanh()
        self.regressor = torch.nn.Linear(768, 1)
        

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, return_dict=True)
        hidden = bert_output['hidden_states'][5][:, 0, :]
        # out = self.bn(hidden)
        #out = self.pooler(hidden)
        #out = self.act(out)
        out = self.regressor(hidden) 
        #out = self.regressor(out)
        return out

