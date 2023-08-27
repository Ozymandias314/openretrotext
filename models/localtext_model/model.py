import torch.nn as nn
from dgllife.model import MPNNGNN

from models.localtext_model.model_utils import \
    pair_atom_feats, unbatch_mask, unbatch_feats, Global_Reactivity_Attention

from models.localtext_model.tokenizer import get_tokenizers
from transformers import get_scheduler, EncoderDecoderModel, EncoderDecoderConfig, AutoTokenizer, AutoConfig, AutoModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead

from models.localtext_model.textreact_utils import expand_position_embeddings


class LocalText(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats,
                 edge_hidden_feats,
                 num_step_message_passing,
                 attention_heads,
                 attention_layers,
                 AtomTemplate_n, 
                 BondTemplate_n,
                 encoder_path,
                 encoder_tokenizer,
                 max_length):
        super(LocalText, self).__init__()
                
        self.mpnn = MPNNGNN(node_in_feats=node_in_feats,
                            node_out_feats=node_out_feats,
                            edge_in_feats=edge_in_feats,
                            edge_hidden_feats=edge_hidden_feats,
                            num_step_message_passing=num_step_message_passing)
        
        self.linearB = nn.Linear(node_out_feats*2, node_out_feats)

        self.att = Global_Reactivity_Attention(node_out_feats, attention_heads, attention_layers)

        self.encoder = get_encoder(encoder_path, max_length)

        self.atom_linear = nn.Sequential(
            nn.Linear(node_out_feats + 768, node_out_feats),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(node_out_feats, AtomTemplate_n+1))
        self.bond_linear = nn.Sequential(
            nn.Linear(node_out_feats + 768, node_out_feats),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(node_out_feats, BondTemplate_n+1))
    def forward(self, g, node_feats, edge_feats, input_ids, attention_mask):
        node_feats = self.mpnn(g, node_feats, edge_feats)
        atom_feats = node_feats
        bond_feats = self.linearB(pair_atom_feats(g, node_feats))
        edit_feats, mask = unbatch_mask(g, atom_feats, bond_feats)
        attention_score, edit_feats = self.att(edit_feats, mask)

        text_feats = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0:1, :]

        text_feats = text_feats.view(text_feats.size()[0], 768)



        atom_feats, bond_feats = unbatch_feats(g, edit_feats, text_feats)
        atom_outs = self.atom_linear(atom_feats)
        bond_outs = self.bond_linear(bond_feats) 

        return atom_outs, bond_outs, attention_score

def get_encoder(encoder_path, max_length):

  
    encoder = AutoModel.from_pretrained(encoder_path)

    if max_length > encoder.config.max_position_embeddings:
        expand_position_embeddings(encoder, max_length)

    return encoder
