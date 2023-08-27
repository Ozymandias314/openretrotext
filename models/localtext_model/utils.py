import torch
import dgl
import errno
import json
import os
import pandas as pd
from functools import partial

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

from dgl.data.utils import Subset
from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph, EarlyStopping

from models.localtext_model.model import LocalText
from models.localtext_model.dataset import USPTODataset, USPTOTestDataset


def init_featurizer(args):
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs']
    args['node_featurizer'] = WeaveAtomFeaturizer(atom_types = atom_types)
    args['edge_featurizer'] = CanonicalBondFeaturizer(self_loop=True)
    return args

def get_configure(args):
    with open(args['config_path'], 'r') as f:
        config = json.load(f)
    global AtomTemplate_n, BondTemplate_n
    AtomTemplate_n = len(pd.read_csv('%s/atom_templates.csv' % args['data_dir']))
    BondTemplate_n = len(pd.read_csv('%s/bond_templates.csv' % args['data_dir']))
    args['AtomTemplate_n'], config['AtomTemplate_n'] = AtomTemplate_n, AtomTemplate_n
    args['BondTemplate_n'], config['BondTemplate_n'] = BondTemplate_n, BondTemplate_n
    config['in_node_feats'] = args['node_featurizer'].feat_size()
    config['in_edge_feats'] = args['edge_featurizer'].feat_size()
    return config

def mkdir_p(path):
    try:
        os.makedirs(path)
        print('Created directory %s'% path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory %s already exists.' % path)
        else:
            raise

def load_dataloader(args, enc_tokenizer):
    if args.mode == 'train':
        dataset = USPTODataset(args,
                               smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                               node_featurizer=args.node_featurizer,
                               edge_featurizer=args.edge_featurizer,
                               enc_tokenizer=enc_tokenizer)

        train_set = Subset(dataset, dataset.train_ids)
        val_set = Subset(dataset, dataset.val_ids)
        test_set = Subset(dataset, dataset.test_ids)
        collate_molgraphs = get_collate_molgraphs(enc_tokenizer.pad_token_id)
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_molgraphs, num_workers=args.num_workers)
        val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size,
                                collate_fn=collate_molgraphs, num_workers=args.num_workers)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                                 collate_fn=collate_molgraphs, num_workers=args.num_workers)
        return train_loader, val_loader, test_loader
    else:
        collate_molgraphs_test = get_collate_molgraphs_test(enc_tokenizer.pad_token_id)
        test_set = USPTOTestDataset(args,
                                    smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                    node_featurizer=args.node_featurizer,
                                    edge_featurizer=args.edge_featurizer,
                                    enc_tokenizer=enc_tokenizer)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                                 collate_fn=collate_molgraphs_test, num_workers=args.num_workers)
    return test_loader

def load_model(args):
    exp_config = get_configure(args)
    model = LocalText(
        node_in_feats=exp_config['in_node_feats'],
        edge_in_feats=exp_config['in_edge_feats'],
        node_out_feats=exp_config['node_out_feats'],
        edge_hidden_feats=exp_config['edge_hidden_feats'],
        num_step_message_passing=exp_config['num_step_message_passing'],
        attention_heads = exp_config['attention_heads'],
        attention_layers = exp_config['attention_layers'],
        AtomTemplate_n = exp_config['AtomTemplate_n'],
        BondTemplate_n = exp_config['BondTemplate_n'])
    model = model.to(args['device'])
    print ('Parameters of loaded LocalText:')
    print (exp_config)

    if args['mode'] == 'train':
        loss_criterion = nn.CrossEntropyLoss(reduction = 'none')
        optimizer = Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args['schedule_step'])
        
        if os.path.exists(args['model_path']):
            user_answer = input('%s exists, want to (a) overlap (b) continue from checkpoint (c) make a new model?' % args['model_path'])
            if user_answer == 'a':
                stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
                print ('Overlap exsited model and training a new model...')
            elif user_answer == 'b':
                stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
                stopper.load_checkpoint(model)
                print ('Train from exsited model checkpoint...')
            elif user_answer == 'c':
                model_name = input('Enter new model name: ')
                args['model_path'] = args['model_path'].replace('%s.pth' % args['dataset'], '%s.pth' % model_name)
                stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
                print ('Training a new model %s.pth' % model_name)
        else:
            stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
        return model, loss_criterion, optimizer, scheduler, stopper
    
    else:
        model.load_state_dict(torch.load(args['model_path'])['model_state_dict'])
        return model

def make_labels(graphs, labels, masks):
    atom_labels = []
    bond_labels = []
    for g, label, m in zip(graphs, labels, masks):
        n_atoms = g.number_of_nodes()
        n_bonds = g.number_of_edges()
        atom_label, bond_label = [0]*(n_atoms), [0]*(n_bonds-n_atoms)
        if m == 1:
            for l in label:
                label_type = l[0]
                label_idx = l[1]
                label_template = l[2]
                if label_type == 'a':
                    atom_label[label_idx] = label_template
                else:
                    bond_label[label_idx] = label_template
        atom_labels += atom_label
        bond_labels += bond_label   
    return torch.LongTensor(atom_labels), torch.LongTensor(bond_labels)

def flatten_list(t):
    return torch.LongTensor([item for sublist in t for item in sublist])


def get_collate_molgraphs(pad_token):
    def collate_molgraphs(data):
        inputs = [feat[4] for feat in data]
        smiles, graphs, labels, masks, _ = map(list, zip(*data))
        atom_labels, bond_labels = make_labels(graphs, labels, masks)
        bg = dgl.batch(graphs)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)

        batch_in = {
            'input_ids': pad_sequences([feat['input_ids'] for feat in inputs], pad_token),
            'attention_mask': pad_sequences([feat['attention_mask'] for feat in inputs], 0)
        }
        if 'position_ids' in inputs[0]:
            batch_in['position_ids'] = pad_sequences([feat['position_ids'] for feat in inputs], 0)

        return smiles, bg, atom_labels, bond_labels, batch_in
    return collate_molgraphs

def get_collate_molgraphs_test(pad_token):
    def collate_molgraphs_test(data):
        inputs = [feat[3] for feat in data]
        smiles, graphs, rxns, _ = map(list, zip(*data))
        bg = dgl.batch(graphs)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)

        batch_in = {
            'input_ids': pad_sequences([feat['input_ids'] for feat in inputs], pad_token),
            'attention_mask': pad_sequences([feat['attention_mask'] for feat in inputs], 0)
        }
        if 'position_ids' in inputs[0]:
            batch_in['position_ids'] = pad_sequences([feat['position_ids'] for feat in inputs], 0)

        return smiles, bg, rxns, batch_in 
    return collate_molgraphs_test

def predict(args, model, bg, input_ids, attention_mask):
    bg = bg.to(args.device)
    node_feats = bg.ndata.pop('h').to(args.device)
    edge_feats = bg.edata.pop('e').to(args.device)
    return model(bg, node_feats, edge_feats, input_ids, attention_mask)

def pad_sequences(sequences, pad_id, max_length=None, return_tensor=True):
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    output = []
    for seq in sequences:
        if len(seq) < max_length:
            output.append(seq + [pad_id] * (max_length - len(seq)))
        else:
            output.append(seq[:max_length])
    if return_tensor:
        return torch.tensor(output)
    else:
        return output
