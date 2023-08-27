import logging
import os
import pandas as pd
from rdkit import Chem
from dgl.data.utils import save_graphs, load_graphs

import pickle
import random


import json

def canonicalize_rxn(rxn):
    canonicalized_smiles = []
    r, p = rxn.split('>>')
    for s in [r, p]:
        mol = Chem.MolFromSmiles(s)
        [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
        canonicalized_smiles.append(Chem.MolToSmiles(mol))
    return '>>'.join(canonicalized_smiles)

class USPTODataset:
    def __init__(self, args, smiles_to_graph, node_featurizer, edge_featurizer, enc_tokenizer, load=True, log_every=1000):
        fn = os.path.join(args.processed_data_path, "labeled_data.csv")
        df = pd.read_csv(fn)
        self.train_ids = df.index[df['Split'] == 'train'].values
        self.val_ids = df.index[df['Split'] == 'val'].values
        self.test_ids = df.index[df['Split'] == 'test'].values
        self.smiles = df['Products'].tolist()
        self.masks = df['Mask'].tolist()
        self.labels = [eval(t) for t in df['Labels']]
        self.indices = df['nn_id'].tolist()
        self.enc_tokenizer = enc_tokenizer
        self.args = args
        self.corpus = read_corpus(args.corpus_file, args.cache_path)
        self.neighbors = {}
        phases = ['train', 'valid', 'test']


        base_dir = args.nn_path 

        for phase in phases:

            json_file_path = os.path.join(base_dir, f'{phase}_rank.json')


            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)

                for ex in data:
                    self.neighbors[ex['id']] = ex['nn']

        self.product_smile = df['product_smiles']
        self.cache_file_path = os.path.join(args.processed_data_path, "dglgraph.bin")
        self._pre_process(smiles_to_graph, node_featurizer, edge_featurizer, load, log_every)

    def _pre_process(self, smiles_to_graph, node_featurizer, edge_featurizer, load, log_every):
        if os.path.exists(self.cache_file_path) and load:
            logging.info('Loading previously saved dgl graphs...')
            self.graphs, label_dict = load_graphs(self.cache_file_path)
        else:
            logging.info('Processing dgl graphs from scratch...')
            self.graphs = []
            for i, s in enumerate(self.smiles):
                if (i + 1) % log_every == 0:
                    logging.info(f'\rProcessing molecule {i + 1}/{len(self.smiles)}')
                self.graphs.append(smiles_to_graph(s, node_featurizer=node_featurizer,
                                                   edge_featurizer=edge_featurizer, canonical_atom_order=False))
            save_graphs(self.cache_file_path, self.graphs)

    def __getitem__(self, item):
        enc_input = self.prepare_encoder_input(item)
        enc_input = {k: v[:self.args.max_length] for k, v in enc_input.items()}
        return self.smiles[item], self.graphs[item], self.labels[item], self.masks[item], enc_input

    def __len__(self):
        return len(self.smiles)

    def load_corpus(self, nn_file):

        with open(nn_file) as f:
            nn_data = json.load(f)
            self.neighbors = {ex['id']: ex['nn'] for ex in nn_data}

    def deduplicate_neighbors(self, neighbors_ids):
        output = []
        for i in neighbors_ids:
            flag = False
            for j in output:
                if self.corpus[i] == self.corpus[j]:
                    flag = True
                    break
            if not flag:
                output.append(i)
        return output

    def get_neighbor_text(self, idx, return_list=False):
        rxn_id = self.indices[idx]
        neighbors_ids = [i for i in self.neighbors[rxn_id] if i in self.corpus]
        if idx in self.train_ids:
            if self.args.use_gold_neighbor:
                if rxn_id in neighbors_ids:
                    neighbors_ids.remove(rxn_id)
                if rxn_id in self.corpus:
                    neighbors_ids = [rxn_id] + neighbors_ids
            neighbors_ids = self.deduplicate_neighbors(neighbors_ids)
            neighbors_texts = [self.corpus[i] for i in neighbors_ids[:self.args.max_num_neighbors]]
            if random.random() < self.args.random_neighbor_ratio:
                selected_texts = random.sample(neighbors_texts, k=min(self.args.num_neighbors, len(neighbors_texts)))
            else:
                selected_texts = neighbors_texts[:self.args.num_neighbors]
        else:

            neighbors_ids = self.deduplicate_neighbors(neighbors_ids)
            selected_texts = [self.corpus[i] for i in neighbors_ids[:self.args.num_neighbors]]
        return selected_texts if return_list \
            else ''.join([f' ({i}) {text}' for i, text in enumerate(selected_texts)])
    
    def prepare_encoder_input(self, idx):
 
        product_smiles = self.smiles[idx]
        # Data augmentation

        nn_text = self.get_neighbor_text(idx) if self.args.num_neighbors > 0 else None

        enc_input = self.enc_tokenizer(
            product_smiles, text_pair=nn_text, truncation=False, return_token_type_ids=False, verbose=False)
        return enc_input

        
class USPTOTestDataset:
    def __init__(self, args, smiles_to_graph, node_featurizer, edge_featurizer, enc_tokenizer, load=True, log_every=1000):
        df = pd.read_csv(args.test_file)
        self.rxns = df['rxn_smiles'].tolist()
        self.rxns = [canonicalize_rxn(rxn) for rxn in self.rxns]
        self.smiles = [rxn.split('>>')[-1] for rxn in self.rxns]
        self.cache_file_path = os.path.join(args.processed_data_path, "test_dglgraph.bin")
        self._pre_process(smiles_to_graph, node_featurizer, edge_featurizer, load, log_every)
        self.indices = df['nn_id'].tolist()
        self.enc_tokenizer = enc_tokenizer
        self.args = args
        self.corpus = read_corpus(args.corpus_file, args.cache_path)
        self.neighbors = {}
        phases = ['test']


        base_dir = args.nn_path 

        for phase in phases:

            json_file_path = os.path.join(base_dir, f'{phase}_rank.json')


            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)

                for ex in data:
                    self.neighbors[ex['id']] = ex['nn']

    def _pre_process(self, smiles_to_graph, node_featurizer, edge_featurizer, load, log_every):
        if os.path.exists(self.cache_file_path) and load:
            logging.info('Loading previously saved test dgl graphs...')
            self.graphs, label_dict = load_graphs(self.cache_file_path)
        else:
            logging.info('Processing test dgl graphs from scratch...')
            self.graphs = []
            for i, s in enumerate(self.smiles):
                if (i + 1) % log_every == 0:
                    logging.info(f'Processing molecule {i + 1}/{len(self.smiles)}')
                self.graphs.append(smiles_to_graph(s, node_featurizer=node_featurizer,
                                                   edge_featurizer=edge_featurizer, canonical_atom_order=False))
            save_graphs(self.cache_file_path, self.graphs)

    def __getitem__(self, item):
        enc_input = self.prepare_encoder_input(item)
        enc_input = {k: v[:self.args.max_length] for k, v in enc_input.items()}
        return self.smiles[item], self.graphs[item], self.rxns[item], enc_input

    def __len__(self):
        return len(self.smiles)
    def load_corpus(self, nn_file):

        with open(nn_file) as f:
            nn_data = json.load(f)
            self.neighbors = {ex['id']: ex['nn'] for ex in nn_data}

    def deduplicate_neighbors(self, neighbors_ids):
        output = []
        for i in neighbors_ids:
            flag = False
            for j in output:
                if self.corpus[i] == self.corpus[j]:
                    flag = True
                    break
            if not flag:
                output.append(i)
        return output

    def get_neighbor_text(self, idx, return_list=False):
        rxn_id = self.indices[idx]
        neighbors_ids = [i for i in self.neighbors[rxn_id] if i in self.corpus]


        neighbors_ids = self.deduplicate_neighbors(neighbors_ids)
        selected_texts = [self.corpus[i] for i in neighbors_ids[:self.args.num_neighbors]]
        return selected_texts if return_list \
            else ''.join([f' ({i}) {text}' for i, text in enumerate(selected_texts)])
    
    def prepare_encoder_input(self, idx):

        product_smiles = self.smiles[idx]
        # Data augmentation

        nn_text = self.get_neighbor_text(idx) if self.args.num_neighbors > 0 else None

        enc_input = self.enc_tokenizer(
            product_smiles, text_pair=nn_text, truncation=False, return_token_type_ids=False, verbose=False)
        return enc_input


def read_corpus(corpus_file, cache_path=None):
    if cache_path:
        cache_file = os.path.join(cache_path, corpus_file.replace('.csv', '.pkl'))
        if os.path.exists(cache_file):
            logging.info(f' Load corpus from: {cache_file}')
            with open(cache_file, 'rb') as f:
                corpus = pickle.load(f)
            return corpus
    corpus_df = pd.read_csv(corpus_file, keep_default_na=False)
    corpus = {}
    for i, row in corpus_df.iterrows():
        if len(row['heading_text']) > 0:
            corpus[row['id']] = row['heading_text'] + '. ' + row['paragraph_text']
        else:
            corpus[row['id']] = row['paragraph_text']
    if cache_path:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        logging.info(f' Save corpus to: {cache_file}')
        with open(cache_file, 'wb') as f:
            pickle.dump(corpus, f)
    return corpus


def generate_train_label_corpus(train_file):
    """Use rxn smiles and labels in the training data as the corpus, instead of text."""
    train_df = pd.read_csv(train_file, keep_default_na=False)
    corpus = {}
    for i, row in train_df.iterrows():
        condition = ''
        for col in CONDITION_COLS:
            if len(row[col]) > 0:
                if condition == '':
                    condition += row[col]
                else:
                    condition += '.' + row[col]
        rxn_smiles = row['canonical_rxn']
        corpus[row['id']] = rxn_smiles.replace('>>', f'>{condition}>')
    return corpus


def random_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        new_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        return new_smiles
    except:
        return smiles


def random_shuffle_reaction_smiles(rxn_smiles, p=0.8):
    if random.random() > p:
        return rxn_smiles
    reactant_str, product_str = rxn_smiles.split('>>')
    reactants = [random_smiles(smiles) for smiles in reactant_str.split('.')]
    products = [random_smiles(smiles) for smiles in product_str.split('.')]
    random.shuffle(reactants)
    random.shuffle(products)
    return '.'.join(reactants) + '>>' + '.'.join(products)