import logging
import os
import pandas as pd
from rdkit import Chem
from dgl.data.utils import save_graphs, load_graphs

def canonicalize_rxn(rxn):
    canonicalized_smiles = []
    r, p = rxn.split('>>')
    for s in [r, p]:
        mol = Chem.MolFromSmiles(s)
        [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
        canonicalized_smiles.append(Chem.MolToSmiles(mol))
    return '>>'.join(canonicalized_smiles)

class USPTODataset:
    def __init__(self, args, smiles_to_graph, node_featurizer, edge_featurizer, load=True, log_every=1000):
        fn = os.path.join(args.processed_data_path, "labeled_data.csv")
        df = pd.read_csv(fn)
        self.train_ids = df.index[df['Split'] == 'train'].values
        self.val_ids = df.index[df['Split'] == 'val'].values
        self.test_ids = df.index[df['Split'] == 'test'].values
        self.smiles = df['Products'].tolist()
        self.masks = df['Mask'].tolist()
        self.labels = [eval(t) for t in df['Labels']]
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
        return self.smiles[item], self.graphs[item], self.labels[item], self.masks[item]

    def __len__(self):
        return len(self.smiles)
        
class USPTOTestDataset:
    def __init__(self, args, smiles_to_graph, node_featurizer, edge_featurizer, load=True, log_every=1000):
        df = pd.read_csv(args.test_file)
        self.rxns = df['rxn_smiles'].tolist()
        self.rxns = [canonicalize_rxn(rxn) for rxn in self.rxns]
        self.smiles = [rxn.split('>>')[-1] for rxn in self.rxns]
        self.cache_file_path = os.path.join(args.processed_data_path, "test_dglgraph.bin")
        self._pre_process(smiles_to_graph, node_featurizer, edge_featurizer, load, log_every)

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
        return self.smiles[item], self.graphs[item], self.rxns[item]

    def __len__(self):
        return len(self.smiles)
