import json
import logging
import numpy as np
import os
import pandas as pd
import time
from base.processor_base import Processor
from collections import Counter
from concurrent.futures import TimeoutError
from models.textcomposer_model.chemutils import cano_smiles, \
    cano_smarts, cano_smarts_, get_pattern_fingerprint_bitstr
from models.textcomposer_model.extract_templates import get_tpl, _Reactor
from models.textcomposer_model.prepare_mol_graph import MoleculeDataset
from multiprocessing import Pool
from pebble import ProcessPool
from rdkit import Chem
from tqdm import tqdm
from typing import Dict, List

from models.localtext_model.tokenizer import get_tokenizers

Reactor = _Reactor()
global G_cano_templates, G_smarts_mol_cache


def match_template(task):
    global G_cano_templates, G_smarts_mol_cache
    templates_train = G_cano_templates['templates_train']
    react_smarts_list = G_cano_templates['react_smarts_list']
    prod_smarts_list = G_cano_templates['prod_smarts_list']
    prod_smarts_fp_list = G_cano_templates['prod_smarts_fp_list']
    prod_smarts_fp_to_templates = G_cano_templates['prod_smarts_fp_to_templates']
    smarts_mol_cache = G_smarts_mol_cache

    idx, val = task
    reactant = cano_smiles(val['reactant'])
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol_prod = Chem.MolFromSmiles(val['product'], params)
    prod_fp_vec = int(get_pattern_fingerprint_bitstr(mol_prod), 2)

    sequences = []
    template_cands = []
    templates_list = []
    atom_indexes_fp_labels = {}
    # multiple templates may be valid for a reaction, find all of them
    for prod_smarts_fp_idx, prod_smarts_tmpls in prod_smarts_fp_to_templates.items():
        prod_smarts_fp_idx = int(prod_smarts_fp_idx)
        prod_smarts_fp = prod_smarts_fp_list[prod_smarts_fp_idx]
        for prod_smarts_idx, tmpls in prod_smarts_tmpls.items():
            # skip if fingerprint not match
            if (prod_smarts_fp & prod_fp_vec) < prod_smarts_fp:
                continue
            prod_smarts_idx = int(prod_smarts_idx)
            prod_smarts = prod_smarts_list[prod_smarts_idx]
            if prod_smarts not in smarts_mol_cache:
                smarts_mol_cache[prod_smarts] = Chem.MergeQueryHs(Chem.MolFromSmarts(prod_smarts))
            # we need also find matched atom indexes
            matches = mol_prod.GetSubstructMatches(smarts_mol_cache[prod_smarts])
            if len(matches):
                found_okay_tmpl = False
                for tmpl in tmpls:
                    pred_mols = Reactor.run_reaction(val['product'], tmpl)
                    if reactant and pred_mols and (reactant in pred_mols):
                        found_okay_tmpl = True
                        template_cands.append(templates_train.index(tmpl))
                        templates_list.append(tmpl)
                        reacts = tmpl.split('>>')[1].split('.')
                        if len(reacts) > 2:
                            logging.info(f'too many reacts: {reacts}, {idx}')
                        seq_reacts = [react_smarts_list.index(cano_smarts(r)) for r in reacts]
                        seq = [prod_smarts_fp_idx] + sorted(seq_reacts)
                        sequences.append(seq)
                # for each prod center, there may be multiple matches
                for match in matches:
                    match = tuple(sorted(match))
                    if match not in atom_indexes_fp_labels:
                        atom_indexes_fp_labels[match] = {}
                    if prod_smarts_fp_idx not in atom_indexes_fp_labels[match]:
                        atom_indexes_fp_labels[match][prod_smarts_fp_idx] = [[], []]
                    atom_indexes_fp_labels[match][prod_smarts_fp_idx][0].append(prod_smarts_idx)
                    atom_indexes_fp_labels[match][prod_smarts_fp_idx][1].append(found_okay_tmpl)

    reaction_center_cands = []
    reaction_center_cands_labels = []
    reaction_center_cands_smarts = []
    reaction_center_atom_indexes = []
    for atom_index in sorted(atom_indexes_fp_labels.keys()):
        for fp_idx, val in atom_indexes_fp_labels[atom_index].items():
            reaction_center_cands.append(fp_idx)
            reaction_center_cands_smarts.append(val[0])
            reaction_center_cands_labels.append(True in val[1])
            reaction_center_atom_indexes.append(atom_index)

    tmpl_res = {
        'templates': templates_list,
        'template_cands': template_cands,
        'template_sequences': sequences,
        'reaction_center_cands': reaction_center_cands,
        'reaction_center_cands_labels': reaction_center_cands_labels,
        'reaction_center_cands_smarts': reaction_center_cands_smarts,
        'reaction_center_atom_indexes': reaction_center_atom_indexes,
    }

    return idx, tmpl_res


class TextComposerProcessorS1(Processor):
    """Class for TextComposer Preprocessing, Stage 1"""

    def __init__(self,
                 model_name: str,
                 model_args,
                 model_config: Dict[str, any],
                 data_name: str,
                 raw_data_files: List[str],
                 processed_data_path: str,
                 num_cores: int = None):
        super().__init__(model_name=model_name,
                         model_args=model_args,
                         model_config=model_config,
                         data_name=data_name,
                         raw_data_files=raw_data_files,
                         processed_data_path=processed_data_path)
        self.train_file, self.val_file, self.test_file = raw_data_files
        self.num_cores = num_cores

        self.model_args = model_args

        self.enc_tokenizer = get_tokenizers(self.model_args)

    def preprocess(self) -> None:
        """Actual file-based preprocessing"""
        self.extract_templates()
        self.find_cano_templates()
        self.match_templates()
        self.prepare_mol_graph()

    def extract_templates(self):
        """Adapted from extract_templates.py"""
        logging.info(f"Extracting templates for {self.data_name}")

        for phase, csv_file in [("train", self.train_file),
                                ("val", self.val_file),
                                ("test", self.test_file)]:
            output_file = os.path.join(self.processed_data_path, f"templates_{phase}.json")
            if os.path.exists(output_file):
                logging.info(f"Output file found at {output_file}, skipping template extraction for phase {phase}")
                continue

            start = time.time()
            df = pd.read_csv(csv_file)

            reaction_list = df["rxn_smiles"]
            ids = df["id"].tolist()
            classes = df["class"].tolist()

            nn_ids = df["nn_id"].tolist()

            reactant_list = []
            product_list = []
            rxns = []
            for idx, reaction in enumerate(tqdm(reaction_list)):
                r, _, p = reaction.strip().split(">")
                reactant_list.append(r)
                product_list.append(p)
                rxns.append((idx, r, p, ids[idx], classes[idx], nn_ids[idx]))
            logging.info(f"Total reactions: {len(rxns)}")

            cnt = 0
            templates = {}
            with ProcessPool(max_workers=self.num_cores) as pool:
                # had to resort to pebble to add timeout. rdchiral could hang
                future = pool.map(get_tpl, rxns, timeout=10)

                iterator = future.result()
                while True:
                    try:
                        idx, react, prod, reaction_smarts, cano_reacts, retro_okay, id, cls, nn_id = next(iterator)
                        cnt += 'exact_match' in retro_okay or 'equal_mol' in retro_okay
                        templates[idx] = {
                            'id': id,
                            'class': cls,
                            'reactant': react,
                            'product': prod,
                            'reaction_smarts': reaction_smarts,
                            'cano_reactants': cano_reacts,
                            'nn_id': nn_id,
                        }
                    except StopIteration:
                        break
                    except TimeoutError as error:
                        logging.info(f"get_tpl call took more than {error.args} seconds")

            logging.info(f'retro_okay cnt: {cnt}, {len(rxns)}, {cnt / len(rxns)}, time: {time.time() - start} s')
            with open(output_file, 'w') as of:
                json.dump(templates, of, indent=4)

    def find_cano_templates(self):
        """Adapted from extract_templates.py"""
        logging.info("Finding canonical templates")

        prod_k = self.model_args.prod_k
        react_k = self.model_args.react_k

        output_file = os.path.join(self.processed_data_path, f"templates_cano_train.json")
        if os.path.exists(output_file):
            logging.info(f"Output file found at {output_file}, skipping template canonicalization")
            return

        train_template_file = os.path.join(self.processed_data_path, "templates_train.json")
        with open(train_template_file, "r") as f:
            templates_data_train = json.load(f)

        smarts_fp_cache = {}
        fp_prod_smarts_dict = {}
        cano_react_smarts_dict = Counter()
        for idx, template in tqdm(templates_data_train.items()):
            items = template['reaction_smarts'].split('>>')
            if len(items) == 2 and items[0] and items[1]:
                prod_smarts, reacts_smarts = items
                prod_smarts = cano_smarts_(prod_smarts)
                if prod_smarts not in smarts_fp_cache:
                    mol = Chem.MergeQueryHs(Chem.MolFromSmarts(prod_smarts))
                    smarts_fp_cache[prod_smarts] = int(get_pattern_fingerprint_bitstr(mol), 2)
                if smarts_fp_cache[prod_smarts] not in fp_prod_smarts_dict:
                    fp_prod_smarts_dict[smarts_fp_cache[prod_smarts]] = {'cnt': 0, 'cano_smarts': []}
                fp_prod_smarts_dict[smarts_fp_cache[prod_smarts]]['cnt'] += 1
                fp_prod_smarts_dict[smarts_fp_cache[prod_smarts]]['cano_smarts'].append(prod_smarts)

                cano_reacts_smarts = cano_smarts(reacts_smarts)
                cano_react_smarts_dict.update(cano_reacts_smarts.split('.'))
                template['cano_reaction_smarts'] = prod_smarts + '>>' + cano_reacts_smarts
            else:
                logging.info('invalid reaction_smarts:', idx, items)

        logging.info(f'fp_prod_smarts_dict size: {len(fp_prod_smarts_dict)}, '
                     f'cano_react_smarts_dict size: {len(cano_react_smarts_dict)}')
        logging.info(f'smarts filter threshold: {prod_k}, {react_k}')

        prod_smarts_list = set()
        prod_smarts_fp_to_remove = []
        # filter product smarts less than the frequency
        for fp, val in fp_prod_smarts_dict.items():
            if val['cnt'] < prod_k:
                prod_smarts_fp_to_remove.append(fp)
            else:
                fp_prod_smarts_dict[fp] = list(set(val['cano_smarts']))
                prod_smarts_list.update(fp_prod_smarts_dict[fp])

        [fp_prod_smarts_dict.pop(fp) for fp in prod_smarts_fp_to_remove]
        prod_smarts_fp_list = sorted(fp_prod_smarts_dict.keys())
        prod_smarts_list = sorted(prod_smarts_list)
        # find smarts indexes
        for fp, val in fp_prod_smarts_dict.items():
            fp_prod_smarts_dict[fp] = [prod_smarts_list.index(v) for v in val]

        cano_react_smarts_dict = dict(filter(lambda elem: elem[1] >= react_k, cano_react_smarts_dict.items()))
        # sort reactants by frequency from high to low
        react_smarts_list = [k for k, v in
                             sorted(cano_react_smarts_dict.items(), key=lambda item: item[1], reverse=True)]

        logging.info(f'after filtering, prod_smarts_fp_list size: {len(prod_smarts_fp_list)}, '
                     f'cano_react_smarts_list size: {len(cano_react_smarts_dict)}')

        prod_smarts_fp_to_templates = {}
        for idx, template in tqdm(templates_data_train.items()):
            if 'cano_reaction_smarts' in template:
                cano_prod_smarts, cano_reacts_smarts = template['cano_reaction_smarts'].split('>>')
                if cano_prod_smarts not in smarts_fp_cache:
                    mol = Chem.MergeQueryHs(Chem.MolFromSmarts(cano_prod_smarts))
                    smarts_fp_cache[cano_prod_smarts] = int(get_pattern_fingerprint_bitstr(mol), 2)
                if smarts_fp_cache[cano_prod_smarts] not in fp_prod_smarts_dict:
                    logging.info(f'skip cano_prod_smarts: {idx}, {cano_prod_smarts}')
                    continue

                cano_reacts_smarts = set(cano_reacts_smarts.split('.'))
                if not cano_reacts_smarts.issubset(cano_react_smarts_dict):
                    logging.info(f'skip cano_reacts_smarts: {idx}, {cano_reacts_smarts}')
                    continue

                cano_prod_smarts_fp_idx = prod_smarts_fp_list.index(smarts_fp_cache[cano_prod_smarts])
                if cano_prod_smarts_fp_idx not in prod_smarts_fp_to_templates:
                    prod_smarts_fp_to_templates[cano_prod_smarts_fp_idx] = {}
                cano_prod_smarts_idx = prod_smarts_list.index(cano_prod_smarts)
                if cano_prod_smarts_idx not in prod_smarts_fp_to_templates[cano_prod_smarts_fp_idx]:
                    prod_smarts_fp_to_templates[cano_prod_smarts_fp_idx][cano_prod_smarts_idx] = set()
                prod_smarts_fp_to_templates[cano_prod_smarts_fp_idx][cano_prod_smarts_idx].add(
                    template['reaction_smarts'])

        tmpl_lens = []
        templates_train = set()
        for fp, val in prod_smarts_fp_to_templates.items():
            for cano_prod_smarts, tmpls in val.items():
                tmpl_lens.append(len(tmpls))
                templates_train.update(tmpls)
                prod_smarts_fp_to_templates[fp][cano_prod_smarts] = list(tmpls)
        logging.info(f'#average template variants per cano_prod_smarts: {np.mean(tmpl_lens)}')

        logging.info(f'templates_data_train: {len(templates_data_train)}')
        templates_train = sorted(list(templates_train))
        data = {
            'templates_train': templates_train,
            'react_smarts_list': react_smarts_list,
            'prod_smarts_list': prod_smarts_list,
            'prod_smarts_fp_list': prod_smarts_fp_list,
            'fp_prod_smarts_dict': fp_prod_smarts_dict,
            'prod_smarts_fp_to_templates': prod_smarts_fp_to_templates,
        }
        for key, val in data.items():
            logging.info(f"{key}: {len(val)}")
        with open(output_file, 'w') as of:
            json.dump(data, of, indent=4)

    def match_templates(self):
        """Adapted from extract_templates.py"""
        logging.info(f"Matching templates for {self.data_name}")
        global G_cano_templates, G_smarts_mol_cache

        cano_templates_file = os.path.join(self.processed_data_path, f"templates_cano_train.json")
        with open(cano_templates_file, "r") as f:
            G_cano_templates = json.load(f)
        G_smarts_mol_cache = {}

        # find all applicable templates for each reaction
        # since multiple templates may be valid for a reaction
        p = Pool(self.num_cores)
        for phase in ["train", "val", "test"]:
            template_file = os.path.join(self.processed_data_path, f"templates_{phase}.json")
            output_file = os.path.join(self.processed_data_path, f"templates_{phase}_new.json")
            if os.path.exists(output_file):
                logging.info(f"Output file found at {output_file}, skipping template matching for phase {phase}")
                continue

            logging.info(f'find all applicable templates for each reaction in {template_file}')
            with open(template_file, "r") as f:
                templates_data = json.load(f)
            cnt = 0
            for res in tqdm(p.imap_unordered(match_template, templates_data.items()), total=len(templates_data)):
                idx, tmpl_res = res
                templates_data[idx].update(tmpl_res)
                cnt += len(tmpl_res['templates']) > 0
            logging.info(f'template coverage: {cnt / len(templates_data)} for {self.data_name} {phase}')
            with open(output_file, 'w') as of:
                json.dump(templates_data, of, indent=4)

        p.close()
        p.join()

    def prepare_mol_graph(self):
        """Adapted from prepare_mol_graph.py"""
        logging.info('prepare retrosynthesis data')
        for phase in ['train', 'val', 'test']:
            dataset_valid = MoleculeDataset(args = self.model_args, root = self.processed_data_path, enc_tokenizer = self.enc_tokenizer, split = phase, load_mol=True)
            dataset_valid.process_data()
