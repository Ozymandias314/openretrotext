import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch
#from torch._six import container_abcs, string_classes, int_classes


class CustomDataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """
    def __init__(self, dataset, pad_token_id = 0, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        def collate(batch):
            elem = batch[0]
            inputs = [b.encoder_input for b in batch]
            batch_in = {
            'input_ids': self.pad_encoder_sequences([feat['input_ids'] for feat in inputs], pad_token_id),
            'attention_mask': self.pad_encoder_sequences([feat['attention_mask'] for feat in inputs], 0)
        }
            if 'position_ids' in inputs[0]:
                batch_in['position_ids'] = self.pad_encoder_sequences([feat['position_ids'] for feat in inputs], 0)
            if isinstance(elem, Data):
                return Batch.from_data_list(batch, follow_batch), batch_in
           

            raise TypeError('DataLoader found invalid type: {}'.format(
                type(elem)))

        super(CustomDataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=lambda batch: collate(batch), **kwargs)
    
    def pad_encoder_sequences(self, sequences, pad_id, max_length=None, return_tensor=True):
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