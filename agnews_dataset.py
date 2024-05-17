import pandas as pd
import numpy as np
import os
from fedlab.contrib.dataset.basic_dataset import FedDataset, Subset, BaseDataset
import torch
from torch.utils.data import DataLoader
from datasets import Dataset

class PartitionedAGNews():
    def __init__(self, root, path, num_clients) -> None:
        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients

    def preprocess(self, dir_alpha=0.5):
       ...

    def get_dataset(self, id, type="train"):
        """Load subdataset for client with client ID ``cid`` from local file.

        Args:
            cid (int): client id
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.

        Returns:
            Dataset
        """
        dataset = torch.load(os.path.join(self.path, type, "data{}.pkl".format(id)))
        return dataset

    def get_dataloader(self, id, batch_size=None, type="train"):
        """Return dataload for client with client ID ``cid``.

        Args:
            cid (int): client id
            batch_size (int, optional): batch size in DataLoader.
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
        """
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size

        def iteration(dataset, batch_size, idx=0):
            yield dataset[idx:idx+batch_size]
            idx += batch_size

        return iteration(dataset, batch_size)


def AGNews_TestDataset(tokenizer, batch_size=256):
    df = pd.read_csv("datasets/agnews/test.csv")
    df = df.rename(columns={oc:nc for oc, nc in zip(list(df.columns), ['label','Title', 'Description'])})
    df['text']=(df['Title']+df['Description'])
    df.drop(columns=['Title','Description'],axis=1,inplace=True)
    df['label'] -= 1
    
    def pipeline(dataframe):
        """
        Prepares the dataframe so that it can be given to the transformer model
        
        input -> pandas dataframe
        output -> tokenized dataset (columns = text, label, input, attention)
        """    
        def preprocess_function(examples):
            """
            Tokenizes the given text

            input -> dataset (columns = text, label)
            output -> tokenized dataset (columns = text, label, input, attention)
            """
            return tokenizer(examples["text"], truncation=True, padding=True, max_length=100)
        # This step isn't mentioned anywhere but is vital as Transformers library only seems to work with this Dataset data type
        dataset = Dataset.from_pandas(dataframe, preserve_index=False)
        tokenized_ds = dataset.map(preprocess_function, batched=True)
        tokenized_ds = tokenized_ds.remove_columns('text')
        return tokenized_ds

    tokenized_dataset = pipeline(df)
    torch.save(tokenized_dataset,os.path.join("datasets/partitioned_agnews", "test", "data.pkl"))

    def iteration(dataset, batch_size, idx=0):
        yield dataset[idx:idx+batch_size]
        idx += batch_size
    return iteration(tokenized_dataset, batch_size=batch_size)

def get_AGNEWs_testloader(batch_size=1024):
    dataset = torch.load(os.path.join("datasets/partitioned_agnews", "test", "data.pkl"))
    def iteration(dataset, batch_size, idx=0):
        yield dataset[idx:idx+batch_size]
        idx += batch_size
    return iteration(dataset, batch_size=batch_size)

if __name__ == "__main__":
    dataset = PartitionedAGNews(root="datasets", path="datasets/partitioned_agnews", num_clients=100)
    dataset.preprocess()
    test_loader = AGNews_TestDataset()

    # data partition and preprocessing

    from transformers import AutoTokenizer, DataCollatorWithPadding
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
    from datasets import Dataset
    import os
    import pandas as pd
    import torch

    # load tokenizer from bert base uncased model available from huggingface.co
    # tokenizer=AutoTokenizer.from_pretrained("/data/distilbert")
    tokenizer=AutoTokenizer.from_pretrained("/data/pythia-70m")
    df = pd.read_csv("datasets/agnews/train.csv")
    df = df.rename(columns={oc:nc for oc, nc in zip(list(df.columns), ['label','Title', 'Description'])})
    df['text']=(df['Title']+df['Description'])
    df.drop(columns=['Title','Description'],axis=1,inplace=True)
    df['label'] -= 1

    def preprocess_function(examples):
        """
        Tokenizes the given text

        input -> dataset (columns = text, label)
        output -> tokenized dataset (columns = text, label, input, attention)
        """
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=100)

    def pipeline(dataframe):
        """
        Prepares the dataframe so that it can be given to the transformer model
        
        input -> pandas dataframe
        output -> tokenized dataset (columns = text, label, input, attention)
        """    
        # This step isn't mentioned anywhere but is vital as Transformers library only seems to work with this Dataset data type
        dataset = Dataset.from_pandas(dataframe, preserve_index=False)
        tokenized_ds = dataset.map(preprocess_function, batched=True)
        tokenized_ds = tokenized_ds.remove_columns('text')
        return tokenized_ds

    import numpy as np

    def hetero_dir_partition(targets, num_clients, num_classes, dir_alpha, min_require_size=None):
        if min_require_size is None:
            min_require_size = num_classes

        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)
        num_samples = targets.shape[0]

        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            # for each class in the dataset
            for k in range(num_classes):
                idx_k = np.where(targets == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(
                    np.repeat(dir_alpha, num_clients))
                # Balance
                proportions = np.array(
                    [p * (len(idx_j) < num_samples / num_clients) for p, idx_j in
                    zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                            zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        client_dict = dict()
        for cid in range(num_clients):
            np.random.shuffle(idx_batch[cid])
            client_dict[cid] = np.array(idx_batch[cid])

        return client_dict

    labels = df['label'].to_list()
    partition = hetero_dir_partition(labels, 100, 4, dir_alpha=0.1)

    datasets = []
    for key, value in partition.items():
        data = df.iloc[value]
        tokenized_dataset = pipeline(data)
        torch.save(tokenized_dataset,os.path.join("datasets/partitioned_agnews", "train", "data{}.pkl".format(key)))