import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager
from utils.utils import load_dataset
import os
import torch
import datetime

class DatasetBase():
    def __init__(self, coord_dim, num_samples, num_nodes, annotation, parallel, random_seed, num_cpus):
        self.coord_dim = coord_dim
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.annotation = annotation
        self.parallel = parallel
        self.num_cpus = num_cpus
        self.seed = random_seed

    def generate_instance(self, seed):
        raise NotImplementedError
    
    def generate_dataset(self):
        dataset = []
        num_required_samples = self.num_samples
        seed = self.seed
        end = False
        print("Data generation started.", flush=True)
        while(not end):
            seeds = seed + np.arange(num_required_samples)
            instances = [
                self.generate_instance(seed=s)
                for s in tqdm(seeds, desc="Generating instances")
            ]
            if self.annotation:
                if self.parallel:
                    instances = self.generate_labeldata_para(instances, self.num_cpus)
                else:
                    instances = self.generate_labeldata(instances)

            dataset.extend(filter(None, instances))
            
            seed += num_required_samples
            num_required_samples = self.num_samples - len(dataset) 
            if len(dataset) == self.num_samples:
                end = True
            else:
                print(f"No feasible tour was not found in {num_required_samples} instances. Trying other {num_required_samples} instances.", flush=True)
        print("Data generation completed.", flush=True)
        return dataset
    
    def annotate(self, instance):
        raise NotImplementedError
    
    def generate_labeldata(self, dataset):
        """
        Parameters
        ----------
        dataset_path: str
            path to the tsptw dataset
        
        Returns
        -------
        dataset: 
        """
        return [self.annotate(instance) for instance in tqdm(dataset, desc="Annotating instances")]

    def generate_labeldata_para(self, dataset, num_cpus):
        with Pool(num_cpus) as pool:
            annotation_data = list(tqdm(pool.imap(self.annotate, [instance for instance in dataset]), total=len(dataset), desc="Annotating instances"))
        return annotation_data

import multiprocessing
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

class DataLoaderBase(torch.utils.data.Dataset):
    def __init__(self, fpath, sequential=False, parallel=False, num_cpus=1):
        now = datetime.datetime.now()
        dir_name = f"test/data_load_{now.strftime('%Y%m%d_%H%M%S%f')}"
        os.makedirs(dir_name)
        annotation_data = load_dataset(fpath)
        load = self.load_sequentially if sequential else self.load_randomly
        if parallel:
            data = []
            chunk_size = 1000
            num_process = multiprocessing.cpu_count()
            pool = torch.multiprocessing.Pool(num_process)
            for i in tqdm(range(0, len(annotation_data), chunk_size)):
                chunk_data = annotation_data[i:i+chunk_size]
                for fname in pool.starmap(load, [(instance, f"{dir_name}/chunk{i}_{j}.pkl") for j, instance in enumerate(chunk_data)]):
                    data.extend(load_dataset(fname))
                    os.remove(fname)
            pool.close()
            self.data = data
        else:
            self.data = [elem for instance in tqdm(annotation_data) for elem in load(instance)]
        self.size = len(self.data)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def load_sequentially(self, instance, fname=None):
        NotImplementedError
    
    def load_randomly(self, instance, fname=None):
        NotImplementedError