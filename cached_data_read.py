#!/usr/bin/env python

import numpy as np
import torch
import torch.utils.data
import h5py
import glob2
import random
import atexit
from time import time, strftime, localtime
from datetime import timedelta

import timing


torch.manual_seed(0)

EMBED_DIMS = 5000
NUM_CLASSES = 10
SEQ_LENGTH = 768
SUBSETS = 15
SAMPLES_SET = 10000
#
GENERATE_DATA = False
CACHE_SIZE = 1000

MAX_OPEN_FILES = 3

NUM_SAMPLES = SAMPLES_SET*SUBSETS

#@profile
def generate(data_dir):
    sequences = np.random.randint(EMBED_DIMS, size = (NUM_SAMPLES, SEQ_LENGTH))
    labels = np.random.randint(NUM_CLASSES, size = (NUM_SAMPLES,1))
    #writes labels numpy array as a dataset called 'labels' in labels.h5 file
    with h5py.File(data_dir + 'labels.h5', mode='w', swmr=True) as hf:
        print("labels shape: ", labels.shape)
        hf.create_dataset('targets', data=labels)
    # writes subsets of sequences numpy array as 'samples' dataset in data_*.h5 files - one file for each subset
    for st in range(SUBSETS):
        seq_set = sequences[st*SAMPLES_SET:(st+1)*SAMPLES_SET, :]
        file_subset = data_dir + 'data_'+str(st).zfill(3)+'.h5'
        print("Writing to {file_subset} sequences shape {seq_set.shape} ")
        with h5py.File(file_subset, mode='w', swmr=True) as hf:
            hf.create_dataset('samples', data=seq_set)
    return
#@profile
# This function will work in a multiprocessing pool framework

def read(data_dir):
    labels_read = None
    with h5py.File(data_dir + 'labels.h5', mode='r', swmr=True) as hf:
        labels_read = hf.get('targets')[()]
        print("labels shape: ", labels_read.shape)

    filenames = glob2.glob(data_dir+"data_*.h5")
    if len(filenames) != SUBSETS:
        raise Warning("Training partitions don't match the training partition files")

    seq_read = np.zeros((SAMPLES_SET, SEQ_LENGTH))
    seq = [seq_read] * SUBSETS

    for f in range(len(filenames)):
        filename = filenames[f]
        with h5py.File(filename, mode='r', swmr=True) as hf:
            hf['samples'].read_direct(seq[f])
            print(f"seq_read_direct {filename} shape {seq[f].shape}: ")
    return

class DatasetWithCaching(torch.utils.data.Dataset):
    ''' This dataset caches data by obtaining slices of h5 files.
    The schedule of sample ids is generated in an instance of the RandomScheduler class.
    If the dataset is to be used with a dataloader, the dataloader should be initialised with sample set to a instance of the RandomScheduler class
    This schedule allows the caching to know which samples to cache next.
    cache_size is the number of sample to be cached at a time.

    '''

    def __init__(self, data_dir, sample_ids, all_labels, cache_size=0):
        '''

        :param data_dir: string, location of data directory
        :param sample_ids: list of sample ids in dataset
        :param all_labels: list of labels
        :param cache_size: int, Number of samples to retrieve when filling the cache
        '''
        self.sample_ids = sample_ids
        self.data_dir = data_dir
        self.all_labels = all_labels
        self.num_samples = len(sample_ids)
        self.data_seq = None
        self.scheduled_indices = []
        self.cached_data =None
        self.num_cached_samples = 0
        self.cache_size = cache_size

    def __getitem__(self, index):
        '''
        This is a modified get_item that reads samples from a cache and refill the cache if empty
        :param index:
        :return: data and label for index sample
        '''
        sample_id, file_sample_id, file_id = self.file_location(index)

        #if self.cache_size=0, just read sample directly from file
        if self.cache_size == 0:
            filename = glob2.glob(self.data_dir + "data_" + str(file_id).zfill(3) + ".h5")[0]
            ind_list = [file_sample_id]
            data = self.read_samples(filename, ind_list)
            label = self.all_labels[index]
            return data.reshape(-1), label

        #if there are cached samples, get sample
        # if cache is empty, fill cache
        if self.num_cached_samples == 0:
            if len(self.scheduled_indices)  > 0:
                #first check if remaining samples in scheduled list is less that size of cache
                if len(self.scheduled_indices) < self.cache_size:
                    # get cache for last time
                    self.fill_cache(self.scheduled_indices)
                    self.scheduled_indices = []
                    print('finished caching samples')
                else:
                    # fill cache
                    self.fill_cache(self.scheduled_indices[:self.cache_size])
                    self.scheduled_indices = self.scheduled_indices[self.cache_size:]

        samplelist = self.cached_data[file_id]['index']
        try:
            self.num_cached_samples -= 1
            return self.cached_data[file_id]['data'][samplelist.index(index),:], self.all_labels[index]
        except:
            raise Exception('item not in list')



    def __len__(self):
        return self.num_samples

    def read_samples(self, filename, ids):
        '''
        read sampleids from hdf file
        filename = filename of hdf file containing data
        sample ids  list of sample ids in filename indicating the sample data to be read.
        '''
        try:
            with h5py.File(filename, mode='r', swmr=True) as hf:
                return  hf.get('samples')[(ids)]
        except:
            print(f'in __read_samples, could not read samples {ids} from {filename}')
            return

    def file_location(self, index):
        sample_id = index
        file_sample_id = sample_id % SAMPLES_SET
        file_id = sample_id // SAMPLES_SET
        return sample_id, file_sample_id, file_id

    def fill_cache(self, indices_list):
        # load all samples in indices list, first sorting by subset so that all items in same subset are loaded at once.
        def last(t):
            return t[-1]
        def list_last_elems(l):
            return [last(i) for i in l]
        def first(t):
            return t[0]
        def list_first_elems(l):
            return [first(i) for i in l]

        data_mapping = {}
        self.cached_data = {}
        for index in indices_list:
            sample_id, file_sample_id, file_id = self.file_location(index)
            #check if file_id in dict and put file_sample_id in dictionary
            if file_id not in data_mapping.keys():
                data_mapping[file_id]={'index':[(index, sample_id, file_sample_id)]}#,'sample_ids':[sample_id],'file_sample_ids':[file_sample_id] }
            else:
                data_mapping[file_id]['index'].append((index, sample_id, file_sample_id))


        #Read all samples from each file in turn, store in cached data

        for file_id, file_id_dict in data_mapping.items():
            # sort index by file_sample_id so slices can be read
            data_mapping[file_id]['index'].sort(key = last)
            filename = glob2.glob(data_dir + "data_" + str(file_id).zfill(3) + ".h5")[0]
            n_samples = len(file_id_dict['index'])
            self.cached_data[file_id] = \
                {'data': self.read_samples(filename, list_last_elems(file_id_dict['index'])),
                'index':list_first_elems(file_id_dict['index'])}
            self.num_cached_samples += n_samples
        return


class RandomScheduler(torch.utils.data.RandomSampler):
    '''
    This class to creates a random list for samples in an epoch, and stores a list of the scheduled indices on the datasource

    '''
    def __init__(self, dataset):
        self._num_samples = None
        self.data_source = dataset
        self.n = len(self.data_source)
        self.schedule = []

    def set_schedule(self):
        self.schedule = [self.data_source.sample_ids[i] for i in torch.randperm(self.n).tolist()]
        try:
            self.data_source.scheduled_indices = self.schedule
            return self.schedule
        except:
            raise ValueError('Dataset provided does not allow scheduled indices list to be stored')

    def get_schedule(self):
        return self.schedule

    def __iter__(self):
        return iter(self.set_schedule())


if __name__ == '__main__':

    start = timing.time()
    timing.log("Start Program")
    atexit.register(timing.endlog, start)

    data_dir = ""
    if GENERATE_DATA:
        generate(data_dir)
        read(data_dir)

    # Decide which random_samples are going to be used for training and validation each and store them separately

    rand_ids = list(range(NUM_SAMPLES))
    random.seed(42)
    random.shuffle(rand_ids)

    num_tr_samples = round(0.7*NUM_SAMPLES)
    train_samples = rand_ids[:num_tr_samples]
    val_samples = rand_ids[num_tr_samples:NUM_SAMPLES]
    train_samples.sort()
    val_samples.sort()
    print(f'Created train/valuation split: train: {len(train_samples)}, valuation {len(val_samples)}')
    all_labels = []

    with h5py.File(data_dir + 'labels.h5', mode='r') as hf:
        # Read only labels, ideally stored in a separate file that is small
        all_labels = hf.get('targets')[()]

    train_ds = DatasetWithCaching(data_dir, train_samples, all_labels, cache_size=CACHE_SIZE)
    train_random_schedule = RandomScheduler(dataset = train_ds)
    train_ldr = torch.utils.data.DataLoader(train_ds, shuffle=False, sampler=train_random_schedule, batch_size=40)

    for idx, sample in enumerate(train_ldr):
        last_index = idx

    print(f'last sample read: {sample[0][:,0]}')


