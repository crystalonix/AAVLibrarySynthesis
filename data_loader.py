import numpy as np
import common_utils


class BatchDataLoader:
    def __init__(self, batch_size, number_of_batches, seed=-1):
        self.batch_size = batch_size
        self.number_of_batches = number_of_batches
        self.number_of_samples = number_of_batches * batch_size
        if seed >= 0:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        self.p1 = rng.permutation(common_utils.NUMBER_OF_ASSEMBLED_SEQUENCES)[0:int(self.number_of_samples / 2)]
        self.p2 = rng.permutation(common_utils.NUMBER_OF_UNASSEMBLED_SEQUENCES)[0:int(self.number_of_samples / 2)]

        with open(common_utils.positive_data_file) as fl:
            self.pos_sequences = fl.readlines()

        # if neg_sequences is None:
        with open(common_utils.negative_date_file) as fl:
            self.neg_sequences = fl.readlines()

    def get_batch_data(self, batch_id, randomize=True):
        pi_batch_pos = self.p1[int(batch_id * (self.batch_size / 2)):int((batch_id + 1) * self.batch_size / 2)]
        pi_batch_neg = self.p2[int(batch_id * self.batch_size / 2):int((batch_id + 1) * self.batch_size / 2)]
        sequence_data = np.array(
            [common_utils.sequence_to_number(self.pos_sequences[pi_batch_pos[0]], common_utils.SEQUENCE_LENGTH)])
        for k in range(1, len(pi_batch_pos)):
            sequence_data = np.r_[
                sequence_data, common_utils.sequence_to_number(self.pos_sequences[pi_batch_pos[k]],
                                                               common_utils.SEQUENCE_LENGTH).reshape(1, -1)]
        for k in range(len(pi_batch_neg)):
            sequence_data = np.r_[
                sequence_data, common_utils.sequence_to_number(self.neg_sequences[pi_batch_neg[k]],
                                                               common_utils.SEQUENCE_LENGTH).reshape(1, -1)]

        labels = np.ones(len(pi_batch_pos))
        labels = np.r_[labels, np.zeros(len(pi_batch_neg))]
        sequence_data = np.c_[sequence_data, labels]
        if randomize:
            sequence_data = common_utils.randomly_shuffle_data(sequence_data)
        return sequence_data


###################################################################################
##                            usage of the batch dataloader                      ##
###################################################################################

dtloader = BatchDataLoader(batch_size=1000, number_of_batches=10)
seq_dt = dtloader.get_batch_data(batch_id=5)
print(f'check the data size: {len(seq_dt)}')
