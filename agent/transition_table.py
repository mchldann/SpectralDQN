import torch
import torch.nn as nn
import numpy as np
from random import randrange

# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():

    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False    # Used to track actual capacity
        self.sum_tree = np.zeros((2 * size - 1, ), dtype=np.float32)    # Initialise fixed size tree with all (priority) zeros
        self.data = np.zeros((size), dtype=np.int32) # Wrap-around cyclic buffer
        self.max = 1    # Initial max value to return (1 = 1^w)


    # Propagates value up tree given a tree index
    def _propagate(self, tree_idx, value):
        parent = (tree_idx - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)


    # Updates value given a tree index
    def update(self, tree_idx, value):
        self.sum_tree[tree_idx] = value    # Set new value
        self._propagate(tree_idx, value)    # Propagate value
        self.max = max(value, self.max)


    def append(self, data, value):
        self.data[self.index] = data    # Store data in underlying data structure
        self.update(self.index + self.size - 1, value)    # Update tree
        self.index = (self.index + 1) % self.size    # Update index
        self.full = self.full or self.index == 0    # Save when capacity reached
        self.max = max(value, self.max)


    # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])


    # Searches for a value in sum tree and returns value, data index and tree index
    def find(self, value):
        index = self._retrieve(0, value)    # Search for index of item from root
        data_index = index - self.size + 1
        return (self.sum_tree[index], data_index, index)    # Return value, data index, tree index


    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]


    def total(self):
        return self.sum_tree[0]


class TransitionTable(object):

    def __init__(self, transition_params, agent):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transition_params = transition_params
        self.agent = agent

        self.agent_params = transition_params["agent_params"]
        self.gpu = self.agent_params["gpu"]
        self.hist_len = self.agent_params["hist_len"]
        self.discount = self.agent_params["discount"]
        self.downsample_w = self.agent_params["downsample_w"]
        self.downsample_h = self.agent_params["downsample_h"]
        self.n_step_n = self.agent_params["n_step_n"]
        self.num_freqs = self.agent_params["num_freqs"]

        self.replay_size = transition_params["replay_size"]
        self.hist_spacing = transition_params["hist_spacing"]
        self.bufferSize = transition_params["bufferSize"]

        if self.agent.prioritized:
            self.index_priorities = SegmentTree(self.replay_size)

        self.zeroFrames = True
        self.recentMemSize = self.hist_spacing * self.hist_len
        self.numEntries = 0
        self.insertIndex = 0
        self.buf_ind = self.bufferSize + 1 # To ensure the buffer is always refilled initially

        self.histIndices = []

        for i in range(0, self.hist_len):
            self.histIndices.append(i * self.hist_spacing)

        self.s = torch.empty(self.replay_size, self.downsample_w, self.downsample_h, dtype=torch.uint8).zero_()
        self.a = np.zeros((self.replay_size), dtype=np.int32)
        self.r = np.zeros((self.replay_size), dtype=np.float32)
        self.r_spectral = np.zeros((self.replay_size, 2 * self.num_freqs), dtype=np.float32)
        self.ret_spectral = np.zeros((self.replay_size, 2 * self.num_freqs), dtype=np.float32)
        self.ret_partial = np.zeros((self.replay_size), dtype=np.float32)
        self.ret_partial_pos_spectral = np.zeros((self.replay_size, self.num_freqs), dtype=np.float32)
        self.ret_partial_neg_spectral = np.zeros((self.replay_size, self.num_freqs), dtype=np.float32)
        self.t = np.zeros((self.replay_size), dtype=np.int32)
        self.timed_out = np.zeros((self.replay_size), dtype=np.int32)
        self.steps_until_term = np.zeros((self.replay_size), dtype=np.int32)

        self.recent_s = []
        self.recent_a = []
        self.recent_t = []

        self.buf_a = np.zeros((self.bufferSize), dtype=np.int32)
        self.buf_r = np.zeros((self.bufferSize), dtype=np.float32)
        self.buf_r_spectral = np.zeros((self.bufferSize, 2 * self.num_freqs), dtype=np.float32)
        self.buf_ret_spectral = np.zeros((self.bufferSize, 2 * self.num_freqs), dtype=np.float32)
        self.buf_ret_partial = np.zeros((self.bufferSize), dtype=np.float32)
        self.buf_ret_partial_pos_spectral = np.zeros((self.bufferSize, self.num_freqs), dtype=np.float32)
        self.buf_ret_partial_neg_spectral = np.zeros((self.bufferSize, self.num_freqs), dtype=np.float32)
        self.buf_term_under_n = np.zeros((self.bufferSize), dtype=np.float32)
        self.buf_timed_out_under_n = np.zeros((self.bufferSize), dtype=np.float32)
        self.buf_s = torch.zeros(self.bufferSize, self.hist_len, self.downsample_w, self.downsample_h).to(self.device)
        self.buf_s_plus_n = torch.zeros(self.bufferSize, self.hist_len, self.downsample_w, self.downsample_h).to(self.device)
        self.buf_tree_idx = np.zeros((self.bufferSize), dtype=np.int32)
        self.buf_weight = np.zeros((self.bufferSize), dtype=np.float32)


    def size(self):
        return self.numEntries


    def is_full(self):
        return (self.numEntries == self.replay_size)


    def fill_buffer(self):

        assert self.numEntries > self.bufferSize, 'Not enough transitions stored to learn'

        # clear CPU buffers
        self.buf_ind = 0

        for buf_ind in range(0, self.bufferSize):
            s, a, r, r_spectral, ret_spectral, ret_partial, ret_partial_pos_spectral, ret_partial_neg_spectral, s_plus_n, term_under_n, timed_out_under_n, tree_idx, weight = self.sample_one()
            self.buf_s[buf_ind].copy_(s)
            self.buf_a[buf_ind] = a
            self.buf_r[buf_ind] = r
            np.copyto(self.buf_r_spectral[buf_ind], r_spectral)
            np.copyto(self.buf_ret_spectral[buf_ind], ret_spectral)
            self.buf_ret_partial[buf_ind] = ret_partial
            np.copyto(self.buf_ret_partial_pos_spectral[buf_ind], ret_partial_pos_spectral)
            np.copyto(self.buf_ret_partial_neg_spectral[buf_ind], ret_partial_neg_spectral)
            self.buf_s_plus_n[buf_ind].copy_(s_plus_n)
            self.buf_term_under_n[buf_ind] = term_under_n
            self.buf_timed_out_under_n[buf_ind] = timed_out_under_n
            self.buf_tree_idx[buf_ind] = tree_idx
            self.buf_weight[buf_ind] = weight

        self.buf_s.div_(255)
        self.buf_s_plus_n.div_(255)


    def sample_one(self):

        assert self.numEntries > 1, 'Experience cache is empty'

        valid = False

        while not valid:

            if self.agent.prioritized:

                # Retrieve sum of all priorities (used to create a normalised probability distribution)
                p_total = self.index_priorities.total()

                sample = np.random.uniform(0.0, p_total)

                # Retrieve sample from tree with un-normalised probability
                unnormalised_prob, data_index, tree_idx = self.index_priorities.find(sample)

                if unnormalised_prob == 0:
                    continue
                else:
                    index = self.index_priorities.get(data_index)

            else:
                index = randrange(0, self.numEntries)

            ar_index = self.wrap_index(index + self.recentMemSize - 1)

            if self.t[ar_index] == 0:
                valid = True

        if self.agent.prioritized:
            prob = unnormalised_prob / p_total
            weight = (self.numEntries * prob) ** -self.agent.priority_weight # Compute importance-sampling weight w (normalised by dividing by the largest weight in the batch later on)
        else:
            tree_idx = -1
            weight = 1.0

        s = self.concatFrames(index, False)
        s_plus_n = self.concatFrames(self.wrap_index(index + self.n_step_n), False)

        term_under_n = 0
        if (self.steps_until_term[ar_index] - 1) < self.n_step_n:
            term_under_n = 1

        timed_out_under_n = 0
        for i in range(0, self.n_step_n):
            if self.timed_out[self.wrap_index(ar_index + i + 1)] == 1:
                timed_out_under_n = 1

        return s, self.a[ar_index], self.r[ar_index], self.r_spectral[ar_index], self.ret_spectral[ar_index], self.ret_partial[ar_index], self.ret_partial_pos_spectral[ar_index], self.ret_partial_neg_spectral[ar_index], s_plus_n, term_under_n, timed_out_under_n, tree_idx, weight


    def sample(self, batch_size):

        assert batch_size < self.bufferSize, 'Batch size must be less than the buffer size'

        if self.buf_ind + batch_size > self.bufferSize:
            self.fill_buffer()

        index = self.buf_ind
        index2 = index + batch_size

        self.buf_ind = self.buf_ind + batch_size

        return self.buf_s[index:index2], self.buf_a[index:index2], self.buf_r[index:index2], self.buf_r_spectral[index:index2], self.buf_ret_spectral[index:index2], self.buf_ret_partial[index:index2], self.buf_ret_partial_pos_spectral[index:index2], self.buf_ret_partial_neg_spectral[index:index2], self.buf_s_plus_n[index:index2], self.buf_term_under_n[index:index2], self.buf_timed_out_under_n[index:index2], self.buf_tree_idx[index:index2], self.buf_weight[index:index2]


    def update_priorities(self, tree_indices, priorities):
        priorities = np.power(priorities, self.agent.priority_exponent)
        [self.index_priorities.update(tree_idx, priority) for tree_idx, priority in zip(tree_indices, priorities)]


    def concatFrames(self, index, use_recent):

        if use_recent:
            s, t = self.recent_s, self.recent_t
        else:
            s, t = self.s, self.t

        fullstate = torch.empty(self.hist_len, self.downsample_w, self.downsample_h, dtype=torch.uint8).zero_()

        # Zero out frames from all but the most recent episode.
        # This is achieved by looking *back* in time from the current frame (index + self.histIndices[self.hist_len - 1])
        # until a terminal state is found. Frames before the terminal state then get zeroed.
        # The index logic is a bit tricky... The index where self.t[index] == 1 is actually the first state of the new episode.
        zero_out = False
        episode_start = self.hist_len - 1

        for i in range(self.hist_len - 2, -1, -1):

            if not zero_out:

                for j in range(index + self.histIndices[i], index + self.histIndices[i + 1]):

                    if t[self.wrap_index(j, use_recent)] == 1:
                        zero_out = True
                        break

            if zero_out:
                fullstate[i].zero_()
            else:
                episode_start = i

        # Could get rid of this since it's never called. Just here to match up with the Lua code (where it is also never called).
        if not self.zeroFrames:
            episode_start = 0

        # Copy frames from the current episode.
        for i in range(episode_start, self.hist_len):
            fullstate[i].copy_(s[self.wrap_index(index + self.histIndices[i], use_recent)])

        return fullstate


    def get_recent(self):

        # Assumes that the most recent state has been added, but the action has not
        return self.concatFrames(0, True).float().div(255)


    def wrap_index(self, index, use_recent=False):

        if use_recent:
            return index

        if self.numEntries == 0:
            return index

        while index < 0:
            index += self.numEntries

        while index >= self.numEntries:
            index -= self.numEntries

        return index


    def add(self, s, a, r, r_spectral, ret_spectral, ret_partial, ret_partial_pos_spectral, ret_partial_neg_spectral, term, timed_out, steps_until_term):

        assert s is not None, 'State cannot be nil'
        assert a is not None, 'Action cannot be nil'
        assert r is not None, 'Reward cannot be nil'

        term_stored_value = 0
        if term:
            term_stored_value = 1

        timed_out_stored_value = 0
        if timed_out:
            timed_out_stored_value = 1

        # Overwrite (s, a, r, t) at insertIndex
        self.s[self.insertIndex].copy_(s.mul(255).byte())
        self.a[self.insertIndex] = a
        self.r[self.insertIndex] = r
        np.copyto(self.r_spectral[self.insertIndex], r_spectral)
        np.copyto(self.ret_spectral[self.insertIndex], ret_spectral)
        self.ret_partial[self.insertIndex] = ret_partial
        np.copyto(self.ret_partial_pos_spectral[self.insertIndex], ret_partial_pos_spectral)
        np.copyto(self.ret_partial_neg_spectral[self.insertIndex], ret_partial_neg_spectral)
        self.t[self.insertIndex] = term_stored_value
        self.timed_out[self.insertIndex] = timed_out_stored_value
        self.steps_until_term[self.insertIndex] = steps_until_term

        if self.agent.prioritized:
            # Store new transition with maximum priority
            self.index_priorities.append(self.insertIndex, self.index_priorities.max)

        # Increment until at full capacity
        if self.numEntries < self.replay_size:
            self.numEntries += 1

        # Always insert at next index, then wrap around
        self.insertIndex += 1

        # Overwrite oldest experience once at capacity
        if self.insertIndex >= self.replay_size:
            self.insertIndex = 0


    def add_recent_state(self, s, term):

        s = s.mul(255).byte()

        if len(self.recent_s) == 0:
            for i in range(0, self.recentMemSize):
                self.recent_s.append(torch.zeros_like(s))
                self.recent_t.append(1)

        self.recent_s.append(s)

        if term:
            self.recent_t.append(1)
        else:
            self.recent_t.append(0)

        # Keep recentMemSize states.
        if len(self.recent_s) > self.recentMemSize:
            del self.recent_s[0]
            del self.recent_t[0]
