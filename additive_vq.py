import numpy as np
import torch
from torch.distributions import normal, uniform
import itertools


class AVQ(torch.nn.Module):
    def __init__(self, num_stages, vq_bitrate_per_stage, data_dim, beam_width, discard_threshold=0.01, device=torch.device('cpu')):
        super(AVQ, self).__init__()

        self.num_stages = num_stages # number of stages used for vector quantization
        self.num_codebooks = int(2 ** vq_bitrate_per_stage) # number of codebook entries for each stage
        self.data_dim = data_dim # data samples or codebook entries dimension
        self.beam_width = beam_width # beam width used for beam search in additive VQ
        self.eps = 1e-12
        self.device = device
        self.dtype = torch.float32
        self.normal_dist = normal.Normal(0,1)
        # threshold (in percent) for discarding the unused codebook entries (used in "replace_unused_codebooks" function)
        self.discard_threshold = discard_threshold

        # sample initial codebooks from Normal distribution
        # initial_codebooks = torch.randn(self.num_stages * self.num_codebooks, self.data_dim, dtype=self.dtype, device=self.device)

        # sample initial codebooks from Uniform distribution (you can initialize with Gaussian distribution or any other initialization you prefer for your application)
        initial_codebooks = torch.zeros(self.num_stages * self.num_codebooks, self.data_dim, device=device)
        for k in range(num_stages):
            initial_codebooks[k * self.num_codebooks:(k + 1) * self.num_codebooks] = uniform.Uniform(-1 / self.num_codebooks,1 / self.num_codebooks).sample(
                [self.num_codebooks, self.data_dim])

        self.codebooks = torch.nn.Parameter(initial_codebooks, requires_grad=True)

        # used for counting how many times codebooks entries are being used
        self.codebooks_used = torch.zeros(self.num_stages * self.num_codebooks, dtype=torch.int32, device=self.device)



    def forward(self, input_data, train_mode):

        """
        This function performs additive vector quantization function on input data batch.

        N: number of data samples in input_data
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        inputs:
                input_data (input data matrix which is going to be quantized | shape: (NxD) )
                train_mode: True for training mode | False for inference (evaluation) mode
        outputs:
                final_input_quantized_nsvq or final_input_quantized (vector quantized form of input_data| shape: (NxD) )
                codebooks_used: Codebook entries usage for all codebooks (shape: num_stages x num_coebook_entries_per_stage)
        """

        with torch.no_grad():
            best_tuples, selected_indices = self.main(input_data) # finds the best elements in all codebooks for quantization

        quantized_input_list = []

        for i in range(self.num_stages):
            quantized_input_list.append(self.codebooks[best_tuples[:, i]])

        final_input_quantized = sum(quantized_input_list[:])

        final_input_quantized_nsvq = self.noise_substitution_vq(input_data, final_input_quantized)

        with torch.no_grad():
            # increment the used codebooks indices by one
            self.codebooks_used[selected_indices] += 1
            used_entries = np.asarray(np.array_split(self.codebooks_used.cpu().numpy(), self.num_stages, axis=0))

        # return "final_input_quantized_nsvq" for training phase to pass the gradients over non-differentiable
        # Vector Quantization module in backpropagation.
        if train_mode==True:
            return final_input_quantized_nsvq, used_entries
        else: # otherwise return "final_input_quantized" for inference (evaluation) phase
            return final_input_quantized.detach(), used_entries



    def main(self, input_data):
        self.batch_size = input_data.shape[0]

        indexing_tuples_general = np.zeros((self.batch_size, self.beam_width, self.num_stages))
        indexing_tuples_general[:, :, :] = np.nan
        codebooks = self.codebooks.detach().clone()
        distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
                     - 2 * (torch.matmul(input_data, codebooks.t()))
                     + torch.sum(codebooks.t() ** 2, dim=0, keepdim=True))
        distances_sorted, indices = torch.sort(distances, dim=1)
        sorted_indices = indices[:, 0:self.beam_width]
        indexing_tuples_general[:, :, 0] = sorted_indices.cpu().detach().numpy()

        best_tuples, selected_indices = self.beam_searching(indexing_tuples_general, input_data, codebooks)

        return best_tuples, selected_indices



    def noise_substitution_vq(self, input_data, hard_quantized_input):
        random_vector = self.normal_dist.sample(input_data.shape).to(input_data.device)
        norm_hard_quantized_input = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()
        vq_error = ((norm_hard_quantized_input / norm_random_vector + self.eps) * random_vector)
        quantized_input = input_data + vq_error
        return quantized_input



    def codebook_generator(self, iter_number, case):
        codebooks_temp = self.codebooks.detach().clone()
        for k in range(iter_number+1):
            codebooks_temp[case[k] * self.num_codebooks:(case[k] + 1) * self.num_codebooks, :] = float('nan')
        return codebooks_temp



    def find_best_tuples(self, remainder, mask, iter_number, indexing_tuples_prime):
        mask_prime = np.sort(mask[:,:,0:iter_number+1], axis=2)
        remainder = remainder.reshape(-1, self.data_dim)
        indexing_tuples_prime = indexing_tuples_prime.reshape(self.batch_size*self.beam_width,self.beam_width,self.num_stages)

        cases = list(itertools.combinations(np.arange(self.num_stages, dtype=np.int8), iter_number + 1))
        cases = np.asarray(cases[:])

        for j in range(len(cases)):
            case_indices = np.where((mask_prime.reshape(-1,iter_number+1) == cases[j]).all(axis=1))[0]
            codebooks_temp = self.codebook_generator(iter_number, cases[j])
            input_data = remainder[case_indices,:]
            distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
                         - 2 * (torch.matmul(input_data, codebooks_temp.t()))
                         + torch.sum(codebooks_temp.t() ** 2, dim=0, keepdim=True))
            distances_sorted, indices = torch.sort(distances, dim=1)
            sorted_indices = indices[:,0:self.beam_width]
            indexing_tuples_prime[case_indices,0:self.beam_width,iter_number+1] = sorted_indices.cpu().detach().numpy()

        indexing_tuples_prime = indexing_tuples_prime.reshape(self.batch_size, self.beam_width**2, self.num_stages)
        return indexing_tuples_prime



    # applies vector quantization and calculates the remainder of the quantization
    def vq_and_remainder(self, input_tuples ,input_data, codebooks, iter_number):
        quantized_input = codebooks[input_tuples[:, :, iter_number].reshape(-1)]
        remainder = input_data - quantized_input.reshape(self.batch_size, self.beam_width, self.data_dim)
        return remainder



    def mse_calculation(self, input_data, codebooks, indexing_tuples_prime, iter_number):
        final_quantized_input = torch.zeros((self.batch_size, self.beam_width**2, self.data_dim), device=input_data.device, dtype=input_data.dtype)

        for i in range(iter_number+2):
            quantized_input = codebooks[indexing_tuples_prime[:, :, i]]
            final_quantized_input += quantized_input

        mse = torch.mean((input_data.unsqueeze(1) - final_quantized_input).square(), dim=2)
        return mse



    def beam_searching(self, input_tuples, input_data, codebooks):

        for iter_number in range(self.num_stages-1):
            indexing_tuples_prime = input_tuples.repeat(repeats=self.beam_width, axis=1)
            mask_general = np.floor(input_tuples / self.num_codebooks)

            vq_input = input_data.unsqueeze(dim=1)
            for i in range(iter_number+1):
                remainder = self.vq_and_remainder(input_tuples, vq_input, codebooks, iter_number)
                vq_input = remainder

            indexing_tuples_prime = self.find_best_tuples(remainder, mask_general, iter_number, indexing_tuples_prime)
            indexing_tuples_prime = np.unique(indexing_tuples_prime, axis=1)

            mse = self.mse_calculation(input_data, codebooks, indexing_tuples_prime, iter_number)
            if iter_number < self.num_stages-2:
                sorted_mse, sorted_tuples_indices = torch.sort(mse, dim=1)
                best_tuples_indices = sorted_tuples_indices[:, 0:self.beam_width].cpu().detach().numpy().reshape(-1)
                indexing_tuples_new = indexing_tuples_prime.reshape(self.batch_size, 1, self.beam_width ** 2,self.num_stages)
                indexing_tuples_new = np.repeat(indexing_tuples_new, self.beam_width, axis=1)
                indexing_tuples_new = indexing_tuples_new.reshape(-1, self.beam_width ** 2, self.num_stages)
                temp = np.arange(np.size(best_tuples_indices))
                indexing_tuples_new = indexing_tuples_new[temp, best_tuples_indices[temp], :]
                input_tuples = indexing_tuples_new.reshape(self.batch_size, self.beam_width, self.num_stages)
            else:
                best_tuples_indices = torch.argmin(mse, dim=1).cpu().detach().numpy()
                indexing_tuples_new = indexing_tuples_prime.reshape(self.batch_size, 1, self.beam_width ** 2,self.num_stages)
                indexing_tuples_new = indexing_tuples_new.reshape(-1, self.beam_width ** 2, self.num_stages)
                temp = np.arange(np.size(best_tuples_indices))
                indexing_tuples_new = indexing_tuples_new[temp, best_tuples_indices[temp], :]

                final_best_tuples = indexing_tuples_new.reshape(self.batch_size, 1, self.num_stages).squeeze(axis=1)
                final_selected_indices = np.unique(final_best_tuples)

                return final_best_tuples, final_selected_indices



    # codebook replacement function: used to replace inactive codebook entries with the active ones
    # call this function periodically with the periods of "num_batches"
    def replace_unused_codebooks(self, num_batches):
        """
        This function is used to replace the inactive codebook entries with the active ones, to make all codebooks
        entries to be used for training. The function has to be called periodically with the periods of "num_batches".
        In more details, the function waits for "num_batches" training batches and then discards the codebook entries
        which are used less than a specified percentage (self.discard_threshold) during this period, and replace them
        with the codebook entries which were used (active).

        Recommendation: Call this function after a specific number of training batches. In the beginning the number of
        replaced codebooks might increase (the number of replaced codebooks will be printed out during training).
        However, the main trend must be decreasing after some training time. If it is not the case for you, increase the
        "num_batches" or decrease the "discard_threshold" to make the trend for number of replacements decreasing.
        Stop calling the function at the latest stages of training in order not to introduce new codebook entries which
        would not have the right time to be tuned and optimized until the end of training.

        Play with "self.discard_threshold" value and the period ("num_batches") you call the function. A common trend
        could be to select the self.discard_threshold from the range [0.01-0.1] and the num_batches from the set
        {100,500,1000,...}. For instance, as a commonly used case, if we set the self.discard_threshold=0.01 and
        num_batches=100, it means that you want to discard the codebook entries which are used less than 1 percent
        during 100 training batches. Remember you have to set the values for "self.discard_threshold" and "num_batches"
        in a logical way, such that the number of discarded codebook entries have to be in a decreasing trend during
        the training phase.

        :param num_batches: period of training batches that you want to replace inactive codebooks with the active ones

        """
        with torch.no_grad():
            unused_indices = torch.where((self.codebooks_used.cpu() / num_batches) < self.discard_threshold)[0]
            used_indices = torch.where((self.codebooks_used.cpu() / num_batches) >= self.discard_threshold)[0]

            unused_count = unused_indices.shape[0]
            used_count = used_indices.shape[0]

            if used_count == 0:
                print(f'####### used_indices equals zero / shuffling whole codebooks ######')
                self.codebooks += self.eps * torch.randn(self.codebooks.size(), device=self.device).clone()
            else:
                used = self.codebooks[used_indices].clone()
                if used_count < unused_count:
                    used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                    used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                else:
                    used_codebooks = used[torch.randperm(used.shape[0])]

                self.codebooks[unused_indices] *= 0
                self.codebooks[unused_indices] += used_codebooks[range(unused_count)] + self.eps * torch.randn((unused_count, self.data_dim), device=self.device).clone()

            # prints out the number of unused codebook vectors among all num_stages codebooks
            print(f'************* Replaced ' + str(unused_count) + f' entries among all codebooks *************')
            self.codebooks_used[:] = 0.0