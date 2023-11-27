import torch
from torch.distributions import normal, uniform



class PVQ(torch.nn.Module):
    def __init__(self, num_stages, vq_bitrate_per_stage, data_dim, discard_threshold=0.01, device=torch.device('cpu')):
        super(PVQ, self).__init__()

        self.num_stages = num_stages # number of stages used for vector quantization
        self.num_codebooks = int(2 ** vq_bitrate_per_stage) # number of codebook entries for each stage
        self.pvq_dim = int(data_dim / num_stages) # data samples or codebook entries dimension for each product VQ stage
        self.eps = 1e-12
        self.device = device
        self.dtype = torch.float32
        self.normal_dist = normal.Normal(0,1)
        # threshold (in percent) for discarding the unused codebook entries (used in "replace_unused_codebooks" function)
        self.discard_threshold = discard_threshold

        initial_codebooks = torch.zeros(self.num_stages, self.num_codebooks, self.pvq_dim, device=device)

        # sample initial codebooks from Uniform distribution (you can initialize with Gaussian distribution or any other initialization you prefer for your application)
        for k in range(num_stages):
            initial_codebooks[k] = uniform.Uniform(-1 / self.num_codebooks, 1 / self.num_codebooks).sample(
                [self.num_codebooks, self.pvq_dim])

        self.codebooks = torch.nn.Parameter(initial_codebooks, requires_grad=True)

        # used for counting how many times codebooks entries are being used
        self.codebooks_used = torch.zeros((num_stages, self.num_codebooks), dtype=torch.int32, device=self.device)


    def forward(self, input_data, train_mode):

        """
        This function performs residual vector quantization function on input data batch.

        N: number of data samples in input_data
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        inputs:
                input_data (input data matrix which is going to be quantized | shape: (NxD) )
                train_mode: True for training mode | False for inference (evaluation) mode
        outputs:
                final_input_quantized_nsvq or final_input_quantized (vector quantized form of input_data| shape: (NxD) )
                codebooks_used: Codebook entries usage for all codebooks (shape: num_stages x num_coebook_entries_per_stage)
        """

        quantized_input_list = []
        min_indices_list = []

        input_data_splited = torch.split(input_data, self.pvq_dim, dim=1)

        for i in range(self.num_stages):
            quantized_input, min_indices = self.hard_vq(input_data_splited[i], self.codebooks[i])
            quantized_input_list.append(quantized_input)
            min_indices_list.append(min_indices)

        final_input_quantized = torch.cat(quantized_input_list[:], dim=1)

        final_input_quantized_nsvq = self.noise_substitution_vq(input_data, final_input_quantized)

        with torch.no_grad():
            # increment the used codebooks indices by one
            for i in range(self.num_stages):
                self.codebooks_used[i, min_indices_list[i]] += 1

        # return "final_input_quantized_nsvq" for training phase to pass the gradients over non-differentiable
        # Vector Quantization module in backpropagation.
        if train_mode==True:
            return final_input_quantized_nsvq, self.codebooks_used.cpu().numpy()
        else: # otherwise return "final_input_quantized" for inference (evaluation) phase
            return final_input_quantized.detach(), self.codebooks_used.cpu().numpy()



    def noise_substitution_vq(self, input_data, hard_quantized_input):
        random_vector = self.normal_dist.sample(input_data.shape).to(input_data.device)
        norm_hard_quantized_input = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()
        vq_error = ((norm_hard_quantized_input / norm_random_vector + self.eps) * random_vector)
        quantized_input = input_data + vq_error
        return quantized_input



    def hard_vq(self, input_data, codebooks):
        distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
                     - 2 * (torch.matmul(input_data, codebooks.t()))
                     + torch.sum(codebooks.t() ** 2, dim=0, keepdim=True))
        min_indices = torch.argmin(distances, dim=1)
        quantized_input = codebooks[min_indices]
        return quantized_input, torch.unique(min_indices)



    # codebook replacement function: used to replace inactive codebook entries with the active ones
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
            for k in range(self.num_stages):

                unused_indices = torch.where((self.codebooks_used[k].cpu() / num_batches) < self.discard_threshold)[0]
                used_indices = torch.where((self.codebooks_used[k].cpu() / num_batches) >= self.discard_threshold)[0]

                unused_count = unused_indices.shape[0]
                used_count = used_indices.shape[0]

                if used_count == 0:
                    print(f'####### used_indices equals zero / shuffling whole codebooks ######')
                    self.codebooks[k] += self.eps * torch.randn(self.codebooks[k].size(), device=self.device).clone()
                else:
                    used = self.codebooks[k, used_indices].clone()
                    if used_count < unused_count:
                        used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                        used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                    else:
                        used_codebooks = used[torch.randperm(used.shape[0])]

                    self.codebooks[k, unused_indices] *= 0
                    self.codebooks[k, unused_indices] += used_codebooks[range(unused_count)] + self.eps * torch.randn(
                        (unused_count, self.pvq_dim), device=self.device).clone()

                # prints out the number of unused codebook vectors for each individual codebook
                print(f'************* Replaced ' + str(unused_count) + f' for codebook {k+1} *************')
                self.codebooks_used[k, :] = 0.0
