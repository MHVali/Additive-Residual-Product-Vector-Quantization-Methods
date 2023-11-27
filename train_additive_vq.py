"""
An example code to show how to train a Additive VQ module on a Normal distribution using 4 codebooks.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from additive_vq import AVQ
import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
beam_width = 8 # Beam width for beam searching algorithm in Additive VQ
num_stages = 3 # Number of codebooks to apply Additive VQ
vq_bitrate_per_stage = 5
embedding_dim = 128
learning_rate = 1e-3
batch_size = 64
num_training_batches = 20000 #number of training updates
normal_mean = 0 #mean for normal distribution
normal_std = 1 #standard deviation for normal distribution
training_log_batches = 1000 #number of batches to get logs of training
replacement_num_batches = 1000 #number of batches to check codebook activity and discard inactive codebook vectors
num_of_logs = int(num_training_batches / training_log_batches)

# Arrays to save the logs of training
total_vq_loss = [] # tracks VQ loss
used_codebook_indices_list = [] # tracks indices of used codebook entries

vector_quantizer = AVQ(num_stages, vq_bitrate_per_stage, embedding_dim, beam_width, device=device)
vector_quantizer.to(device)

optimizer = optim.Adam(vector_quantizer.parameters(), lr=learning_rate)

vq_loss_accumulator = 0

for iter in range(num_training_batches):

    data_batch = torch.normal(normal_mean, normal_std, size=(batch_size, embedding_dim)).to(device)

    optimizer.zero_grad()

    quantized_batch, used_codebook_indices = vector_quantizer(data_batch, train_mode=True)
    vq_loss = F.mse_loss(data_batch, quantized_batch)

    vq_loss.backward()
    optimizer.step()

    vq_loss_accumulator += vq_loss.item()

    # save and print logs
    if (iter+1) % training_log_batches == 0:
        vq_loss_average = vq_loss_accumulator / training_log_batches
        total_vq_loss.append(vq_loss_average)
        vq_loss_accumulator = 0
        used_codebook_indices_list.append(used_codebook_indices)
        print("training iter:{}, vq loss:{:.6f}".format(iter + 1, vq_loss_average))

    # codebook replacement
    if ((iter + 1) % replacement_num_batches == 0) & (iter <= num_training_batches - 2*replacement_num_batches):
        vector_quantizer.replace_unused_codebooks(replacement_num_batches)


save_address = './output/'
os.makedirs(save_address, exist_ok=True)
np.save(save_address + f'total_vq_loss_{num_stages}stages_{vq_bitrate_per_stage}bits_bs{batch_size}_lr{learning_rate}.npy', np.asarray(total_vq_loss))

with open(save_address + f"used_codebooks_{num_stages}stages_{vq_bitrate_per_stage}bits_bs{batch_size}_lr{learning_rate}", "wb") as fp:
    pickle.dump(used_codebook_indices_list, fp)

checkpoint_state = {"vector_quantizer": vector_quantizer.state_dict()}
torch.save(checkpoint_state, save_address + f"vector_quantizer_{num_stages}stages_{vq_bitrate_per_stage}bits_bs{batch_size}_lr{learning_rate}.pt")

print("\nTraining Finished >>> Logs and Checkpoints Saved!!!")

######################## Evaluation (Inference) of Additive VQ #############################

data = torch.normal(normal_mean, normal_std, size=(2**18, embedding_dim)).to(device)
quantized_data = torch.zeros_like(data)

eval_batch_size = 64
num_batches = int(data.shape[0]/eval_batch_size)
with torch.no_grad():
    for i in range(num_batches):
        data_batch = data[(i*eval_batch_size):((i+1)*eval_batch_size)]
        quantized_data[(i*eval_batch_size):((i+1)*eval_batch_size)], _ = vector_quantizer(data_batch, train_mode=False)

mse = F.mse_loss(data, quantized_data).item()
print("Mean Squared Error = {:.4f}".format(mse))