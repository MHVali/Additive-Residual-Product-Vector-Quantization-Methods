"""
Code to plot the training logs saved during executing the code "train.py". The plots will be saved as a pdf file.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# hyper-parameters you used for training. Now they are needed to load your saved arrays.
num_stages = 3
vq_bitrate_per_stage = 5
batch_size = 64
learning_rate = 1e-3

# create pdf file
pdf_file = PdfPages(f'log_{num_stages}stages_{vq_bitrate_per_stage}bits_bs{batch_size}_lr{learning_rate}.pdf')

# loading the training logs
load_address = './output/'
total_vq_loss = np.load(load_address + f'total_vq_loss_{num_stages}stages_{vq_bitrate_per_stage}bits_bs{batch_size}_lr{learning_rate}.npy')
with open(load_address + f"used_codebooks_{num_stages}stages_{vq_bitrate_per_stage}bits_bs{batch_size}_lr{learning_rate}", "rb") as fp:
    used_codebooks = pickle.load(fp)


num_of_logs = np.size(total_vq_loss)

# plotting used codebook indices during training
num_bars = int(2**vq_bitrate_per_stage)

for j in range(len(used_codebooks)):
    fig = plt.figure(figsize=(9.93,14))
    histogram = np.log10(used_codebooks[j] + 1)
    for i in range(num_stages):
        plt.subplot(num_stages,1,i+1)
        plt.bar(np.arange(1, num_bars + 1), height=histogram[i], width=1)
        plt.title(f'Codebook Usage Histogram During Training for Codebook {i+1}')
        plt.xlabel('Codebook Index')
        plt.ylabel('log10(codebook usage)')
    pdf_file.savefig(fig, bbox_inches='tight')

# plotting VQ loss during training
fig = plt.figure(figsize=(15, 5))
total_vq_loss = total_vq_loss.reshape(-1,1)
plt.plot(total_vq_loss)
plt.title(f'VQ Loss')
plt.xlabel('Training Iterations')
plt.ylabel('Mean Squared Error')
pdf_file.savefig(fig, bbox_inches='tight')

pdf_file.close()
