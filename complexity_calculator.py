"""
An example code to calculate the complexity of VQ methods in terms of weighted million operations per second (WMOPS).
"""

from complexity import AVQ_Complexity, RVQ_Complexity, PVQ_Complexity


# Parameters of Vector Quantization method
beam_width = 8 # Beam width for beam searching algorithm in Additive VQ (Exclusive for AVQ method)
num_stages = 4 # Number of codebooks to apply Additive VQ (or Additive VQ or Product VQ)
vq_bitrate_per_stage = 5 # VQ bitrate per stage
num_entries_per_stage = int(2**vq_bitrate_per_stage) # number of codebook entries per stage
embedding_dim = 128
single_precision = True # True if single precision, False if Double precision

additive_vq_calculator = AVQ_Complexity(embedding_dim, num_entries_per_stage, num_stages, beam_width, single_precision=True)
residual_vq_calculator = RVQ_Complexity(embedding_dim, num_entries_per_stage, num_stages, single_precision=True)
product_vq_calculator = PVQ_Complexity(embedding_dim, num_entries_per_stage, num_stages, single_precision=True)

additive_vq_complexity = additive_vq_calculator.complexity_calculator()
residual_vq_complexity = residual_vq_calculator.complexity_calculator()
product_vq_complexity = product_vq_calculator.complexity_calculator()

print('AVQ complexity =', additive_vq_complexity, 'WMOPS')
print('RVQ complexity =', residual_vq_complexity, 'WMOPS')
print('PVQ complexity =', product_vq_complexity, 'WMOPS')