# Performance Analysis on Placement and Migration Strategies

### Objective

To present the difference between doing data placement and migration and not doing.

### Settings

HBM capacity: 4 GB

Model weight size: 0.5B (FP16) = 1 GB

Number of Prefill tokens: 1024 = 1 GB

Number of Decode tokens: 

- 1024 (1 GB) for HBM size is enough to handle the inference
- 4096 (4 GB) for HBM size is little insufficient to handle the inference
- 8192 (8 GB) for HBM size is very insufficient to handle the inference
- 16384 (16 GB) for attention computation required more data from KV cache 



### Performance

We compare the performance of 2 different method in different situations (HBM is exclusive):

- baseline:
  - Initialization (HBMInit): All model weight are stored on the HBM, all prefill tokens' KV cache are stored on the HBM
  - Placement (PreferHBM): If HBM has storage space, store the current write data on HBM, else write to the external memory
  - Migration: No migration
- Self-derived strategy:
  - Initialization: We set the ratio of HBM bandwidth / (HBM bandwidth + external memory bandwidth) as the **best ratio** that HBM should be read at each step. In the initialization, we stored the best ratio of model weight on HBM and remaining on the external memory. We suppose the best ratio of read on the model weight at each step will come from the HBM and the remaining come from the external memory. For prefill tokens, we also store them in this best ratio.
  - Placement: We track how many tokens' KV cache of the current layer are stored on the HBM. If the ratio of it and total tokens exceeds the best ratio, we write the current token's KV cache of the current layer to the external memory. Otherwise, we write it to the HBM.
  - Migration: By look ahead the trace file, we firstly migrate the skipped tokens of the next token at current layer to the external memory. Then, we adjust the distribution of tokens on the HBM and the external memory to make sure that we can reach the best ratio read on HBM at next token's current layer. *Should suppose the skipped tokens of the current token and next token do not have huge difference. The difference is better covered in the read size from the external memory.*

##### HBM not fully utilized

![](.\plots\mem_usage_1_1_1.png)

Since HBM is enough to handle all the works, the baseline method fully utilized the HBM bandwidth, while the self-derived strategy leverage the bandwidth of HBM and the external memory.

**Alpha rate changes:**

Baseline

<img src=".\plots\alpha_2048_HBM_skip0.png" style="zoom:20%;" />

No token skipped:



40% token skipped:

##### HBM fully utilized and exceed a little

![](.\plots\mem_usage_1_1_4.png)

No token skipped:

40% token skipped:

##### HBM fully utilized and exceed a lot

![](.\plots\mem_usage_1_1_8.png)

No token skipped:

40% token skipped: