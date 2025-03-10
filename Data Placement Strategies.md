### Set up

Model: 

- 32 layers with FP16. 
- 75B parameters.
- 30000 tokens are generated in the prefill stage
- will generate 4096 tokens in the decode stage

Memory: 

- HBM

  - 141GB
  - 4915 GB/s bandwidth

- the external bandwidth

  - interface bandwidth 900 GB/s for read / write

  - internal bandwidth 1900 GB/s

    

### Data Placement Strategies:

- PreferHBM: Store all KV cache on the HBM. When the HBM is full, store on the external memory.
- SplitToken: For each token, store part of layers' KV cache on the HBM, and the remaining on the external memory.
- BatchRatio: Take a batch size (e.g. 16 tokens). Part of tokens in the batch are stored on the HBM, and the remaining on the external memory.
- LookAheadBatch: Look ahead to see if the current token n or layer is skipped or not in the future (e.g. future 16 tokens). If it appears in the future skipped token list, we store it on the external memory.
- LayerImportance: Consider each layer's importance. If a layer was skipped frequently in the past, we place the generated KV cache at current layer l on the external memory.

### Data Migration Strategies:

- No migration
- PriorMigration: When reach the threshold, migrate some tokens KV cache (e.g n=0 to n=15, 16 tokens) from HBM to the external memory.
- SkippedTokensMigration: When reach the threshold, migrate skipped tokens at current step from the HBM to the external memory. 
- PastWindowMigration: Look the past window size (e.g. past 16 tokens) tokens to accumulate the skipped tokens of them. Migrate those skipped tokens from the HBM to the external memory.
- LookAheadMiration: Look ahead the next token n+1 to migrate important tokens that from the external memory to HBM. Unimportant tokens from the HBM to the external memory.
- LookAheadBatchMigration: Look ahead the next batch (e.g. 16 tokens) of tokens to migrate important tokens that from the external memory to HBM. Unimportant tokens from the HBM to the external memory.
- AlphaMigration: Look ahead the n+1 token's alpha, try to migrate KV caches to maintain it at the best ratio (bandwidth of HBM : bandwidth of the external memory) which leverage the bandwidth of HBM and the external memory.

### Performance

Plots present the performance of combinations of different data placement strategies with data migration strategies.

**alpha**: How much data will read from HBM at step token n, layer l, and s (0 for MHA and 1 for MLP)

When **whole model weights** are on the HBM (model weights and KV cache at prefill stage are all read from HBM):

![](.\plots\PreferHBM.png)

![](.\plots\split.png)

![](.\plots\batchRatio.png)

![](.\plots\laB.png)

![](.\plots\layerimp.png)

The alpha is really high at above set of plots because all previous KV cache generated in the prefill stage will read from the HBM.

The performance of **look ahead migration** looks not good may because the data size with token granularity in the migration is high at each step. The **AlphaMigration** gains better performance because it migrates data in fine granularity (layer level), reducing the data size in migration.

When **part of the model weights** are on the HBM (0.8 model weights and prefill KV cache read from HBM):

![](.\plots\PreferHBM1.png)

![](.\plots\Split1.png)

![](.\plots\BatchRatio1.png)

![](.\plots\LookAheadB1.png)

![](.\plots\layerimp1.png)

### Next Steps

- Since strategies of maintaining the future alpha rate that derived from the ratio of bandwidth between the HBM and the external memory, we could do more tests on it
  - Try to compare the strategies that maintaining different alpha values.
  -  Analyze the performance under different ratios of model weights on the HBM (different initial states)
  -  Try different batch sizes of AlphaMigration from only look n+1 to look a range (n+1, n+k)





加dimension

- inclusive exclusive
- initial states：model weights如何分部，prefill的kv cache如何分部
- passive data migration：在这一层从ext mem 来的KV cache是否需要移动到HBM，可以节省读的bandwidth
- 不同level的token skipping：不skip，skip 30%，60%， 100%（model weight）
