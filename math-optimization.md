### Objective

The effective bandwidth utilization in LLM inference can be reflected by the time. The less time used on the inference represents the higher effective bandwidth utilization ratio. Hence, our objective is to achieve the minimum time cost on LLM inference.    

### Initial State

Since prefilling is mainly a computation bound problem, our initial state starts from the decode stage. At the prefilling stage, we assume that the model weights and KV cache are arranged in a reasonable way.

### Variables

N: N tokens in total

L: L layers in the model

d: **hidden dimension** for each token embedding (sometimes called “model dimension”), equal to number of attention heads $\times$ dimension per head.

h: number of attention heads (so each head has dimension $\frac{d}{h}$).

n: number of previous tokens, start counting from the beginning of the decode stage. 

$s_t$: sequence length at n (number of tokens) in a single‐batch inference of current token.

dtype_size: number of bytes (or bits) per parameter/activation (e.g., 2 bytes for FP16, 1 byte for INT8, etc.).

- HBM Bandwidth (R/W) $B_{HBM}$, $B_{ext}^R$ as the actual read bandwidth and $B_{ext}^W$ as the actual write bandwidth of the external memory. $B_{ext\_internal}$ as the internal memory bandwidth and $B_{ext\_interface}^R$ and $B_{ext\_interface}^W$ as the physical bandwidth of channels for read and write respectively. 
- R/W data size

  - MHA ($s=0$)
    - Read: $D_R(n,l,s)=D_{MHA\_R_{n,l,s}}=(4d^2+2\times (n+N_{pre}) d)\times dtype\_size$ where $N_{pre}$ is the number of previous tokens in prefilling.
    - For each new token n, we need to append $d\times dtype\_size$ on the $D_{MHA\_R_{n,l,s}}$ for token embedding read.
    - Write: $D_W(n,l,s)=D_{MHA\_W_{n,l,s}}=2\times d\times dtype\_size$   store the new KV cache.
  - MLP ($s=1$)
    - Read: $D_R(n,l,s)=D_{MLP\_R_{n,l,s}}=2\times d\times d_{ff}\times dtype\_size$  Typically 2 linear layers of shape ($d\times d_{ff}$) and ($d_{ff}\times d$)
    - Write: $D_W(n,l,s)=0$
  - $D_{tot}(n,l,s)=D_R(n,l,s)+D_W(n,l,s)$
- Let $\alpha_{n,l,s}$ ($0\le\alpha_{n,l,s}\le1$) be the ratio of **data that is accessed** **from HBM** at token n layer l sublayer s. Hence, $\alpha_{n,l,s} * D_{tot}(n,l,s)$ is the data stored on the HBM at step ${n,l,s}$, $(1-\alpha_{n,l,s})*D_{tot}(n,l,s)$ is the data stored on the external memory at step $(n,l,s)$.
- Let $\beta_{n,l,s}$ ($0\le\beta_{n,l,s}\le1$) be the ratio of **data that GPU stores to HBM** at step ${n,l,s}$, so $\beta_{n,l,s} * D_{MHA\_W_{n,l,s}}$ is the data GPU stores to the HBM at step $(n,l,s)$, $(1-\beta_{n,l,s})*D_{MHA\_W_{n,l,s}}$ is the data GPU stores to the external memory at step $i$.
- Data Migration

  - $D_M^R$ is the data read from the external memory for migration, $D^W_M$ is the data write to the external memory for migration. $D_M=D_M^R+D_M^W$ is the total data migration size.

### Constraints

- HBM Capacity $C_{HBM}(n,l,s)=D_{ini}+\sum_{n,l,s}(\beta_{n,l,s}D_W({n,l,s})+D_{M}^R(n,l,s)-D_{M}^W(n,l,s)) \le C_{HBM}^{max},\ ∀i.$ where $D_{ini}$ is initialized data on HBM.

### Formula

To represent the time we used on LLM inference, we calculate the time $T(n,l,s)$ used at each step (token $n$ layer $l$ sublayer $s$). The time depends on the longest time cost on HBM or external memory device.

$T(n,l,s)=max\{T_{HBM}(n,l,s),T_{ext}(n,l,s)\}$         [1]



$D_R(n,l,s)$ represents the data required to **read** from the memory at token n layer l and sublayer s.

$D_W(n,l,s)$ represents the data required to **write** from the memory at token n layer l and sublayer s.

$D_M(n,l,s)$ represents the data required to **migrate** between HBM and external memory at token n layer l and sublayer s.



Function [2] represents the time **HBM** need to take to complete data transfers at step ($n,l,s$).

$T_{HBM}(n,l,s)=\frac{data\ read\ on\ HBM+\ data\ write\ on\ HBM+\ data\ migration\ on\ HBM}{HBM\ bandwidth}=\frac{\alpha_{n,l,s}D_R(n,l,s)+\beta_{n,l,s}D_{W}(n,l,s)+D_M(n,l,s)}{B_{HBM}}$               [2]

where $\alpha_{n,l,s}$ is the ratio of data $D_R(n,l,s)$ required to read from HBM to GPU, $\beta_{n,l,s}$ is the ratio of data $D_W(n,l,s)$ required to write from GPU to HBM.



Function [3] represents the time **external memory** need to take to complete data transfers at step (n,l,s). we need to consider the internal bandwidth and interface bandwidth of the external memory.

The critical path is the time consumed on GPU read + GPU write. The migration could be overlapped during this process. 

$T_{ext}(n,l,s)=\frac{(1-\alpha_{n,l,s})D_R(n,l,s)}{min(B_{ext\_interface}^R,\ B_{ext\_internal})}+max\{\frac{(1-\beta_{n,l,s})D_W(n,l,s)+D_M^W}{B_{ext\_interface}^W},\frac{D_M^R}{B_{ext\_interface}^R},\frac{(1-\beta_{n,l,s})D_W(n,l,s)+D_M}{B_{ext\_internal}}\}$          [3]

where $D_M^R$ is the data read from the external memory for migration, $D^W_M$ is the data write to the external memory for migration.

$D_M=D_M^R+D_M^W$

### Search Space

The space we can search to find the optimal solution can be described as a matrix. 

| Data Placement | $a_{1}$                          | $p_{2}$          | $a_{3}$                          | $p_{4}$          | ...  | $a_{n,l,s}$                      |
| -------------- | -------------------------------- | ---------------- | -------------------------------- | ---------------- | ---- | -------------------------------- |
| G-H            | $\beta_{1,1,0} * D_W(1,1,0)$     | 0                | $\beta_{1,2,0} * D_W(1,2,0)$     | 0                | ...  | $\beta_{1,1,0} * D_W(n,l,s)$     |
| G-N            | $(1-\beta_{1,1,0}) * D_W(1,1,0)$ | 0                | $(1-\beta_{1,2,0}) * D_W(1,2,0)$ | 0                | ...  | $(1-\beta_{n,l,s}) * D_W(n,l,s)$ |
| H-e            | $D_{M}^W(1,1,0)$                 | $D_{M}^W(1,1,1)$ | $D_{M}^W(1,2,0)$                 | $D_{M}^W(1,2,1)$ | ...  | $D_{M}^W(n,l,s)$                 |
| e-H            | $D_{M}^R(1,1,0)$                 | $D_{M}^R(1,1,1)$ | $D_{M}^R(1,2,0)$                 | $D_{M}^R(1,2,1)$ | ...  | $D_{M}^R(n,l,s)$                 |

**Variables to Optimize**:

- $\beta_{n,l,s}$: How much data we should write to HBM at each step.
- $\alpha_{n,l,s}$: How much data we should read from HBM at each step.
- $D_M(n,l,s)$: How to migrate the data.
