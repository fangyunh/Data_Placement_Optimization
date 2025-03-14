# Directions

Add dimensions:

- inclusive / exclusive
- initial states: model weights, prefill stage 结束后生成的KV cache分布
- passive data migration：在执行(n,l,s)这一步的时候，从ext memory读上来的KV cache可以考虑是否要存进HBM内，存了的话就省了读上来的bandwidth
- 不同level的token skipping: 不skip，skip 30%， 50%， 100%；使用不同的traces.txt去模拟
- Extension: When executing step (n,l,s), we can decide to choose which method to compute the KV cache based on the distribution of data on HBM and external memory. If a read cost is too high we can choose the method that read less tokens to reduce the overhead.



steps

- 修改migration，增加D_HBM_MR, D_HBM_MW, D_ext_MR, D_ext_MW和对应的inclusive exclusive
- 修改trace generation，将n从n_pre开始算起，调整不同的skip strategy
- 修改simulator，主要修改时间的计算，考虑inclusive exclusive