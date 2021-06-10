# conjugate-nonparametric-Hawkes-process
The repository implements the demo code for the paper "Efficient Inference for Nonparametric Hawkes Processes Using Auxiliary Latent Variables". It includes three statistical inference methods in the paper. 

We recommend to read the tutorial to get familiar with this module. It introduces the key points in the model and illustrates how to perfrom inference.

For any further details, please refer to the paper. 

Update
=====================================================
We provide a more elegant mean-field variational inference in the module  `conjugate_np_hawkes_new` where the update of probabilistic branching matrix has a closed-form solution. 

In the original mean-field variational inference, we first integrate out the P\'{o}lya-Gamma random variable $\bm{\omega}$ and then take expectation of the logarithm of the joint distribution (without $\omega$) to compute the optimal density for branching matrix. This results in $\tilde{\mu}(t_i)=\tilde{\lambda}_\mu^*e^{\mathbb{E}(\log\sigma(f(t_i)))}$, $\tilde{\phi}(\tau_{ij})=\tilde{\lambda}_\phi^*e^{\mathbb{E}(\log\sigma(g(\tau_{ij})))}$ where $\mathbb{E}(\log\sigma(\cdot))$ has no closed-form solution and we have to resort to numerical methods. On the contrary, in the revised mean-field variational inference, we directly take expectation of the logarithm of the joint distribution (with $\omega$) to compute the optimal density for branching matrix. After integrating out $\omega$, we obtain
