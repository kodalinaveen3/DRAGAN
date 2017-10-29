# DRAGAN (Deep Regret Analytic Generative Adversarial Networks)

Link to our paper - https://arxiv.org/abs/1705.07215

Pytorch implementation (thanks!) - https://github.com/jfsantos/dragan-pytorch

**Procedure (to use our algorithm):**
1. Pick your favorite architecture, objective functions for the game.
2. Tune the hyperparameter 'c'which decides the size of local regions. Our intuition is that small values extract better performance from a given architecture due to relaxed restrictions while larger values give more stability.
3. Tune lambda if necessary, this has the usual meaning of regularization intensity.
4. If your results are still bad, go back to Step 1 and try a different architecture+objective.

**Some of the repositories that would be helpful and which helped us in our experiments/code (big thanks!):**

https://github.com/igul222/improved_wgan_training

https://github.com/wiseodd/generative-models/tree/master/GAN

https://github.com/openai/improved-gan/tree/master/inception_score
