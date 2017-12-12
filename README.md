# DRAGAN (Deep Regret Analytic Generative Adversarial Networks)

Link to our paper - https://arxiv.org/abs/1705.07215

Pytorch implementation (thanks!) - https://github.com/jfsantos/dragan-pytorch

**Procedure (to use our algorithm):**
1. Pick your favorite architecture, objective function for the game.
2. Tune the hyperparameter 'c' which decides the size of local regions. Our intuition is that small values extract better performance from a given architecture due to relaxed restrictions while slightly larger values give more stability. Be careful to set it appropriately by taking into account your domain range and making sure that perturbation don't lie on data manifold.
3. Tune lambda if necessary, this has the usual meaning of regularization intensity. Set 'k' to be 1.
4. If your results are still bad, go back to Step 1 and try a different architecture+objective.

**Interesting discussion with Ian Goodfellow and Martin Arjovsky on why GANs are unstable and where improvements come from**

https://www.facebook.com/kodali.naveen.90/posts/1047257878740881

An interesting new paper by Fedus et.al came out following this (Many paths to equilibria) 

https://arxiv.org/abs/1710.08446

**Some of the repositories that would be helpful and which helped us (big thanks!):**

https://github.com/igul222/improved_wgan_training

https://github.com/wiseodd/generative-models/tree/master/GAN

https://github.com/openai/improved-gan/tree/master/inception_score
