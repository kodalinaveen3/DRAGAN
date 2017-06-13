# DRAGAN (Deep Regret Analytic Generative Adversarial Networks)

Link to our paper - https://arxiv.org/abs/1705.07215

Pytorch implementation (thanks!) - https://github.com/jfsantos/dragan-pytorch

**A stable algorithm for GAN training. Our main contributions are:**
1. We introduce regret minimization to justify why simultaneous GD can work well in GANs. Simple improvements like historical averaging, decaying learning rate can be used for better convergence (read 'Online Gradient Descent' paper by Zinkevich et al. for more) 

https://arxiv.org/abs/1706.03269 

(Recent paper also proposing regret minimization in GANs!)

2. Given the fact that RM is a perfectly good dynamics to solve games in convex settings, we questioned why mode collapse happens. We hypothesize that basins of attraction of spurious local Nash equilibria might be the reason. To mitigate this, we smoothen D (and thus payoffs) using regularization penalty and show that it works better than the vanilla procedure. By no means did we try to experiment exhaustively to find the best way for implementing this idea (we used one simple algorithm/heuristic to demonstrate that our idea works), we leave it to future works to build a good algorithm.

**A new perspective on GAN game (implicit in our paper):**

Think of GANs as a game between the generator (P_g) and the nature (P_real). The discriminator is the judge who assigns them payoffs in each round. The interesting thing about this game is that payoff assignment is learnt to incentivize players (actually just G) to behave as we want. Hence, we just smooth D(x) function (this is enough) and in the limit, you can think of DRAGAN as imposing locally Lipschitz bias (Lipschitz in the small, to be precise and so, traditional divergence interpretation still holds) on both real and fake distributions (over time). We only use real samples in our penalty but you can try using fake samples as well to impose gradient constraints (basically, the entire pixel space in the limit). This would still be very different from the improved WGAN due to the local nature of our constraints. In fact, we were shooting for this but in hindsight, we realised it might not hold at the beginning when generator distribution is far away from the real manifold (given that mode collpase effect is observed close to real samples, this is okay). 
Note that the resulting family of functions (by applying local constraints) is much bigger than the Lipschitz family. And so, two-sided penalty (which is used partly to make the algorithm robust to hyperparameter changes, and the choice of 'C' can mitigate its negative effects in the rare case) isn't as restrictive here as in the case of improved WGAN.

**Procedure (to use our algorithm):**
1. Pick your favorite architecture, objective/loss/payoff functions for G and D.
2. Tune the hyperparameter 'C' (any value slightly bigger than zero but less than 1 will work) and this decides the size of local regions. Do this until it starts working (default value should work in most cases, tuning is more important in domains like NLP). Our intuition is that small values extract better performance from a given architecture due to relaxed restrictions while larger values give more stability.
3. Tune lambda if necessary, this has the usual meaning of regularization intensity.
4. If your results are still bad, go back to Step 1 and try a different architecture+objective.

**Note:** We now have preliminary BogoNet results for improved WGAN. And we beat them in most metrics by a good margin! Do keep an eye out for the next version of this paper.

**A nice blog post/code comparing DRAGAN and improved WGAN (thanks!):**

http://lernapparat.de/more-improved-wgan/

https://github.com/t-vi/pytorch-tvmisc/blob/master/wasserstein-distance/Improved_Training_of_Wasserstein_GAN.ipynb

Good observation that we should perturb in all directions rather than one quadrant. We missed this 'bug' during the NIPS rush (should have used [-1,1]). I guess regularization makes it somewhat robust though. 

Using pixel wise std() isn't the best heuristic in hindsight, basically we wanted to choose local region sizes somewhat based on distribution of modes in the input space. Even perturbing with small gaussian noise will work to some extent, but PRODGAN isn't a good heuristic in my experience. 

I agree that new non-convex areas might result from using two-sided penalties, but gradients in these regions will be restricted. You are somewhat right that if we use optimal transport interpretation, our algorithm is better than global constraints since good transport plans would move the mass locally to keep costs low :) SLOGAN is a good idea, its exactly the Lipschitz constraint (if you can design something that isn't brittle wrt hyperparameter changes). Our theory tells that it should work decently since you are somewhat smoothing the payoffs! (all of these constraints are on a spectrum, any kind of smoothing takes away some performance but gives general stability)

**I am a big fan of open discussions/democratized research. Do post any questions you have at:**
https://www.reddit.com/r/MachineLearning/comments/6da7pu/170507215_how_to_train_your_dragan_training/

**We had an interesting discussion about connections between LSGAN, improved WGAN and DRAGAN (Ian Goodfellow and Martin Arjovsky were kind enough to give their comments). Check it out at:**
https://www.facebook.com/kodali.naveen.90

**To summarize, recently introduced regularized class of GANs which improve stability can be classified into two types:**

1. Global Constraint: 
This involves linking random real + fake sample pairs and imposing gradient constraints between them. Loss-Sensitive GAN was the first paper to introduce these general ideas (two-sided penalty, regularization). Improved WGAN came up with a different way of imposing the same constraint. While the motivations were very different, both roughly fall under the same class of models.

2. Local Constraint:
DRAGAN came up with the novel idea of imposing local constraints (introduced payoff smoothing as a fundamental tool to improve stability in games/GANs), motivated from local Nash equilibria hypothesis. To clarify the difference between these ideas, lets take an example. If we are modeling CIFAR-10, our algorithm might impose constraints between a cat picture + blurry cat picture, but global constraint based algorithms might do it between a cat picture + blurry airplane picture (this argument is in the limit). We never do such a thing and argue that its bad for generative modeling performance. So, our constraint is strictly weaker than the global one. We observed improved performance and stability in our experiments using the local one. 

Essentially, both of these constraints mitigate the issue of local NE. The idea of using smoothing to make games more tractable is very old. We cite works of Nash from 1950s. To convince yourself that both these ideas are more general, rather than being restricted to a specific divergence measure or a loss function, I suggest trying out different generator functions, f-divergences (or) imposing gradient constraints in a variety of other ways possible. They all will improve stability, albeit to different extents. We give motivation for our final choice in the section 4.2 of our paper. Other papers have come out in the recent days which propose similar techniques that we introduce here! 

**Here's some recent related works:**

https://arxiv.org/pdf/1705.09367v1.pdf

https://arxiv.org/pdf/1705.10461.pdf

https://arxiv.org/pdf/1705.07177.pdf

It is always possible to find architectures which are particularly suited for a specific algorithm. So, we highly recommend our BogoNet metric to test for stability of different training procedures. It is inspired from WGAN and improved WGAN papers where they use 4-5 architectures that are known to be hard for vanilla GAN. We scaled up and generalized this experiment, we show results in our paper using a set of 150 different architectures. We pick them randomly to remove any biases. A future research direction is to find out better values for noise, and investigate if there is an optimal value of noise that provides good tradeoff between performance and stability. 

**Some of the repositories that would be helpful and which helped us in our experiments/code (big thanks!):**

https://github.com/igul222/improved_wgan_training

https://github.com/wiseodd/generative-models/tree/master/GAN

https://github.com/openai/improved-gan/tree/master/inception_score

**Fun Trivia:** The acronym DRAGAN is supposed to be Deep Regret Analytic Generative Adversarial Networks, we used this name since regret minimization perspective helped us figure out what was causing mode collapse
