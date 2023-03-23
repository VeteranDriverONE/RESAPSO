# RESAPSO

RESAPSO, an evolutionary algorithm based on a surrogate model for computationally expensive problems, is the source code of the article "Reliability-enhanced surrogate-assisted particle swarm optimization for feature selection and hyperparameter optimization in landslide displacement prediction."

The development software for the algorithm is Matlab 2018a. RSAPSO.m is the RESAPSO body, CEC2017.m is the publicly avalible benchmark test set, and main.m is the test program entry. 

Regarding the runtime environment, Matlab 2018a meets all requirements except for the SRGTSTToolbox. This toolbox is available at https://sites.google.com/site/felipeacviana/surrogates-toolbox.


# Simple to understand

The difference in feature subsets can greatly affect the accuracy of the prediction model. Among the feature selection methods, the wrapper has the highest accuracy. However, the wrapper requires a large amount of computation time. Therefore, the wrapper is a computationally expensive problem. In addition, hyperparameters are also one of the important factors that affect the accuracy of machine learning algorithms. For this reason, we developed RESAPSO for feature selection and hyperparameter selection. We tested RESAPSO on the landslide displacement prediction problem.

Briefly, RESAPSO can be summarized as two points:

RESAPSO uses three surrogate models (the Kriging model, the radial basis function neural network, and the polynomial regression), and the surrogate models can be arbitrary. When individuals are computed on these surrogate models, three predictive fitnesses are obtained. To find reliable predictive fitnesses from these surrogate models as individual fitnesses, we make use of Bayes' theorem. Briefly, the difference between the previous predicted fitness and the real fitness of the models in the individual neighborhood is sampled as the likelihood, and then the posterior probability of each model is computed. The posterior probabilities are then used as weights and weighted to obtain the final individual fitnesses.

Intuitive fuzzy multi-attribute decision making is used to select outstanding and uncertain individuals from the population and calculate their fitnesses on the true objective function. The outstanding individuals improve the algorithm's local search, while the uncertain individuals improve the algorithm's global exploration. The promising individuals should satisfy the following two conditions: first, their fitness level should be low (to minimize problems), and second, the environment in which they are located should be reliable. We define the reliability of the individual's environment as the consistency across surrogate models within the individual's neighborhood, i.e., the closer all the surrogate models predict the fitnesses for that region, the more reliable they are. The uncertain individual is defined in the opposite way and should satisfy the following two conditions: first, their fitness level should be high (to minimize problems), and second, the environment in which they are located should be unreliable. We extract the above information from the individuals, turn them into intuitionistic fuzzy sets, and use intuitionistic fuzzy multi-attribute decision making to select the most eligible individual.

# Citation Information

Title: Reliability-enhanced surrogate-assisted particle swarm optimization for feature selection and hyperparameter optimization in landslide displacement prediction

Abstract: Landslides are dangerous disasters that are affected by many factors. Neural networks can be used to fit complex observations and predict landslide displacement. However, hyperparameters have a great impact on neural networks, and each evaluation of a hyperparameter requires the construction of a corresponding model and the evaluation of the accuracy of the hyperparameter on the test set. Thus, the evaluation of hyperparameters requires a large amount of time. In addition, not all features are positive factors for predicting landslide displacement, so it is necessary to remove useless and redundant features through feature selection. Although the accuracy of wrapper-based feature selection is higher, it also requires considerable evaluation time. Therefore, in this paper, reliability-enhanced surrogate-assisted particle swarm optimization (RESAPSO), which uses the surrogate model to reduce the number of evaluations and combines PSO with the powerful global optimization ability to simultaneously search the hyperparameters in the long short-term memory (LSTM) neural network and the feature set for predicting landslide displacement is proposed. Specifically, multiple surrogate models are utilized simultaneously, and a Bayesian evaluation strategy is designed to integrate the predictive fitness of multiple surrogate models. To mitigate the influence of an imprecise surrogate model, an intuitional fuzzy set is used to represent individual information. To balance the exploration and development of the algorithm, intuition-fuzzy multiattribute decision-making is used to select the best and most uncertain individuals from the population for updating the surrogate model. The experiments were carried out in CEC2015 and CEC2017. In the experiment, RESAPSO is compared with several well-known and recently proposed SAEAs and verified for its effectiveness and advancement in terms of accuracy, convergence speed, and stability, with the Friedman test ranking first. For the landslide displacement prediction problem, the RESAPSO-LSTM model is established, which effectively solves the feature selection and LSTM hyperparameter optimization and uses less evaluation time while improving the prediction accuracy. The experimental results show that the optimization time of RESAPSO is about one-fifth that of PSO. In the prediction of landslide displacement in the step-like stage, RESAPSO-LSTM has higher prediction accuracy than the contrast model, which can provide a more effective prediction method for the risk warning of a landslide in the severe deformation stage.

Cite this article: Wang, Y., Wang, K., Zhang, M. et al. Reliability-enhanced surrogate-assisted particle swarm optimization for feature selection and hyperparameter optimization in landslide displacement prediction. Complex Intell. Syst. (2023). https://doi.org/10.1007/s40747-023-01010-w
