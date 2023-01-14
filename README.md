# Machine Learning Problems
<div align="center">    

  <br>[![](https://img.shields.io/badge/author-ZhiZhang-red "author")](https://github.com/LewisZhang-01/)
  ![](https://img.shields.io/badge/dynamic/json?label=GitHub%20Followers&query=%24.data.totalSubs&url=https%3A%2F%2Fapi.spencerwoo.com%2Fsubstats%2F%3Fsource%3Dgithub%26queryKey%3DLewisZhang-01&labelColor=282c34&color=181717&logo=github&longCache=true "followers")
  ![](https://img.shields.io/badge/Python-Jupyter_Notebook-green.svg "Python")
</div>

# 1. Pulsar stars classifier

Pulsar stars are a very rare type of Neutron star that produce radio emission detectable on Earth and they are of considerable scientific interest as probes of space-time and states of matter. Their emission spreads across the sky and produces a detectable pattern of broadband radio emission. However in practice almost all detections are caused by radio frequency interference and noise, making legitimate signals hard to find.

The main purpose of this problem is to build a simple classifier in order to predict wether a detected signal comes from a pulsar star or from other sources such as noises, interferences, etc. In other words, the goal in this assignment is to work with the data to build and evaluate prediction models that capture the relationship between the descriptive features and the target feature "target_class".

## Dataset 

Each candidate is described by 8 continuous variables. The first four are simple statistics obtained from the integrated pulse profile (folded profile). This is an array of continuous variables that describe a longitude-resolved version of the signal that has been averaged in both time and frequency. The remaining four variables are similarly obtained from the DM-SNR curve. These are summarised below:
 
1. Mean of the integrated profile.
2. Standard deviation of the integrated profile.
3. Excess kurtosis of the integrated profile.
4. Skewness of the integrated profile.
5. Mean of the DM-SNR curve.
6. Standard deviation of the DM-SNR curve.
7. Excess kurtosis of the DM-SNR curve.
8. Skewness of the DM-SNR curve.

## Tasks

1. Prepare a data quality plan for the dataset. Mark down all the features where there are potential problems or data quality issues. Propose solutions to deal with the problems identified. Explain why did you choose one solution over potentially many other. It is very important to provide justification for your thinking in this part and to list potential solutions, including the solution that will be implemented to clean the data. In particular, pay attention to missing data and carefully address this issue.

2. Normalise or Standardise your features as necessary. Carefully decide the normalisation or standardisation technique used.

3. Carefully decide the evaluation measure that is best suited to this application and the dataset. Justify your choice -- What characteristics of the application and the dataset made you decide the evaluation measure you chose.

4. Compare a decision tree classifier, a kNN classifier and four SVM classifiers (one each with "linear", "poly", "rbf" and "sigmoid" kernel) based on the evaluation measure selected in Task 3. Carefully decide the evaluation methodology for this comparison (e.g., cross validation or a single train/validation/test split or other alternatives). Explore the effect of different parameter settings on these classifiers and find the winner classifier / parameter setting. Why do you think you got those comparison results? In particular, are you surprised at the relative performance of "linear", "rbf" and "sigmoid" kernels?

5. Based on a filter technique, identify the three most discriminative features and the three least discriminative features in this dataset. Run the SVM classifiers with the four kernels on the top three and the bottom three features. How do the results compare?

6. Carefully identify the most discriminating features to predict the binary outcome of the dataset using one wrapper feature selection technique. This should be done for each of the decision tree, kNN and four SVM classifiers from part Task 4. Report and discuss the differences between the feature subsets produced by the filter (Task 5) and the wrapper technique.

7. Compare the performance of different classifiers using the different feature subsets found in Tasks 5 and 6 and compare it to the results on original dataset that you reported in Task 4. Have the results improved or worsened after feature selection? Is the relative performance of different classifiers and configuration settings in line with your expectation?

8. Plot the ROC curves for the "1" class and the different classification models. What do you learn from this ROC curve? Which classifier/configuration is best suited for this task? Are you satisfied with the performance?

9. *BONUS Question* This part is open-ended -- Take the exploration and the discussion deeper than what is asked in the above questions and gain further insights into:
    * Correlation of the various features with the target class.
    * Feature selection and feature importance.
    * Relative performance of different classifiers (different kernels in case of SVM) and different parameter settings w.r.t different evaluation measures.
    * Effect of different ways of imputing missing values on the final performance of different classifiers.

For more detail about this problem, please check the jupyter notebook file: [Pulsar Stars Classifier](https://github.com/LewisZhang-01/Machine_Learning/blob/main/Pulsar_stars_classifier.ipynb)

# Census Classifiers Ensembles & Reinforcement Learning
## Part A - Census Classifiers Ensembles

1. Clean and prepare the dataset for machine learning analysis. You can do basic feature engineering to make your techniques scalable, but there is no need to go overboard with the dataset cleaning. Carefully consider the evaluation measure(s) that you use for this exercise and justify why you selected the particular evaluation measure(s).

2. Evaluate the performance of three basic classifiers on your dataset: a decision tree with depth at most 3, a neural network with at most 10 hidden nodes and 1-NN. You can do basic parameter tuning, but there is no need to go overboard. The goal in this step is simply to create better than random classifiers.

3. Apply ensembles with bagging using the three classifiers from Task (b). Investigate the performance of each of these classifiers as the ensemble size increases (e.g., in steps of 2 from 2 to 20 members). Using the best performing ensemble size, investigate how changing the number of instances in the bootstrap samples affects classification performance.

4. Apply ensembles with random subspacing using the three classifiers from Task (b). Investigate the performance of each of these classifiers as the ensemble size increases (e.g., in steps of 2 from 2 to 20 members). Using the best performing ensemble size, investigate how changing the number of features used when applying random subspacing affects classification performance.

5. Based on the lectures, which set of classifiers is expected to benefit more from bagging techniques than random subspacing and which classifiers benefit more from random subspacing? For your dataset, determine the best ensemble strategy for each of these classifiers. Discuss if this is in line with what you expected. Discuss if there is enough diversity in your ensemble and what else could you have done to improve the performance of your ensemble.

For more detail about this problem, please check the jupyter notebook file: [Census Classifiers Ensembles](https://github.com/LewisZhang-01/Machine_Learning/blob/main/Census_Classifiers_ensembles_%26_Reinforcement_learning.ipynb)

## Part B - Reinforcement Learning: Balls Bins
You are given a set of 10 balls and 10 bins (numbered 0 to 9). Initially, all the balls are in the bin number 5 and your goal is to ensure that the balls are well distributed across the bins eventually. Ideally, all balls should be in different bins. Train a reinforcement learning (RL) agent that should consider all balls in turn and for each ball $\mathcal{b}$, decide whether to keep the ball in the current bin $\mathscr{B}\mathcal{b}$, move it to the bin $\mathscr{B}\mathcal{b} - 1$ or move it to the bin $\mathscr{B}\mathcal{b} + 1$. Note that the action to move to bin $\mathscr{B}\mathcal{b} - 1$ is only permissible if $\mathscr{B}\mathcal{b} - 1 ≥ 0$ and the action to move it to the bin $\mathscr{B}\mathcal{b} + 1$ is only permissible if $\mathscr{B}\mathcal{b} + 1 ≤ 9$.

### Tasks
1. Carefully decide the features to use in the state representation of your RL agent. Also, carefully decide the reward function for your RL agent and the length of an episode. Note that you can also have contextual features in your state representation, i.e., features that are specific to the ball with the current turn.

2. Implement how the agent interacts with the environment (i.e., the step function, restart function, init function etc.) 

3. Compare the RL strategies PPO, DQN and A2C for this problem. Are they able to learn a consistent winning policy after (i) 50,000 episodes, (ii) 100,000 episodes and (iii) 200,000 episodes? Which policy learns the strategy quickest? Why do you think this is the case? 

4. For the best RL strategy, start with a neural network architecture consisting of 3 hidden layers of 64 neurons each and gradually decrease the number of neurons and number of layers. What is the minimal architecture that still allows you to successfully learn a winning strategy? For example, can you learn a winning strategy with a single hidden layer of 2 neurons? Why do you think this is the case? 

5. Compare the effect of different state representations and different reward functions on the ability of the RL agent to learn a winning strategy quickly.

For more detail about this problem, please check the jupyter notebook file: [Reinforcement Learning - Balls Bins](https://github.com/LewisZhang-01/Machine_Learning/blob/main/Census_Classifiers_ensembles_%26_Reinforcement_learning.ipynb)

## Author Info
Author Name: Zhi Zhang.

Email: lewiszhang01@gmail.com
      
Project Link: [https://github.com/LewisZhang-01/Machine_Learning](https://github.com/LewisZhang-01/Machine_Learning)
