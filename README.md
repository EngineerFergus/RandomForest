# RandomForest

An implementation of the Random Forest machine learning algorithm. This algorithm relies on ensemble methods where many decision trees are made from random samples of a training dataset to provide a model that does not overfit to the data and gives robust results. Preliminary results gave ~94% accuracy on test data from the ionosphere dataset (201 train, 51 test). Tuning of hyperparameters can and should be done when used in practice.

Hyperparameters for the individual decision trees include:
- Minimum samples in a split
- Maximum depth of the tree
- Maximum number of features to randomly select

Hyperparameters for the random forest include:
- Number of decision trees to train
- Bootstrapped dataset size

These 5 hyperparameters allow for tuning of a model to determine what works best for a given application.

## Sources
- [Python Implementation](https://github.com/Suji04/ML_from_Scratch/blob/master/decision%20tree%20classification.ipynb)
- [Bootstrapping and Feature Randomization](https://towardsdatascience.com/decision-tree-and-random-forest-explained-8d20ddabc9dd)
- [Ionosphere Dataset](https://archive.ics.uci.edu/ml/datasets/Ionosphere)
