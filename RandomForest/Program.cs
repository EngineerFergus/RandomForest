// See https://aka.ms/new-console-template for more information
using RandomForest;

Console.WriteLine("= = = Random Forest Algorithm = = =");

LabeledData[] data = Util.ReadData("C:\\temp\\ionosphere.csv");
data = Util.BalanceClasses(data);
(var train, var test) = Util.Split(data);


DecisionTree tree = new DecisionTree(2, 6);
tree.Train(train);

int numCorrect = 0;

foreach(LabeledData d in train)
{
    int yPred = tree.Predict(d);
    if(yPred == d.Label)
    {
        numCorrect++;
    }
}

Console.WriteLine($"Decision tree train accuracy: {((double)numCorrect / train.Length):F3}, ({numCorrect}/{train.Length})");

numCorrect = 0;

foreach(LabeledData d in test)
{
    int yPred = tree.Predict(d);
    if(yPred == d.Label)
    {
        numCorrect++;
    }
}

Console.WriteLine($"Decision tree test accuracy: {((double)numCorrect / test.Length)}, ({numCorrect}/{test.Length})");

Forest forest = new Forest();

