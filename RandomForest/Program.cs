// See https://aka.ms/new-console-template for more information
using RandomForest;

Console.WriteLine("= = = Random Forest Algorithm = = =");

LabeledData[] data = Util.ReadData("C:\\temp\\ionosphere.csv");
data = Util.BalanceClasses(data);
(var train, var test) = Util.Split(data);


DecisionTree tree = new DecisionTree(2, 6, data[0].Length);
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

Console.WriteLine($"Decision tree test accuracy: {((double)numCorrect / test.Length):F3}, ({numCorrect}/{test.Length})");

Forest forest = new Forest(300, train.Length, 10, 4, 12);

forest.Train(train);

numCorrect = 0;

foreach(LabeledData d in train)
{
    Dictionary<int, double> probs = forest.Predict(d);
    double maxProb = double.MinValue;
    int chosenClass = -1;

    foreach(var pair in probs)
    {
        if(pair.Value > maxProb)
        {
            maxProb = pair.Value;
            chosenClass = pair.Key;
        }
    }

    if(chosenClass == d.Label)
    {
        numCorrect++;
    }
}

Console.WriteLine($"Random forest train accuracy: {((double)numCorrect / train.Length):F3}, ({numCorrect}/{train.Length})");

numCorrect = 0;

foreach (LabeledData d in test)
{
    Dictionary<int, double> probs = forest.Predict(d);
    double maxProb = double.MinValue;
    int chosenClass = -1;

    foreach (var pair in probs)
    {
        if (pair.Value > maxProb)
        {
            maxProb = pair.Value;
            chosenClass = pair.Key;
        }
    }

    if (chosenClass == d.Label)
    {
        numCorrect++;
    }
}

Console.WriteLine($"Random forest test accuracy: {((double)numCorrect / test.Length):F3}, ({numCorrect}/{test.Length})");

