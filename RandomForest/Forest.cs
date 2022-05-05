using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RandomForest
{
    internal class Forest
    {
        private int numTrees;
        private int bootStrapSize;
        private List<DecisionTree> trees;

        public Forest(int numTrees, int bootStrapSize, int numClasses)
        {
            this.numTrees = numTrees;
            this.bootStrapSize = bootStrapSize;
            trees = new List<DecisionTree>();
        }

        public void Train(LabeledData[] data)
        {
            for(int i = 0; i < numTrees; i++)
            {
                LabeledData[] bootStrapped = Bootstrap(data, bootStrapSize);
                DecisionTree tree = new DecisionTree(2, 6);
                tree.Train(bootStrapped);
                trees.Add(tree);
            }
        }

        public Dictionary<int, double> Predict(LabeledData d)
        {
            Dictionary<int, int> classVotes = new Dictionary<int, int>();

            foreach(DecisionTree tree in trees)
            {
                int yPred = tree.Predict(d);
                if (!classVotes.ContainsKey(yPred))
                {
                    classVotes.Add(yPred, 0);
                }

                classVotes[yPred]++;
            }


            Dictionary<int, double> classProbs = new Dictionary<int, double>();
            
            foreach(var pair in classVotes)
            {
                classProbs.Add(pair.Key, (double)pair.Value / numTrees);
            }

            return classProbs;
        }

        private LabeledData[] Bootstrap(LabeledData[] data, int numSamples)
        {
            Random rnd = new Random();
            List<LabeledData> bootstrappedData = new List<LabeledData>();

            for(int i = 0; i < numSamples; i++)
            {
                int idx = rnd.Next(data.Length);
                bootstrappedData.Add(data[idx]);
            }

            return bootstrappedData.ToArray();
        }
    }
}
