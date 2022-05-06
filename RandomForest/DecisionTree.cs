using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RandomForest
{
    internal class DecisionTree
    {
        public DecisionNode? Root { get; private set; }
        public int MinSamplesSplit { get; private set; }
        public int MaxDepth { get; private set; }
        public int MaxFeatures { get; private set; }

        private List<int> shuffledFeatureIndices;

        public DecisionTree(int minSamplesSplit, int maxDepth, int maxFeatures)
        {
            MinSamplesSplit = minSamplesSplit;
            MaxDepth = maxDepth;
            MaxFeatures = maxFeatures;
            shuffledFeatureIndices = new List<int>();
        }

        public void Train(LabeledData[] data)
        {
            for(int i = 0 ;i < data[0].Length; i++)
            {
                shuffledFeatureIndices.Add(i);
            }

            Random rng = new Random();
            shuffledFeatureIndices = shuffledFeatureIndices.OrderBy(x => rng.Next()).ToList();

            MaxFeatures = Math.Min(MaxFeatures, data[0].Length - 1);

            Root = BuildTree(data);
        }

        private DecisionNode BuildTree(LabeledData[] data, int currentDepth = 0)
        {
            int numSamples = data.Length;
            DecisionNode node;

            if(numSamples > MinSamplesSplit && currentDepth <= MaxDepth)
            {
                // run training split
                (node, LabeledData[] left, LabeledData[] right) = GetBestSplit(data);

                if(node.InfoGain > 0)
                {
                    node.LeftNode = BuildTree(left, currentDepth++);
                    node.RightNode = BuildTree(right, currentDepth++);
                    return node;
                }
            }

            node = new DecisionNode()
            {
                IsLeaf = true,
                Value = CalcLeafValue(data)
            };

            return node;
        }

        private (DecisionNode node, LabeledData[] left, LabeledData[] right) GetBestSplit(LabeledData[] data)
        {
            double maxInfoGain = double.MinValue;
            DecisionNode bestNode = new DecisionNode();
            LabeledData[] bestLeft = new LabeledData[0];
            LabeledData[] bestRight = new LabeledData[0];

            for(int i = 0; i < MaxFeatures; i++)
            {
                int index = shuffledFeatureIndices[i];
                var thresholds = data.Select(x => x[index]).Distinct();

                foreach(double threshold in thresholds)
                {
                    (LabeledData[] left, LabeledData[] right) = Split(data, index, threshold);
                    double currentGain = CalcInformationGain(data, left, right);

                    if(currentGain > maxInfoGain)
                    {
                        maxInfoGain = currentGain;
                        bestNode = new DecisionNode()
                        {
                            Index = index,
                            Threshold = threshold,
                            InfoGain = currentGain
                        };

                        bestLeft = left;
                        bestRight = right;
                    }
                }
            }

            return (bestNode, bestLeft, bestRight);
        }

        private static (LabeledData[] left, LabeledData[] right) Split(LabeledData[] data, int featureIdx, double threshold)
        {
            LabeledData[] leftData = data.Where(x => x[featureIdx] <= threshold).ToArray();
            LabeledData[] rightData = data.Where(x => x[featureIdx] > threshold).ToArray();

            return (leftData, rightData);
        }

        private static double CalcInformationGain(LabeledData[] parentData, LabeledData[] leftChildData, LabeledData[] rightChildData)
        {
            double leftWeight = (double)leftChildData.Length / parentData.Length;
            double rightWeight = (double)rightChildData.Length / parentData.Length;

            double parentGini = CalcGiniIndex(parentData);
            double childrenGini = (leftWeight * CalcGiniIndex(leftChildData)) + (rightWeight * CalcGiniIndex(rightChildData));

            return parentGini - childrenGini;
        }

        private int CalcLeafValue(LabeledData[] data)
        {
            if(data.Length == 0) { throw new ArgumentException($"Exception in {nameof(CalcLeafValue)}, empty data array given."); }

            List<int> classes = data.Select(x => x.Label).Distinct().ToList();
            int[] counts = new int[classes.Count];

            foreach(LabeledData d in data)
            {
                for(int i = 0; i < classes.Count; i++)
                {
                    if(classes[i] == d.Label)
                    {
                        counts[i]++;
                        break;
                    }
                }
            }

            int maxCount = int.MinValue;
            int bestIndex = -1;

            for(int i = 0; i < counts.Length; i++)
            {
                if(counts[i] > maxCount)
                {
                    bestIndex = i;
                    maxCount = counts[i];
                }
            }

            return classes[bestIndex];
        }

        private static double CalcGiniIndex(LabeledData[] data)
        {
            List<int> classes = data.Select(x => x.Label).Distinct().ToList();

            double gini = 0.0;
            int N = data.Length;

            foreach(int c in classes)
            {
                int numMembers = data.Where(x => x.Label == c).Count();
                double prob = (double)numMembers / N;
                gini += (prob * prob);
            }

            return 1.0 - gini;
        }

        public int Predict(LabeledData d)
        {
            if(Root == null)
            {
                throw new Exception($"Exception in {nameof(Predict)}, Root node was null.");
            }

            return MakePrediction(d, Root);
        }

        public int Predict(double[] x)
        {
            if (Root == null)
            {
                throw new Exception($"Exception in {nameof(Predict)}, Root node was null.");
            }

            return MakePrediction(x, Root);
        }

        private int MakePrediction(double[] x, DecisionNode node)
        {
            if (node.IsLeaf) { return node.Value; }
            else if(x[node.Index] <= node.Threshold) 
            {
                if(node.LeftNode == null)
                {
                    throw new Exception($"Exception in {nameof(MakePrediction)}, DecisionNode in tree was null.");
                }
                return MakePrediction(x, node.LeftNode); 
            }
            else
            {
                if(node.RightNode == null)
                {
                    throw new Exception($"Exception in {nameof(MakePrediction)}, DecisionNode in tree was null.");
                }

                return MakePrediction(x, node.RightNode);
            }
        }

        private int MakePrediction(LabeledData d, DecisionNode node)
        {
            if (node.IsLeaf) { return node.Value; }
            else if (d[node.Index] <= node.Threshold)
            {
                if (node.LeftNode == null)
                {
                    throw new Exception($"Exception in {nameof(MakePrediction)}, DecisionNode in tree was null.");
                }
                return MakePrediction(d, node.LeftNode);
            }
            else
            {
                if (node.RightNode == null)
                {
                    throw new Exception($"Exception in {nameof(MakePrediction)}, DecisionNode in tree was null.");
                }

                return MakePrediction(d, node.RightNode);
            }
        }
    }
}
