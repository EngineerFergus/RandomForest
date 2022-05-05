using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace RandomForest
{
    internal class Util
    {
        public static LabeledData[] ReadData(string dir)
        {
            Dictionary<string, int> labelDict = new Dictionary<string, int>();
            List<LabeledData> dataList = new List<LabeledData>();

            using(FileStream stream = File.OpenRead(dir))
            {
                using (StreamReader reader = new StreamReader(stream))
                {
                    int count = 0;

                    while (!reader.EndOfStream)
                    {
                        string? line = reader.ReadLine();
                        if(count == 0 || line is null) 
                        {
                            count++;
                            continue;
                        }

                        string[] splits = line.Split(',');
                        int numCols = splits.Length;

                        if(!labelDict.ContainsKey(splits[numCols - 1]))
                        {
                            labelDict.Add(splits[numCols - 1], labelDict.Count);
                        }

                        double[] features = new double[numCols - 1];
                        for(int i = 0; i < numCols - 1; i++)
                        {
                            _ = double.TryParse(splits[i], out features[i]);
                        }

                        dataList.Add(new LabeledData(labelDict[splits[numCols - 1]], features));

                        count++;
                    }
                }
            }

            return dataList.ToArray();
        }

        public static LabeledData[] BalanceClasses(LabeledData[] data)
        {
            List<int> classes = data.Select(x => x.Label).Distinct().ToList();
            List<int> counts = new List<int>();

            foreach(int c in classes)
            {
                counts.Add(data.Where(x => x.Label == c).Count());
            }

            int minCount = counts.Min();

            Dictionary<int, int> classCounts = new Dictionary<int, int>();

            foreach(int c in classes)
            {
                classCounts.Add(c, 0);
            }

            List<LabeledData> balancedData = new List<LabeledData>();

            for(int i = 0; i < data.Length; i++)
            {
                if(classCounts[data[i].Label] < minCount)
                {
                    balancedData.Add(data[i]);
                    classCounts[data[i].Label]++;
                }
            }

            return balancedData.ToArray();
        }

        public static (LabeledData[] train, LabeledData[] test) Split(LabeledData[] data, double trainAmount = 0.8)
        {
            int numTrain = (int)(data.Length * trainAmount);
            Random random = new Random();

            List<LabeledData> shuffled = data.OrderBy(x => random.Next()).ToList();
            List<LabeledData> train = new List<LabeledData>();
            List<LabeledData> test = new List<LabeledData>();

            for(int i = 0; i < shuffled.Count; i++)
            {
                if(i < numTrain)
                {
                    train.Add(shuffled[i]);
                }
                else
                {
                    test.Add(shuffled[i]);
                }
            }

            return (train.ToArray(), test.ToArray());
        }
    }
}
