using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RandomForest
{
    internal class LabeledData
    {
        public int Label { get; private set; }

        public double this[int i]
        {
            get { return data[i]; }
        }

        public int Length => data.Length;

        private double[] data;

        public LabeledData(int label, double[] data)
        {
            this.data = data;
            Label = label;
        }
    }
}
