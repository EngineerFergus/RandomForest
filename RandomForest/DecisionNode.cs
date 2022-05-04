using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RandomForest
{
    internal class DecisionNode
    {
        public int Index { get; set; }
        public double Threshold { get; set; }
        public double InfoGain { get; set; }
        public DecisionNode? LeftNode { get; set; }
        public DecisionNode? RightNode { get; set; }
        public int Value { get; set; }
        public bool IsLeaf { get; set; }
    }
}
