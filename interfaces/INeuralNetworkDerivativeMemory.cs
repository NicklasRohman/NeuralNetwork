using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.interfaces
{
    public interface INeuralNetworkDerivativeMemory
    {
        public void Setup(NeuralNetwork nn);
        public void SwapBPBuffers();
        public void Reset();
        public void Scale(float s);
        public void ResetOnlyBuffer();
    }
}
