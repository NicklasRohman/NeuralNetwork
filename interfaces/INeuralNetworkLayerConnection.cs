using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.interfaces
{
    public interface INeuralNetworkLayerConnection
    {
        public void RandomizeWeights();
        public void CopyWeights(NeuralNetworkLayerConnection nnc);
        public void Mutate(float selectionChance);
        public void Breed(NeuralNetworkLayerConnection partner);
        public void Save(Stream s);
        public void Load(Stream s);
    }
}
