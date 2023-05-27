using NeuralNetwork.interfaces;

namespace NeuralNetwork
{
    /// <summary>
    /// Full execution memory(needed for training).
    /// </summary>
    public class NeuralNetworkFullContext : INeuralNetworkFullContext
    {
        public float[][] hiddenBuffer, hiddenRecurringBuffer;

        public void Setup(NeuralNetwork nn)
        {
            hiddenBuffer = new float[nn.hiddenLayers.Length][];
            hiddenRecurringBuffer = new float[nn.hiddenLayers.Length][];
            for (int i = 0; i < nn.hiddenLayers.Length; i++)
            {
                hiddenBuffer[i] = new float[nn.hiddenLayers[i].numberOfNeurons];
                if (nn.hiddenLayers[i].recurring)
                {
                    hiddenRecurringBuffer[i] = new float[nn.hiddenLayers[i].numberOfNeurons];
                }
            }
        }
    }
}
