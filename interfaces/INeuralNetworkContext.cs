namespace NeuralNetwork.interfaces
{
    public interface INeuralNetworkContext
    {
        public void Setup(NeuralNetwork nn);
        public void Reset(bool resetio);
    }
}
