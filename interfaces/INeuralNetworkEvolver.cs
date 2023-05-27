namespace NeuralNetwork.interfaces
{
    public interface INeuralNetworkEvolver
    {
        public void Record(NeuralNetwork nn, float loss);
        public NeuralNetwork NextGeneration();
        public NeuralNetwork NextGeneration(NeuralNetwork nn, float loss);
        public void Stop();
        public void Start();
        public void Reset();
        public void Save(Stream s);
        public void Load(Stream s);
    }
}
