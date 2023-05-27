namespace NeuralNetwork.interfaces
{
    public interface INeuralNetworkQLearning
    {
        public void Start();
        public void RestartSession();
        public void ClearSession();
        public void ClearAllSessions();
        public void Save(Stream s);
        public void Load(Stream s);
        public int Execute();
        public void Reward(float reward);
        public void Learn(float learningRate, int iter);
        public NeuralNetworkContext GetNeuralNetworkContext();
    }
}
