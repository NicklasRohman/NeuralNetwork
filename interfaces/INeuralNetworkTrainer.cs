namespace NeuralNetwork.interfaces
{
    public interface INeuralNetworkTrainer
    {
        public void StartInit();
        public void Start();
        public void Stop();
        public bool Join(int timeout);
        public float GetLoss();
        public float GetSmoothLoss();
        public long GetIterations();
        public void Learn();
        public bool Running();
        public float GetLossDelta();
    }
}
