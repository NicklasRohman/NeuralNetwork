using NeuralNetwork.interfaces;

namespace NeuralNetwork
{
    public class QLearningContext : IQLearningContext
    {
        public int action;
        public float[] input;

        public QLearningContext(int a, float[] i)
        {
            action = a;
            input = new float[i.Length];
            Array.Copy(i, input, i.Length);
        }
    }
}
