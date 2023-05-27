namespace NeuralNetwork.interfaces
{
    public interface INeuralNetworkGenerator
    {
        public float[][] InputErrorPropagationRecurring(float[][] inputData, float[][] targetData);
        public float[] InputErrorPropagation(float[] inputData, float[] targetData);
    }
}
