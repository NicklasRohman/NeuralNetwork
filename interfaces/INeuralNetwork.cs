namespace NeuralNetwork.interfaces
{
    public interface INeuralNetwork
    {
        public void Execute(NeuralNetworkContext context);
        public void Execute_FullContext(NeuralNetworkContext context, NeuralNetworkFullContext fullContext);
        public void ExecuteBackwards(float[] target, NeuralNetworkContext context, NeuralNetworkFullContext fullContext, NeuralNetworkPropagationState propState, int lossType, int crossEntropyTarget);
        public void Breed(NeuralNetwork partner);
        public void Mutate(float selectionChance);
        public void RandomizeWeightsAndBiasesForAdagrad();
        public void RandomizeWeightsAndBiases();
        public void RandomizeWeightsAndBiases(float minBias, float maxBias, float minWeight, float maxWeight);
        public void CopyWeightsAndBiases(NeuralNetwork nn);
        public void SetupExecutionArrays(out float[] ina, out float[] outa, out float[] hiddena, out float[][] hiddenRecurra);
        public void SaveStructure(Stream s);
        public void Save(Stream s);
        public void Load(Stream s);
        public int TotalNumberOfNeurons();
        public int TotalNumberOfSynapses();
        public int NumberOfLayers();
        public NeuralNetworkLayer GetLayer(int i);
        public NeuralNetworkLayerConnection GetConnection(int i);
        public NeuralNetworkLayerConnection GetRecurringConnection(int i);
    }
}
