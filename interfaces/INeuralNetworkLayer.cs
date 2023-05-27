namespace NeuralNetwork.interfaces
{
    public interface INeuralNetworkLayer
    {
        public void Init();
        public void SaveStructure(Stream s);
        public void LoadStructure(Stream s);
        public void Save(Stream s);
        public void Load(Stream s);
        public void RandomizeBiases();
        public void CopyBiases(NeuralNetworkLayer nnl);
        public void Mutate(float selectionChance);
        public void Breed(NeuralNetworkLayer partner);
    }
}
