using NeuralNetwork.interfaces;

namespace NeuralNetwork
{
    /// <summary>
    /// Neural network layer connection.
    /// </summary>
    public class NeuralNetworkLayerConnection : INeuralNetworkLayerConnection
    {
        /// <summary>
        /// Minimum weight value when randomly generating weights.
        /// </summary>
        public static float MIN_WEIGHT = 0.0f;
        /// <summary>
        /// Maximum weight value when randomly generating weights.
        /// </summary>
        public static float MAX_WEIGHT = 0.06f;

        /// <summary>
        /// Number of synapses(neuron connections).
        /// </summary>
        public int numberOfSynapses;
        /// <summary>
        /// Array of synapse weights.
        /// </summary>
        public float[] weights;

        /// <summary>
        /// Create new struct from existing.
        /// </summary>
        /// <param name="src">Existing to copy.</param>
        public NeuralNetworkLayerConnection(NeuralNetworkLayerConnection src)
        {
            numberOfSynapses = src.numberOfSynapses;
            weights = new float[numberOfSynapses];
        }
        /// <summary>
        /// Create new struct.
        /// </summary>
        /// <param name="inLayer">Input layer.</param>
        /// <param name="outLayer">Output layer.</param>
        public NeuralNetworkLayerConnection(NeuralNetworkLayer inLayer, NeuralNetworkLayer outLayer)
        {
            numberOfSynapses = inLayer.numberOfNeurons * outLayer.numberOfNeurons;

            weights = new float[numberOfSynapses];
        }

        /// <summary>
        /// Randomly generate weights.
        /// </summary>
        public void RandomizeWeights()
        {
            int i = numberOfSynapses;
            while (i-- > 0)
            {
                weights[i] = Utils.NextFloat01() * (MAX_WEIGHT - MIN_WEIGHT) + MIN_WEIGHT;
            }
        }

        /// <summary>
        /// Copy weights from another connection struct.
        /// </summary>
        /// <param name="nnc">Source to copy from.</param>
        public void CopyWeights(NeuralNetworkLayerConnection nnc)
        {
            float[] cb = nnc.weights;

            int i = numberOfSynapses;
            while (i-- > 0)
            {
                weights[i] = cb[i];
            }
        }

        /// <summary>
        /// Mutates a selection of weights randomly.
        /// </summary>
        /// <param name="selectionChance">The chance(0-1) of a weight being mutated.</param>
        public void Mutate(float selectionChance)
        {
            int i = numberOfSynapses;
            while (i-- > 0)
            {
                if (Utils.NextFloat01() <= selectionChance)
                {
                    weights[i] = Utils.NextFloat01() * (MAX_WEIGHT - MIN_WEIGHT) + MIN_WEIGHT;
                }
            }
        }

        //breed with another layer connection data class(partner), partner must have the same # of synapses
        //takes a random selection of weights and biases from partner and replaces the a %(partPartner) of this classes weights/classes with those
        //partPartner is the % of weights and biases to use from the partner, 0 being none and 1 being all the weights/biases
        /// <summary>
        /// Breed with another connection(partner) taking a %(partPartner) of randomly selected weights.
        /// </summary>
        /// <param name="partner">Partner layer.</param>
        /// <param name="partPartner">Percent(0-1) of weights to take from partner.</param>
        public void Breed(NeuralNetworkLayerConnection partner)
        {
            int i = numberOfSynapses;
            while (i-- > 0)
            {
                //randomly mix
                float val = Utils.NextFloat01();
                weights[i] = weights[i] * val + partner.weights[i] * (1.0f - val);
            }
        }

        /// <summary>
        /// Save connection to stream.
        /// </summary>
        /// <param name="s">Stream.</param>
        public void Save(Stream s)
        {
            int i = numberOfSynapses;
            while (i-- > 0)
            {
                s.Write(Utils.FloatToBytes(weights[i]), 0, 4);
            }
        }

        //load data from stream
        /// <summary>
        /// Load connection from stream.
        /// </summary>
        /// <param name="s">Stream.</param>
        public void Load(Stream s)
        {
            byte[] buf = new byte[4];

            int i = numberOfSynapses;
            while (i-- > 0)
            {
                s.Read(buf, 0, 4);
                weights[i] = Utils.FloatFromBytes(buf);
            }
        }
    }
}
