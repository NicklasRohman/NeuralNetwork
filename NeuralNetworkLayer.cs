using NeuralNetwork.interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>
    /// Neural network layer.
    /// </summary>
    public class NeuralNetworkLayer : INeuralNetworkLayer
    {
            /// <summary>
            ///  Minimum bias when randomly generating biases.
            /// </summary>
            public static float MIN_BIAS = 0.0f;
            /// <summary>
            ///  Maximum bias when randomly generating biases.
            /// </summary>
            public static float MAX_BIAS = 0.0f;

            /// <summary>
            /// Number of neurons in layer.
            /// </summary>
            public int numberOfNeurons;
            /// <summary>
            /// Flag indicating whether or not neurons are recurring(last state is fed back in as input).
            /// </summary>
            public bool recurring;
            /// <summary>
            /// Array of biases.
            /// </summary>
            public float[] biases;
            /// <summary>
            /// Neuron activation function to use for all neurons in layer.
            /// </summary>
            public NeuralNetwork.NeuronActivationFunction activationFunction;

            /// <summary>
            /// Create new struct from existing.
            /// </summary>
            /// <param name="src">Existing layer to copy.</param>
            public NeuralNetworkLayer(NeuralNetworkLayer src)
            {
                numberOfNeurons = src.numberOfNeurons;
                recurring = src.recurring;
                activationFunction = src.activationFunction;
                biases = null;
            }
            /// <summary>
            /// Create new struct.
            /// </summary>
            /// <param name="numNeurons">Number of neurons in layer.</param>
            /// <param name="recurrin">Are neurons in layer recurring.</param>
            /// <param name="activeFunc">Neuron activation function.</param>
            public NeuralNetworkLayer(int numNeurons, bool recurrin, NeuralNetwork.NeuronActivationFunction activeFunc)
            {
                numberOfNeurons = numNeurons;
                recurring = recurrin;
                activationFunction = activeFunc;
                biases = null;
            }
            public NeuralNetworkLayer() { }

            /// <summary>
            /// Allocate biases float array.
            /// </summary>
            public void Init()
            {
                biases = new float[numberOfNeurons];
            }

            public void SaveStructure(Stream s)
            {
                Utils.IntToStream(numberOfNeurons, s);
                s.WriteByte(recurring ? (byte)1 : (byte)0);
                Utils.IntToStream(Utils.GetActivationFunctionID(activationFunction), s);
            }

            public void LoadStructure(Stream s)
            {
                numberOfNeurons = Utils.IntFromStream(s);
                recurring = s.ReadByte() == 1;
                activationFunction = Utils.GetActivationFunctionFromID(Utils.IntFromStream(s));
            }

            /// <summary>
            /// Save layer to stream.
            /// </summary>
            /// <param name="s">Stream.</param>
            public void Save(Stream s)
            {
                int i = numberOfNeurons;
                while (i-- > 0)
                {
                    s.Write(Utils.FloatToBytes(biases[i]), 0, 4);
                }
            }

            /// <summary>
            /// Load layer from stream.
            /// </summary>
            /// <param name="s">Stream.</param>
            public void Load(Stream s)
            {
                byte[] rbuf = new byte[4];

                int i = numberOfNeurons;
                while (i-- > 0)
                {
                    s.Read(rbuf, 0, 4);
                    biases[i] = Utils.FloatFromBytes(rbuf);
                }
            }

            /// <summary>
            /// Generate random biases for layer from MIN_BIAS to MAX_BIAS.
            /// </summary>
            public void RandomizeBiases()
            {
                int i = numberOfNeurons;
                while (i-- > 0)
                {
                    biases[i] = Utils.NextFloat01() * (MAX_BIAS - MIN_BIAS) + MIN_BIAS;
                }
            }

            /// <summary>
            /// Copy biases from layer.
            /// </summary>
            /// <param name="nnl">Layer.</param>
            public void CopyBiases(NeuralNetworkLayer nnl)
            {
                float[] cb = nnl.biases;

                int i = numberOfNeurons;
                while (i-- > 0)
                {
                    biases[i] = cb[i];
                }
            }



            /// <summary>
            /// Mutate a selection of biases randomly.
            /// </summary>
            /// <param name="selectionChance">The chance(0-1) of a bias being mutated.</param>
            public void Mutate(float selectionChance)
            {
                int i = numberOfNeurons;
                while (i-- > 0)
                {
                    if (Utils.NextFloat01() <= selectionChance)
                    {
                        biases[i] = Utils.NextFloat01() * (MAX_BIAS - MIN_BIAS) + MIN_BIAS;
                    }
                }
            }

            //breed data with partner, partner must have the same # neurons/synapses
            //takes a random selection of weights and biases from partner and replaces the a %(partPartner) of this classes weights/classes with those
            //partPartner is the % of weights and biases to use from the partner, 0 being none and 1 being all the weights/biases
            /// <summary>
            /// Breed with another layer(partner) taking a %(partPartner) of randomly selected biases.
            /// </summary>
            /// <param name="partner">Partner layer.</param>
            /// <param name="partPartner">Percent(0-1) of biases to take from partner.</param>
            public void Breed(NeuralNetworkLayer partner)
            {
                int i = numberOfNeurons;
                while (i-- > 0)
                {
                    //randomly mix
                    float val = Utils.NextFloat01();
                    biases[i] = biases[i] * val + partner.biases[i] * (1.0f - val);
                }
            }
        }
    }

