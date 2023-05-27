/*
 *The MIT License (MIT)
Copyright (c) 2017 Ethan Alexander Shulman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
 * 
 */

using NeuralNetwork.interfaces;

namespace NeuralNetwork
{
    //neural network class
    /// <summary>
    /// Neural network.
    /// </summary>
    public class NeuralNetwork : INeuralNetwork
    {

        public NeuralNetworkLayer inputLayer;
        public NeuralNetworkLayer outputLayer;
        public NeuralNetworkLayer[] hiddenLayers;

        public NeuralNetworkLayerConnection outputConnection;
        public NeuralNetworkLayerConnection[] hiddenConnections, hiddenRecurringConnections;

        public int maxNumberOfHiddenNeurons, maxNumberOfSynapses;

        /// <summary>
        /// Create new NeuralNetwork from existing(src).
        /// </summary>
        /// <param name="src">Existing NeuralNetwork to copy.</param>
        public NeuralNetwork(NeuralNetwork src)
        {
            inputLayer = new NeuralNetworkLayer(src.inputLayer);

            hiddenLayers = new NeuralNetworkLayer[src.hiddenLayers.Length];
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i] = new NeuralNetworkLayer(src.hiddenLayers[i]);
                hiddenLayers[i].Init();
            }
            outputLayer = new NeuralNetworkLayer(src.outputLayer);
            outputLayer.Init();

            //setup layer connections
            if (hiddenLayers.Length > 0)
            {
                //hidden layer connections
                hiddenConnections = new NeuralNetworkLayerConnection[hiddenLayers.Length];
                hiddenRecurringConnections = new NeuralNetworkLayerConnection[hiddenLayers.Length];
                maxNumberOfHiddenNeurons = 0;
                maxNumberOfSynapses = 0;
                for (int i = 0; i < hiddenLayers.Length; i++)
                {
                    if (i == 0) hiddenConnections[0] = new NeuralNetworkLayerConnection(inputLayer, hiddenLayers[0]);
                    else hiddenConnections[i] = new NeuralNetworkLayerConnection(hiddenLayers[i - 1], hiddenLayers[i]);
                    //recurrent connection for hidden layer
                    if (hiddenLayers[i].recurring)
                    {
                        hiddenRecurringConnections[i] = new NeuralNetworkLayerConnection(hiddenLayers[i], hiddenLayers[i]);
                    }
                    else
                    {
                        hiddenRecurringConnections[i] = null;
                    }
                    //calc max hidden neurons
                    if (hiddenLayers[i].numberOfNeurons > maxNumberOfHiddenNeurons)
                    {
                        maxNumberOfHiddenNeurons = hiddenLayers[i].numberOfNeurons;
                    }
                    if (hiddenConnections[i].numberOfSynapses > maxNumberOfSynapses) maxNumberOfSynapses = hiddenConnections[i].numberOfSynapses;
                }

                //output connection
                outputConnection = new NeuralNetworkLayerConnection(hiddenLayers[hiddenLayers.Length - 1], outputLayer);

                if (outputConnection.numberOfSynapses > maxNumberOfSynapses) maxNumberOfSynapses = outputConnection.numberOfSynapses;
            }
            else
            {
                maxNumberOfHiddenNeurons = 0;
                maxNumberOfSynapses = 0;

                //direct input to output connection
                outputConnection = new NeuralNetworkLayerConnection(inputLayer, outputLayer);
                if (outputConnection.numberOfSynapses > maxNumberOfSynapses) maxNumberOfSynapses = outputConnection.numberOfSynapses;

            }
        }
        /// <summary>
        /// Create new NeuralNetwork.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="hidden"></param>
        /// <param name="output"></param>
        public NeuralNetwork(NeuralNetworkLayer input, NeuralNetworkLayer[] hidden, NeuralNetworkLayer output)
        {

            inputLayer = input;
            hiddenLayers = hidden;
            outputLayer = output;
            outputLayer.Init();

            //setup layer connections
            if (hidden.Length > 0)
            {
                //hidden layer connections
                hiddenConnections = new NeuralNetworkLayerConnection[hidden.Length];
                hiddenRecurringConnections = new NeuralNetworkLayerConnection[hidden.Length];
                maxNumberOfHiddenNeurons = 0;
                maxNumberOfSynapses = 0;
                for (int i = 0; i < hidden.Length; i++)
                {
                    if (i == 0) hiddenConnections[0] = new NeuralNetworkLayerConnection(input, hidden[0]);
                    else hiddenConnections[i] = new NeuralNetworkLayerConnection(hidden[i - 1], hidden[i]);

                    hiddenLayers[i].Init();

                    //recurrent connection for hidden layer
                    if (hidden[i].recurring)
                    {
                        hiddenRecurringConnections[i] = new NeuralNetworkLayerConnection(hidden[i], hidden[i]);
                    }
                    else
                    {
                        hiddenRecurringConnections[i] = null;
                    }
                    //calc max hidden neurons
                    if (hidden[i].numberOfNeurons > maxNumberOfHiddenNeurons)
                    {
                        maxNumberOfHiddenNeurons = hidden[i].numberOfNeurons;
                    }
                    if (hiddenConnections[i].numberOfSynapses > maxNumberOfSynapses) maxNumberOfSynapses = hiddenConnections[i].numberOfSynapses;

                }

                //output connection
                outputConnection = new NeuralNetworkLayerConnection(hidden[hidden.Length - 1], output);
                if (outputConnection.numberOfSynapses > maxNumberOfSynapses) maxNumberOfSynapses = outputConnection.numberOfSynapses;
            }
            else
            {
                maxNumberOfHiddenNeurons = 0;
                maxNumberOfSynapses = 0;

                //direct input to output connection
                outputConnection = new NeuralNetworkLayerConnection(input, output);
                if (outputConnection.numberOfSynapses > maxNumberOfSynapses) maxNumberOfSynapses = outputConnection.numberOfSynapses;
            }
        }

        //execute neural network
        /// <summary>
        /// Execute NeuralNetwork.
        /// </summary>
        /// <param name="context">Execution memory.</param>
        public void Execute(NeuralNetworkContext context)
        {
            float[] input = context.inputData,
                    output = context.outputData,
                    hidden = context.hiddenData;
            float[][] hiddenRecurring = context.hiddenRecurringData;

            int i, weightIndex, recurringWeightIndex;


            NeuronActivationFunction activeFunc;
            if (hiddenLayers.Length > 0)
            {
                int lastNumNeurons = 0;
                float[] weights, biases, recurringWeights;
                for (i = 0; i < hiddenLayers.Length; i++)
                {

                    weights = hiddenConnections[i].weights;
                    biases = hiddenLayers[i].biases;

                    activeFunc = hiddenLayers[i].activationFunction;

                    float[] ina;
                    int alen;
                    if (i == 0)
                    {
                        ina = input;
                        alen = input.Length;
                    }
                    else
                    {
                        ina = hidden;
                        alen = lastNumNeurons;
                    }

                    if (hiddenLayers[i].recurring)
                    {
                        //recurring hidden layer
                        recurringWeights = hiddenRecurringConnections[i].weights;

                        weightIndex = 0;
                        recurringWeightIndex = 0;

                        float[] hrec = hiddenRecurring[i];

                        int k = biases.Length;
                        while (k-- > 0)
                        {
                            float ov = biases[k];

                            int j = alen;
                            while (j-- > 0)
                            {
                                ov += ina[j] * weights[weightIndex++];
                            }

                            j = hrec.Length;
                            while (j-- > 0)
                            {
                                ov += hrec[j] * recurringWeights[recurringWeightIndex++];
                            }

                            hidden[k] = activeFunc(ov);
                        }

                        Array.Copy(hidden, hrec, biases.Length);
                    }
                    else
                    {
                        //non recurring hidden layer
                        weightIndex = 0;

                        int k = biases.Length;
                        while (k-- > 0)
                        {
                            float ov = biases[k];

                            int j = alen;
                            while (j-- > 0)
                            {
                                ov += ina[j] * weights[weightIndex++];
                            }

                            hidden[k] = activeFunc(ov);
                        }
                    }

                    lastNumNeurons = biases.Length;
                }

                activeFunc = outputLayer.activationFunction;

                //last output layer

                //run input to output layer connection
                weights = outputConnection.weights;
                biases = outputLayer.biases;

                weightIndex = 0;
                recurringWeightIndex = 0;

                i = output.Length;
                while (i-- > 0)
                {
                    float ov = biases[i];

                    //input connections
                    int k = lastNumNeurons;
                    while (k-- > 0)
                    {
                        ov += hidden[k] * weights[weightIndex++];
                    }

                    output[i] = activeFunc(ov);
                }
            }
            else
            {
                activeFunc = outputLayer.activationFunction;

                //run input to output layer connection with recurring output
                float[] weights = outputConnection.weights,
                        biases = outputLayer.biases;

                weightIndex = 0;
                recurringWeightIndex = 0;
                i = output.Length;
                while (i-- > 0)
                {
                    float ov = biases[i];

                    //input connections
                    int k = input.Length;
                    while (k-- > 0)
                    {
                        ov += input[k] * weights[weightIndex++];
                    }

                    output[i] = activeFunc(ov);
                }
            }
        }

        //execute neural network and save all calculation results in fullContext for adagrad
        /// <summary>
        /// Execute neural network and save all calculation results in fullContext for adagrad
        /// </summary>
        /// <param name="input"></param>
        /// <param name="context"></param>
        /// <param name="fullContext"></param>
        public void Execute_FullContext(NeuralNetworkContext context, NeuralNetworkFullContext fullContext)
        {
            float[] input = context.inputData,
                    output = context.outputData,
                    hidden = context.hiddenData;

            float[][] hiddenRecurring = context.hiddenRecurringData;

            int i, weightIndex, recurringWeightIndex;


            NeuronActivationFunction activeFunc;
            if (hiddenLayers.Length > 0)
            {
                int lastNumNeurons = 0;
                float[] weights, biases, recurringWeights;
                for (i = 0; i < hiddenLayers.Length; i++)
                {
                    weights = hiddenConnections[i].weights;
                    biases = hiddenLayers[i].biases;

                    activeFunc = hiddenLayers[i].activationFunction;

                    float[] ina;
                    int alen;
                    if (i == 0)
                    {
                        ina = input;
                        alen = input.Length;
                    }
                    else
                    {
                        ina = hidden;
                        alen = lastNumNeurons;
                    }

                    if (hiddenLayers[i].recurring)
                    {
                        //recurring hidden layer
                        float[] hrec = hiddenRecurring[i];

                        recurringWeights = hiddenRecurringConnections[i].weights;

                        //copy over data needed for training
                        Array.Copy(hrec, fullContext.hiddenRecurringBuffer[i], hrec.Length);

                        weightIndex = 0;
                        recurringWeightIndex = 0;


                        int k = biases.Length;
                        while (k-- > 0)
                        {
                            float ov = biases[k];

                            int j = alen;
                            while (j-- > 0)
                            {
                                ov += ina[j] * weights[weightIndex++];
                            }

                            j = hrec.Length;
                            while (j-- > 0)
                            {
                                ov += hrec[j] * recurringWeights[recurringWeightIndex++];
                            }

                            hidden[k] = activeFunc(ov);
                        }

                        Array.Copy(hidden, hrec, biases.Length);
                    }
                    else
                    {
                        //non recurring hidden layer
                        weightIndex = 0;

                        int k = biases.Length;
                        while (k-- > 0)
                        {
                            float ov = biases[k];

                            int j = alen;
                            while (j-- > 0)
                            {
                                ov += ina[j] * weights[weightIndex++];
                            }

                            hidden[k] = activeFunc(ov);
                        }
                    }

                    Array.Copy(hidden, fullContext.hiddenBuffer[i], biases.Length);
                    lastNumNeurons = biases.Length;
                }

                activeFunc = outputLayer.activationFunction;


                //last output layer

                //run input to output layer connection
                weights = outputConnection.weights;
                biases = outputLayer.biases;

                weightIndex = 0;
                recurringWeightIndex = 0;

                i = output.Length;
                while (i-- > 0)
                {
                    float ov = biases[i];

                    //input connections
                    int k = lastNumNeurons;
                    while (k-- > 0)
                    {
                        ov += hidden[k] * weights[weightIndex++];
                    }

                    output[i] = activeFunc(ov);
                }
            }
            else
            {
                activeFunc = outputLayer.activationFunction;

                //run input to output layer connection with recurring output
                float[] weights = outputConnection.weights,
                        biases = outputLayer.biases;


                weightIndex = 0;
                recurringWeightIndex = 0;
                i = output.Length;
                while (i-- > 0)
                {
                    float ov = biases[i];

                    //input connections
                    int k = input.Length;
                    while (k-- > 0)
                    {
                        ov += input[k] * weights[weightIndex++];
                    }

                    output[i] = activeFunc(ov);
                }
            }
        }



        /// <summary>
        /// Run neural network backwards calculating derivatives to use for adagrad or generation.
        /// </summary>
        /// <param name="target"></param>
        /// <param name="context"></param>
        /// <param name="fullContext"></param>
        /// <param name="derivMem"></param>
        public void ExecuteBackwards(float[] target, NeuralNetworkContext context, NeuralNetworkFullContext fullContext, NeuralNetworkPropagationState propState, int lossType, int crossEntropyTarget)
        {
            //prepare for back propagation
            for (int i = 0; i < propState.state.Length; i++) {
                Utils.Fill(propState.state[i], 0.0f);
            }

            //back propagation + calculate max loss
            int lid = hiddenLayers.Length;

            float lossAvg = 0.0f;
            for (int i = 0; i < target.Length; i++)
            {
                float deriv = context.outputData[i] - target[i];

                if (lossType == NeuralNetworkTrainer.LOSS_TYPE_MAX)
                {
                    float aderiv = Math.Abs(deriv);
                    if (aderiv > lossAvg) lossAvg = aderiv;
                }
                else if (lossType == NeuralNetworkTrainer.LOSS_TYPE_AVERAGE)
                {
                    lossAvg += Math.Abs(deriv);
                }

                Backpropagate(lid, i, deriv, propState);
            }

            if (lossType == NeuralNetworkTrainer.LOSS_TYPE_AVERAGE)
            {
                lossAvg /= (float)target.Length;
            }
            else
            {
                if (lossType == NeuralNetworkTrainer.LOSS_TYPE_CROSSENTROPY && crossEntropyTarget != -1)
                {
                    lossAvg = (float)-Math.Log(context.outputData[crossEntropyTarget]);
                    if (float.IsInfinity(lossAvg))
                    {
                        lossAvg = 1e8f;
                    }
                }
            }

            propState.loss = lossAvg;
            propState.derivativeMemory.SwapBPBuffers();


            int k = lid;
            while (k-- > 0)
            {
                int l = hiddenLayers[k].numberOfNeurons;
                while (l-- > 0)
                {
                    Backpropagate(k, l, propState.state[k][l], propState);
                }
            }
        }

        private void Backpropagate(int level, int index, float deriv, NeuralNetworkPropagationState propState)
        {
            if (level < 0) return;

            int i, weightIndex;
            float[] b, m, w;


            //recurring weights
            if (level < propState.recurrWeightMems.Length && propState.recurrWeightMems[level] != null)
            {
                b = propState.recurrBuf[level];
                m = propState.recurrWeightMems[level];
                w = propState.recurrWeights[level];

                i = b.Length;
                weightIndex = w.Length - (index + 1) * i;
                float nhderiv = 0.0f;
                while (i-- > 0)
                {
                    m[weightIndex] += deriv * b[i];
                    nhderiv += deriv * w[weightIndex];
                    weightIndex++;
                }

#pragma warning disable 414,1718
                if (nhderiv != nhderiv || float.IsInfinity(nhderiv))
                {
                    nhderiv = 0.0f;
                }
#pragma warning restore 1718
                propState.derivativeMemory.altRecurringBPBuffer[level][index] = nhderiv;
            }


            float[] bpb = null;

            //biases and weights
            b = propState.buf[level];
            m = propState.weightMems[level];
            w = propState.weights[level];

            bpb = null;
            if (level != 0) bpb = propState.derivativeMemory.recurringBPBuffer[level - 1];

            propState.biasMems[level][index] += deriv;

            i = b.Length;
            weightIndex = w.Length - (index + 1) * i;
            while (i-- > 0)
            {
                float nderiv = b[i];
                m[weightIndex] += deriv * nderiv;
                if (level != 0)
                {
                    nderiv *= nderiv;

                    float bpropderiv = 0.0f;
                    if (bpb != null)
                    {
                        bpropderiv = bpb[i];
                    }

                    propState.state[level - 1][i] += (1.0f - nderiv) * (deriv * w[weightIndex] + bpropderiv);
                }
                else
                {
                    if (propState.inputMem != null)
                    {
                        nderiv *= nderiv;

                        float bpropderiv = 0.0f;
                        if (bpb != null)
                        {
                            bpropderiv = bpb[i];
                        }

                        nderiv = (1.0f - nderiv) * (deriv * w[weightIndex] + bpropderiv);
                        propState.inputMem[i] += nderiv;
                    }
                }
                weightIndex++;
            }
        }





        //breed data with partner
        //partPartner is a value from 0 - 1 indicating the % of data to take from partner, 0 being no data and 1 being 100% partner data
        /// <summary>
        /// Breed weights/biases with partner.
        /// </summary>
        /// <param name="partner"></param>
        public void Breed(NeuralNetwork partner)
        {
            outputLayer.Breed(partner.outputLayer);

            outputConnection.Breed(partner.outputConnection);

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].Breed(partner.hiddenLayers[i]);
                hiddenConnections[i].Breed(partner.hiddenConnections[i]);
                if (hiddenLayers[i].recurring) hiddenRecurringConnections[i].Breed(partner.hiddenRecurringConnections[i]);
            }
        }

        //mutate weights and biases
        /// <summary>
        /// Mutate weights and biases.
        /// </summary>
        /// <param name="selectionChance">Chance(0-1) of a weight/bias being mutated.</param>
        public void Mutate(float selectionChance)
        {
            outputLayer.Mutate(selectionChance);
            outputConnection.Mutate(selectionChance);

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].Mutate(selectionChance);
                hiddenConnections[i].Mutate(selectionChance);
                if (hiddenLayers[i].recurring) hiddenRecurringConnections[i].Mutate(selectionChance);
            }

        }


        /// <summary>
        /// Randomize weights and biases specifically for adagrad.
        /// </summary>
        public void RandomizeWeightsAndBiasesForAdagrad()
        {
            NeuralNetworkLayer.MIN_BIAS = 0.0f;
            NeuralNetworkLayer.MAX_BIAS = 0.0f;
            NeuralNetworkLayerConnection.MIN_WEIGHT = 0.0f;
            NeuralNetworkLayerConnection.MAX_WEIGHT = 1.0f / maxNumberOfHiddenNeurons;
            RandomizeWeightsAndBiases();
        }

        //randomize weights and biases of layers and connections
        /// <summary>
        /// Randomize all weights/biases.
        /// </summary>
        public void RandomizeWeightsAndBiases()
        {
            outputLayer.RandomizeBiases();
            outputConnection.RandomizeWeights();

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].RandomizeBiases();
                hiddenConnections[i].RandomizeWeights();
                if (hiddenLayers[i].recurring) hiddenRecurringConnections[i].RandomizeWeights();
            }

        }

        /// <summary>
        /// Randomize all weights and biases between specified min/max values.
        /// </summary>
        public void RandomizeWeightsAndBiases(float minBias, float maxBias, float minWeight, float maxWeight)
        {
            NeuralNetworkLayer.MIN_BIAS = minBias;
            NeuralNetworkLayer.MAX_BIAS = maxBias;
            NeuralNetworkLayerConnection.MIN_WEIGHT = minWeight;
            NeuralNetworkLayerConnection.MAX_WEIGHT = maxWeight;

            outputLayer.RandomizeBiases();
            outputConnection.RandomizeWeights();

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].RandomizeBiases();
                hiddenConnections[i].RandomizeWeights();
                if (hiddenLayers[i].recurring) hiddenRecurringConnections[i].RandomizeWeights();
            }

        }

        //copy weights and biases from another neural network
        /// <summary>
        /// Copy weights and biases from another NeuralNetwork(nn).
        /// </summary>
        /// <param name="nn"></param>
        public void CopyWeightsAndBiases(NeuralNetwork nn)
        {
            outputLayer.CopyBiases(nn.outputLayer);
            outputConnection.CopyWeights(nn.outputConnection);

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].CopyBiases(nn.hiddenLayers[i]);
                hiddenConnections[i].CopyWeights(nn.hiddenConnections[i]);
                if (hiddenLayers[i].recurring) hiddenRecurringConnections[i].CopyWeights(nn.hiddenRecurringConnections[i]);
            }

        }

        /// <summary>
        /// Setup data arrays for execution.
        /// </summary>
        /// <param name="ina"></param>
        /// <param name="outa"></param>
        /// <param name="hiddena"></param>
        /// <param name="hiddenRecurra"></param>
        public void SetupExecutionArrays(out float[] ina, out float[] outa, out float[] hiddena, out float[][] hiddenRecurra)
        {
            ina = new float[inputLayer.numberOfNeurons];
            outa = new float[outputLayer.numberOfNeurons];
            hiddena = new float[maxNumberOfHiddenNeurons];

            hiddenRecurra = new float[hiddenLayers.Length][];
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                if (hiddenLayers[i].recurring)
                {
                    hiddenRecurra[i] = new float[hiddenLayers[i].numberOfNeurons];
                }
            }
        }

        /// <summary>
        /// Save structure of NeuralNetwork layers to stream.
        /// </summary>
        /// <param name="s"></param>
        public void SaveStructure(Stream s)
        {
            inputLayer.SaveStructure(s);

            Utils.IntToStream(hiddenLayers.Length, s);
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].SaveStructure(s);
            }
            outputLayer.SaveStructure(s);
        }

        /// <summary>
        /// Load NeuralNetwork structure from stream.
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static NeuralNetwork LoadStructure(Stream s)
        {
            NeuralNetworkLayer inLayer = new NeuralNetworkLayer();
            inLayer.LoadStructure(s);

            NeuralNetworkLayer[] hidden = new NeuralNetworkLayer[Utils.IntFromStream(s)];
            for (int i = 0; i < hidden.Length; i++)
            {
                hidden[i] = new NeuralNetworkLayer();
                hidden[i].LoadStructure(s);
            }

            NeuralNetworkLayer outLayer = new NeuralNetworkLayer();
            outLayer.LoadStructure(s);

            return new NeuralNetwork(inLayer, hidden, outLayer);
        }

        //save data to stream
        /// <summary>
        /// Save NeuralNetwork data(weights/biases, no structure data like input/hidden/output) to stream.
        /// </summary>
        /// <param name="s"></param>
        public void Save(Stream s)
        {
            outputLayer.Save(s);

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].Save(s);
            }

            outputConnection.Save(s);

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenConnections[i].Save(s);
                if (hiddenLayers[i].recurring)
                {
                    hiddenRecurringConnections[i].Save(s);
                }
            }
        }

        //load data from stream
        /// <summary>
        /// Load NeuralNetwork from stream(s).
        /// </summary>
        /// <param name="s"></param>
        public void Load(Stream s)
        {
            outputLayer.Load(s);

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].Load(s);
            }

            outputConnection.Load(s);

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                hiddenConnections[i].Load(s);
                if (hiddenLayers[i].recurring)
                {
                    hiddenRecurringConnections[i].Load(s);
                }
            }
        }

        //get total number of neurons in network
        /// <summary>
        /// Get total number of neurons in network.
        /// </summary>
        /// <returns></returns>
        public int TotalNumberOfNeurons()
        {
            int nneurons = inputLayer.numberOfNeurons + outputLayer.numberOfNeurons;
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                nneurons += hiddenLayers[i].numberOfNeurons;
            }
            return nneurons;
        }

        //get total number of synapses in network
        /// <summary>
        /// Get total number of synapses in network.
        /// </summary>
        /// <returns></returns>
        public int TotalNumberOfSynapses()
        {
            int nsynapses = outputConnection.numberOfSynapses;
            if (hiddenConnections != null)
            {
                for (int i = 0; i < hiddenConnections.Length; i++)
                {
                    nsynapses += hiddenConnections[i].numberOfSynapses;
                    if (hiddenLayers[i].recurring) nsynapses += hiddenRecurringConnections[i].numberOfSynapses;
                }
            }
            return nsynapses;
        }


        public int NumberOfLayers()
        {
            return hiddenLayers.Length + 1;
        }

        public NeuralNetworkLayer GetLayer(int i)
        {
            if (i < hiddenLayers.Length) return hiddenLayers[i];
            return outputLayer;
        }
        public NeuralNetworkLayerConnection GetConnection(int i)
        {
            if (i < hiddenLayers.Length) return hiddenConnections[i];
            return outputConnection;
        }
        public NeuralNetworkLayerConnection GetRecurringConnection(int i)
        {
            if (i < hiddenLayers.Length) return hiddenRecurringConnections[i];
            return null;
        }

        public delegate float NeuronActivationFunction(float v);
    }
}