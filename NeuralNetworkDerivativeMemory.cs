namespace NeuralNetwork
{
    /// <summary>
    /// Derivative memory.
    /// </summary>
    public class NeuralNetworkDerivativeMemory
    {
        public float[][] weightMems, biasMems, recurrWeightMems, outputFullConnectedWeightMems, recurringBPBuffer, altRecurringBPBuffer;

        public void Setup(NeuralNetwork nn)
        {
            biasMems = new float[nn.hiddenLayers.Length + 1][];
            weightMems = new float[nn.hiddenLayers.Length + 1][];
            recurrWeightMems = new float[nn.hiddenLayers.Length][];
            recurringBPBuffer = new float[nn.hiddenLayers.Length][];
            altRecurringBPBuffer = new float[nn.hiddenLayers.Length][];

            for (int i = 0; i < nn.hiddenLayers.Length; i++)
            {
                weightMems[i] = new float[nn.hiddenConnections[i].numberOfSynapses];
                biasMems[i] = new float[nn.hiddenLayers[i].numberOfNeurons];

                if (nn.hiddenLayers[i].recurring)
                {
                    recurrWeightMems[i] = new float[nn.hiddenRecurringConnections[i].numberOfSynapses];
                    recurringBPBuffer[i] = new float[nn.hiddenLayers[i].numberOfNeurons];
                    altRecurringBPBuffer[i] = new float[nn.hiddenLayers[i].numberOfNeurons];
                }
            }

            int lid = nn.hiddenLayers.Length;
            biasMems[lid] = new float[nn.outputLayer.numberOfNeurons];
            weightMems[lid] = new float[nn.outputConnection.numberOfSynapses];
        }

        public void SwapBPBuffers()
        {
            float[][] temp = recurringBPBuffer;
            recurringBPBuffer = altRecurringBPBuffer;
            altRecurringBPBuffer = temp;
        }

        public void Reset()
        {
            for (int i = 0; i < biasMems.Length; i++)
            {
                Utils.Fill(biasMems[i], 0.0f);
                Utils.Fill(weightMems[i], 0.0f);
                if (i < recurrWeightMems.Length && recurrWeightMems[i] != null)
                {
                    Utils.Fill(recurrWeightMems[i], 0.0f);
                    Utils.Fill(recurringBPBuffer[i], 0.0f);
                    Utils.Fill(altRecurringBPBuffer[i], 0.0f);
                }
                if (outputFullConnectedWeightMems != null && i < outputFullConnectedWeightMems.Length) Utils.Fill(outputFullConnectedWeightMems[i], 0.0f);
            }
        }

        public void Scale(float s)
        {
            for (int i = 0; i < biasMems.Length; i++)
            {
                Utils.Multiply(biasMems[i], s);
                Utils.Multiply(weightMems[i], s);
                if (i < recurrWeightMems.Length && recurrWeightMems[i] != null)
                {
                    Utils.Multiply(recurrWeightMems[i], s);
                }
                if (outputFullConnectedWeightMems != null && i < outputFullConnectedWeightMems.Length) Utils.Multiply(outputFullConnectedWeightMems[i], s);
            }
        }

        public void ResetOnlyBuffer()
        {
            for (int i = 0; i < biasMems.Length; i++)
            {
                if (i < recurrWeightMems.Length && recurrWeightMems[i] != null)
                {
                    Utils.Fill(recurringBPBuffer[i], 0.0f);
                    Utils.Fill(altRecurringBPBuffer[i], 0.0f);
                }
            }
        }
    }
}
