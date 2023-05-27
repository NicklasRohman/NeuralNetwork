using NeuralNetwork.interfaces;

namespace NeuralNetwork
{
    //structure for saving network run-time state memory
    /// <summary>
    /// Neural network execution memory.
    /// </summary>
    public class NeuralNetworkContext : INeuralNetworkFullContext
    {
        /// <summary>
        /// Input data.
        /// </summary>
        public float[] inputData;
        /// <summary>
        /// Output data.
        /// </summary>
        public float[] outputData;
        /// <summary>
        /// Hidden state data.
        /// </summary>
        public float[] hiddenData;
        /// <summary>
        /// Hidden recurring state data.
        /// </summary>
        public float[][] hiddenRecurringData;

        /// <summary>
        /// Allocate memory arrays.
        /// </summary>
        /// <param name="nn">Source network.</param>
        public void Setup(NeuralNetwork nn)
        {
            nn.SetupExecutionArrays(out inputData, out outputData, out hiddenData, out hiddenRecurringData);
            Reset(true);
        }

        /// <summary>
        /// Reset memory arrays.
        /// </summary>
        /// <param name="resetio">Should reset in/out arrays too?</param>
        public void Reset(bool resetio)
        {
            if (resetio)
            {
                Utils.Fill(outputData, 0.0f);
                Utils.Fill(inputData, 0.0f);
            }
            Utils.Fill(hiddenData, 0.0f);
            for (int i = 0; i < hiddenRecurringData.Length; i++)
            {
                if (hiddenRecurringData[i] != null) Utils.Fill(hiddenRecurringData[i], 0.0f);
            }
        }
    }
}
