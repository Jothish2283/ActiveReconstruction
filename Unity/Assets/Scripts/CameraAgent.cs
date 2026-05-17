// ============================================================
// CameraAgent.cs -- FINAL VERSION
// PPO agent + auto-summary at episode limit + inference mode support
// ============================================================

using System.IO;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

namespace ReconstructionRL
{
    public class CameraAgent : Agent
    {
        // ── Inspector ────────────────────────────────────────────────
        [Header("Dependencies")]
        public ReconstructionManager reconstructionManager;
        public ViewPointManager      viewPointManager;
        public CameraOrbit           cameraOrbit;
        public MetricsLogger         metricsLogger;

        [Header("Episode Settings")]
        [Range(8, 64)] public int maxStepsPerEpisode = 24;
        public float finalCoverageBonus = 0.5f;
        [Range(0f, 1f)] public float coverageThreshold = 0.75f;

        [Header("Reward Weights")]
        public float w1_uncertaintyReduction = 1.0f;
        public float w2_newVoxelBonus        = 0.005f;
        public float w3_revisitPenalty       = 0.3f;

        [Header("Inference / Evaluation")]
        [Tooltip("Stop after this many episodes during inference. 0 = run forever.")]
        public int maxEvalEpisodes = 24;
        [Tooltip("Set true automatically when Behavior Type = Inference Only.")]
        public bool isInferenceMode = false;

        [Header("Mode")]
        public bool baselineOverride = false;

        // ── Private State ────────────────────────────────────────────
        private int   currentStep;
        private float episodeTotalReward;
        private float episodeStartTime;
        private bool[] visitedViewpoints;
        private int    numViewpoints = 64;
        private bool   evalFinished  = false;
        private bool   pauseNextFrame = false;

        [HideInInspector] public int heuristicAction = 0;

        // Pre-allocated obs arrays
        private readonly float[] uncertaintyHistogram = new float[8];
        private float[] viewpointGains = new float[64];

        // ── ML-Agents ────────────────────────────────────────────────

        public override void Initialize()
        {
            numViewpoints    = 64;
            if (viewPointManager != null && viewPointManager.NumViewpoints > 0)
                numViewpoints = viewPointManager.NumViewpoints;

            visitedViewpoints = new bool[numViewpoints];
            viewpointGains    = new float[numViewpoints];

            // Fast training when Python trainer is connected
            if (Academy.Instance.IsCommunicatorOn)
                Time.timeScale = 10f;
        }

        public override void OnEpisodeBegin()
        {
            // Stop evaluation when limit reached (inference mode)
            if (pauseNextFrame)
            {
                pauseNextFrame = false;
#if UNITY_EDITOR
                UnityEditor.EditorApplication.isPaused = true;
#endif
                return;
            }

            currentStep        = 0;
            episodeTotalReward = 0f;
            episodeStartTime   = Time.time;
            heuristicAction    = 0;

            System.Array.Clear(visitedViewpoints, 0, visitedViewpoints.Length);

            if (reconstructionManager != null)
                reconstructionManager.ResetGrid();

            if (viewPointManager != null && viewPointManager.NumViewpoints > 0)
            {
                int startIdx  = Random.Range(0, numViewpoints);
                Vector3 start = viewPointManager.GetViewpointPosition(startIdx);
                transform.position = start;
                transform.LookAt(viewPointManager.GetSceneCenter());
                if (cameraOrbit != null)
                    cameraOrbit.SetPositionImmediate(start);
            }
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // Exactly 80 floats: 3+8+3+32+32+1+1
            if (reconstructionManager == null || viewPointManager == null)
            {
                for (int i = 0; i < 144; i++) sensor.AddObservation(0f);
                return;
            }

            float half = 10f;
            Vector3 pos = transform.position;
            sensor.AddObservation(Mathf.Clamp(pos.x / half, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(pos.y / half, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(pos.z / half, -1f, 1f));

            reconstructionManager.GetUncertaintyHistogram(uncertaintyHistogram, 8);
            for (int i = 0; i < 8; i++)
                sensor.AddObservation(uncertaintyHistogram[i]);

            Vector3 centroid = reconstructionManager.GetUncertaintyCentroid();
            sensor.AddObservation(Mathf.Clamp(centroid.x / half, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(centroid.y / half, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(centroid.z / half, -1f, 1f));

            viewPointManager.EstimateViewpointGains(viewpointGains, reconstructionManager);
            for (int i = 0; i < 64; i++)
                sensor.AddObservation(i < viewpointGains.Length ? viewpointGains[i] : 0f);

            for (int i = 0; i < 64; i++)
                sensor.AddObservation((i < visitedViewpoints.Length && visitedViewpoints[i]) ? 1f : 0f);

            sensor.AddObservation(1f - (float)currentStep / Mathf.Max(maxStepsPerEpisode, 1));
            sensor.AddObservation(reconstructionManager.GetMeanUncertainty());
            // Total: 3+8+3+32+32+1+1 = 80
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            if (actions.DiscreteActions.Length == 0) return;
            if (reconstructionManager == null || viewPointManager == null) return;

            int actionIdx = Mathf.Clamp(actions.DiscreteActions[0], 0, numViewpoints - 1);

            Vector3 targetPos = viewPointManager.GetViewpointPosition(actionIdx);
            transform.position = targetPos;
            transform.LookAt(viewPointManager.GetSceneCenter());
            if (cameraOrbit != null) cameraOrbit.SetTargetPosition(targetPos);

            int   prevVoxels = reconstructionManager.GetCoveredVoxelCount();
            float prevU      = reconstructionManager.GetMeanUncertainty();

            reconstructionManager.UpdateFromViewpoint(
                transform.position, transform.forward, transform.up);

            float currU      = reconstructionManager.GetMeanUncertainty();
            int   newVoxels  = reconstructionManager.GetCoveredVoxelCount() - prevVoxels;
            bool  revisited  = actionIdx < visitedViewpoints.Length && visitedViewpoints[actionIdx];

            float reward = w1_uncertaintyReduction * (prevU - currU)
                         + w2_newVoxelBonus        * newVoxels
                         - w3_revisitPenalty        * (revisited ? 1f : 0f);

            reward = Mathf.Clamp(reward, -1f, 2f);
            AddReward(reward);
            episodeTotalReward += reward;

            if (actionIdx < visitedViewpoints.Length)
                visitedViewpoints[actionIdx] = true;

            currentStep++;

            // Advance sequential baseline if active
            var seq = FindObjectOfType<SequentialViewSelector>();
            if (seq != null && seq.isActive) seq.AdvanceStep();

            if (currentStep >= maxStepsPerEpisode)
            {
                float coverage = reconstructionManager.GetVoxelCoverage();
                if (coverage >= coverageThreshold)
                    AddReward(finalCoverageBonus);

                if (metricsLogger != null)
                    metricsLogger.LogEpisode(
                        episodeTotalReward, coverage, currU,
                        reconstructionManager.GetReconstructionAccuracy(),
                        currentStep, Time.time - episodeStartTime);

                // Auto-stop and summary in inference/eval mode
                if (isInferenceMode && maxEvalEpisodes > 0
                    && metricsLogger != null
                    && metricsLogger.EpisodeCount >= maxEvalEpisodes
                    && !evalFinished)
                {
                    evalFinished = true;
                    PrintSummaryAndSave();
                    pauseNextFrame = true;
                }

                EndEpisode();
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var d = actionsOut.DiscreteActions;
            if (d.Length > 0)
                d[0] = Mathf.Clamp(heuristicAction, 0, numViewpoints - 1);
        }

        // ── Runtime key: press S to print summary at any time ────────

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.S) && !baselineOverride)
                PrintSummaryAndSave();
        }

        // ── Summary ──────────────────────────────────────────────────

        private void PrintSummaryAndSave()
        {
            if (metricsLogger == null)
            {
                Debug.Log("[CameraAgent] No MetricsLogger assigned.");
                return;
            }

            string[] lines = metricsLogger.ReadCSVLines();

            float r = 0, cov = 0, u = 0, acc = 0;
            int count = 0;

            for (int i = 1; i < lines.Length; i++)
            {
                string[] c = lines[i].Split(',');
                if (c.Length < 5) continue;
                float.TryParse(c[1], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float rv);
                float.TryParse(c[2], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float cv);
                float.TryParse(c[3], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float uv);
                float.TryParse(c[4], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float av);
                r += rv; cov += cv; u += uv; acc += av; count++;
            }

            string summary = count == 0
                ? "== RL SUMMARY == No data rows found."
                : $"== RL SUMMARY ({count} eps) ==  " +
                  $"AvgReward={r/count:F4}  " +
                  $"AvgCoverage={cov/count:F4}  " +
                  $"AvgUncertainty={u/count:F4}  " +
                  $"AvgAccuracy={acc/count:F4}";

            Debug.Log(summary);

            try
            {
                string txtPath = metricsLogger.GetCSVPath()
                    .Replace("_metrics.csv", "_SUMMARY.txt");
                File.WriteAllText(txtPath, summary + "\n");
                Debug.Log("[CameraAgent] RL summary saved: " + txtPath);
            }
            catch (System.Exception e)
            {
                Debug.LogWarning("[CameraAgent] Could not save summary: " + e.Message);
            }
        }
    }
}