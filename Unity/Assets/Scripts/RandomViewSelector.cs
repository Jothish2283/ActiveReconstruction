// ============================================================
// RandomViewSelector.cs — uses MetricsLogger.ReadCSVLines()
// avoids sharing violation by letting MetricsLogger manage the file handle
// ============================================================

using UnityEngine;

namespace ReconstructionRL
{
    public class RandomViewSelector : MonoBehaviour
    {
        [Header("Dependencies")]
        public CameraAgent cameraAgent;
        public ViewPointManager viewPointManager;
        public MetricsLogger metricsLogger;

        [Header("Settings")]
        public int randomSeed = 42;
        public bool isActive = false;

        [Header("Episode Limit")]
        [Tooltip("Stop after this many episodes. 0 = unlimited.")]
        public int maxEpisodes = 50;

        // ── Private ──────────────────────────────────────────────────
        private System.Random rng;
        private int numViewpoints = 32;
        private bool finished = false;
        private bool pauseNextFrame = false;

        private void Awake()
        {
            if (cameraAgent == null)      cameraAgent      = FindObjectOfType<CameraAgent>();
            if (viewPointManager == null) viewPointManager = FindObjectOfType<ViewPointManager>();
            if (metricsLogger == null)    metricsLogger    = FindObjectOfType<MetricsLogger>();
            InitRNG();
        }

        private void Update()
        {
            // Delayed pause — one frame after summary prints so console keeps it
            if (pauseNextFrame)
            {
                pauseNextFrame = false;
#if UNITY_EDITOR
                UnityEditor.EditorApplication.isPaused = true;
#endif
                return;
            }

            if (!isActive || finished || cameraAgent == null) return;

            // Check episode count via public property — no CSV polling
            if (maxEpisodes > 0 && metricsLogger != null
                && metricsLogger.EpisodeCount >= maxEpisodes)
            {
                finished = true;
                isActive = false;
                PrintSummary();       // MetricsLogger closes+reads+reopens file internally
                pauseNextFrame = true;
                return;
            }

            if (viewPointManager != null)
                numViewpoints = viewPointManager.NumViewpoints;

            cameraAgent.heuristicAction = rng.Next(0, numViewpoints);
        }

        // ── Public API ───────────────────────────────────────────────

        public void Activate()
        {
            isActive  = true;
            finished  = false;
            pauseNextFrame = false;
            if (cameraAgent != null) cameraAgent.baselineOverride = true;
            var seq = FindObjectOfType<SequentialViewSelector>();
            if (seq != null) seq.Deactivate();
            Debug.Log("[RandomViewSelector] Activated. Max episodes: " + maxEpisodes);
        }

        public void Deactivate() => isActive = false;

        public void ResetSeed() => InitRNG();

        public int PickViewpoint()
        {
            if (viewPointManager != null) numViewpoints = viewPointManager.NumViewpoints;
            return rng.Next(0, numViewpoints);
        }

        // ── Private ──────────────────────────────────────────────────

        private void InitRNG()
        {
            rng = randomSeed >= 0 ? new System.Random(randomSeed) : new System.Random();
        }

        private void PrintSummary()
        {
            if (metricsLogger == null)
            {
                Debug.Log("[RandomViewSelector] No MetricsLogger assigned.");
                return;
            }

            // Use MetricsLogger.ReadCSVLines() — closes writer, reads, reopens
            // This is the ONLY safe way to read while MetricsLogger holds the file
            string[] lines = metricsLogger.ReadCSVLines();

            float r = 0, cov = 0, u = 0, acc = 0;
            int count = 0;

            for (int i = 1; i < lines.Length; i++)   // i=0 is header
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
                ? "══ RANDOM SUMMARY ══  No data rows found."
                : $"══ RANDOM SUMMARY ({count} eps) ══  " +
                  $"AvgReward={r/count:F4}  " +
                  $"AvgCoverage={cov/count:F4}  " +
                  $"AvgUncertainty={u/count:F4}  " +
                  $"AvgAccuracy={acc/count:F4}";

            Debug.Log(summary);

            // Write summary txt next to CSV — survives console clear
            try
            {
                string txtPath = metricsLogger.GetCSVPath().Replace("_metrics.csv", "_SUMMARY.txt");
                System.IO.File.WriteAllText(txtPath, summary + "\n");
                Debug.Log("[RandomViewSelector] Summary saved: " + txtPath);
            }
            catch (System.Exception e)
            {
                Debug.LogWarning("[RandomViewSelector] Could not save summary txt: " + e.Message);
            }
        }
    }
}