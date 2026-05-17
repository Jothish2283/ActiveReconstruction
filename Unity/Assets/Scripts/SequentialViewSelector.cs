// ============================================================
// SequentialViewSelector.cs — uses MetricsLogger.ReadCSVLines()
// ============================================================

using UnityEngine;

namespace ReconstructionRL
{
    public class SequentialViewSelector : MonoBehaviour
    {
        [Header("Dependencies")]
        public CameraAgent cameraAgent;
        public ViewPointManager viewPointManager;
        public MetricsLogger metricsLogger;

        [Header("Settings")]
        public bool isActive = false;
        public bool wrapAround = true;

        [Header("Episode Limit")]
        [Tooltip("Stop after this many episodes. 0 = unlimited.")]
        public int maxEpisodes = 50;

        // ── Private ──────────────────────────────────────────────────
        private int currentIndex = 0;
        private int numViewpoints = 32;
        private bool finished = false;
        private bool pauseNextFrame = false;

        private void Awake()
        {
            if (cameraAgent == null)      cameraAgent      = FindObjectOfType<CameraAgent>();
            if (viewPointManager == null) viewPointManager = FindObjectOfType<ViewPointManager>();
            if (metricsLogger == null)    metricsLogger    = FindObjectOfType<MetricsLogger>();
        }

        private void Update()
        {
            if (pauseNextFrame)
            {
                pauseNextFrame = false;
#if UNITY_EDITOR
                UnityEditor.EditorApplication.isPaused = true;
#endif
                return;
            }

            if (!isActive || finished || cameraAgent == null) return;

            if (maxEpisodes > 0 && metricsLogger != null
                && metricsLogger.EpisodeCount >= maxEpisodes)
            {
                finished = true;
                isActive = false;
                PrintSummary();
                pauseNextFrame = true;
                return;
            }

            if (viewPointManager != null)
                numViewpoints = viewPointManager.NumViewpoints;

            cameraAgent.heuristicAction = Mathf.Clamp(currentIndex, 0, numViewpoints - 1);
        }

        // ── Public API ───────────────────────────────────────────────

        public void AdvanceStep()
        {
            currentIndex++;
            if (wrapAround)
                currentIndex %= Mathf.Max(numViewpoints, 1);
            else
                currentIndex = Mathf.Min(currentIndex, numViewpoints - 1);
        }

        public void ResetSequence() => currentIndex = 0;

        public void Activate()
        {
            isActive = true;
            finished = false;
            pauseNextFrame = false;
            currentIndex = 0;
            if (cameraAgent != null) cameraAgent.baselineOverride = true;
            var rnd = FindObjectOfType<RandomViewSelector>();
            if (rnd != null) rnd.Deactivate();
            Debug.Log("[SequentialViewSelector] Activated. Max episodes: " + maxEpisodes);
        }

        public void Deactivate() => isActive = false;

        public int CurrentIndex => currentIndex;

        // ── Private ──────────────────────────────────────────────────

        private void PrintSummary()
        {
            if (metricsLogger == null)
            {
                Debug.Log("[SequentialViewSelector] No MetricsLogger assigned.");
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
                ? "══ SEQUENTIAL SUMMARY ══  No data rows found."
                : $"══ SEQUENTIAL SUMMARY ({count} eps) ══  " +
                  $"AvgReward={r/count:F4}  " +
                  $"AvgCoverage={cov/count:F4}  " +
                  $"AvgUncertainty={u/count:F4}  " +
                  $"AvgAccuracy={acc/count:F4}";

            Debug.Log(summary);

            try
            {
                string txtPath = metricsLogger.GetCSVPath().Replace("_metrics.csv", "_SUMMARY.txt");
                System.IO.File.WriteAllText(txtPath, summary + "\n");
                Debug.Log("[SequentialViewSelector] Summary saved: " + txtPath);
            }
            catch (System.Exception e)
            {
                Debug.LogWarning("[SequentialViewSelector] Could not save summary txt: " + e.Message);
            }
        }
    }
}