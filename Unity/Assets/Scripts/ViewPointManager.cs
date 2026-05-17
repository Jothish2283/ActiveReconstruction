// ============================================================
// ViewPointManager.cs
// Manages valid viewpoints, tracks visited state, provides
// nearest-view selection and uncertainty gain estimation.
// ============================================================

using UnityEngine;

namespace ReconstructionRL
{
    /// <summary>
    /// Central viewpoint registry. Wraps ViewPointGenerator output,
    /// tracks which viewpoints have been visited, and exposes
    /// per-viewpoint uncertainty gain estimates to the agent.
    /// </summary>
    public class ViewPointManager : MonoBehaviour
    {
        // ── Inspector Variables ──────────────────────────────────────
        [Header("Dependencies")]
        [Tooltip("Generator that creates Fibonacci sphere viewpoints.")]
        public ViewPointGenerator viewPointGenerator;

        [Tooltip("World-space center that camera always looks at.")]
        public Vector3 sceneCenter = Vector3.zero;

        [Header("Settings")]
        [Tooltip("If true, regenerate viewpoints from generator each episode reset.")]
        public bool regenerateEachEpisode = false;

        // ── Private State ────────────────────────────────────────────
        private Vector3[] viewpoints;
        private bool[] visited;
        private int numViewpoints;

        // Reusable gain array (no GC per frame)
        private float[] cachedGains;

        // ── Unity Lifecycle ──────────────────────────────────────────

        private void Awake()
        {
            if (viewPointGenerator == null)
            {
                viewPointGenerator = GetComponentInChildren<ViewPointGenerator>();
                if (viewPointGenerator == null)
                    Debug.LogError("[ViewPointManager] ViewPointGenerator not assigned!");
            }

            LoadViewpoints();
        }

        // ── Public API ───────────────────────────────────────────────

        /// <summary>
        /// Total number of available viewpoints.
        /// </summary>
        public int NumViewpoints => numViewpoints;

        /// <summary>
        /// World-space position of viewpoint at index.
        /// </summary>
        public Vector3 GetViewpointPosition(int index)
        {
            if (viewpoints == null || viewpoints.Length == 0) return Vector3.zero;
            index = Mathf.Clamp(index, 0, numViewpoints - 1);
            return viewpoints[index];
        }

        /// <summary>
        /// Whether viewpoint index was visited this episode.
        /// </summary>
        public bool IsVisited(int index)
        {
            if (visited == null) return false;
            index = Mathf.Clamp(index, 0, numViewpoints - 1);
            return visited[index];
        }

        /// <summary>
        /// Mark viewpoint as visited.
        /// </summary>
        public void MarkVisited(int index)
        {
            if (visited == null) return;
            index = Mathf.Clamp(index, 0, numViewpoints - 1);
            visited[index] = true;
        }

        /// <summary>
        /// Reset all visited flags for new episode.
        /// </summary>
        public void ResetVisited()
        {
            if (visited == null) return;
            System.Array.Clear(visited, 0, visited.Length);
            if (regenerateEachEpisode)
                LoadViewpoints();
        }

        /// <summary>
        /// Find index of viewpoint closest to given world position.
        /// </summary>
        public int GetNearestViewpointIndex(Vector3 worldPos)
        {
            if (viewpoints == null) return 0;
            int best = 0;
            float bestDist = float.MaxValue;
            for (int i = 0; i < numViewpoints; i++)
            {
                float d = Vector3.SqrMagnitude(viewpoints[i] - worldPos);
                if (d < bestDist) { bestDist = d; best = i; }
            }
            return best;
        }

        /// <summary>
        /// Get all viewpoint positions as array reference (read-only).
        /// </summary>
        public Vector3[] GetAllViewpoints() => viewpoints;

        /// <summary>
        /// World center that camera always looks toward.
        /// </summary>
        public Vector3 GetSceneCenter() => sceneCenter;

        /// <summary>
        /// Estimate per-viewpoint uncertainty gain.
        /// Uses centroid proximity heuristic from ReconstructionManager.
        /// Writes into preallocated gains array (no GC).
        /// </summary>
        public void EstimateViewpointGains(float[] gains, ReconstructionManager reconManager)
        {
            if (gains == null || reconManager == null || viewpoints == null) return;

            // Get uncertainty centroid from reconstruction
            Vector3 centroid = reconManager.GetUncertaintyCentroid();
            float meanU = reconManager.GetMeanUncertainty();

            int count = Mathf.Min(gains.Length, numViewpoints);
            float maxGain = 0f;

            for (int i = 0; i < count; i++)
            {
                float dist = Vector3.Distance(viewpoints[i], centroid);
                // Inverse-distance weighted by global uncertainty
                float rawGain = meanU / (1f + dist * 0.5f);
                // Reduce gain for visited viewpoints (discourage revisit)
                if (visited != null && visited[i])
                    rawGain *= 0.2f;
                gains[i] = rawGain;
                if (rawGain > maxGain) maxGain = rawGain;
            }

            // Normalize to [0,1]
            if (maxGain > 1e-6f)
                for (int i = 0; i < count; i++)
                    gains[i] /= maxGain;
        }

        /// <summary>
        /// Get index of unvisited viewpoint nearest to world position.
        /// Falls back to global nearest if all visited.
        /// </summary>
        public int GetNearestUnvisitedIndex(Vector3 worldPos)
        {
            if (viewpoints == null) return 0;
            int best = -1;
            float bestDist = float.MaxValue;
            for (int i = 0; i < numViewpoints; i++)
            {
                if (visited != null && visited[i]) continue;
                float d = Vector3.SqrMagnitude(viewpoints[i] - worldPos);
                if (d < bestDist) { bestDist = d; best = i; }
            }
            return best >= 0 ? best : GetNearestViewpointIndex(worldPos);
        }

        // ── Private Helpers ──────────────────────────────────────────

        private void LoadViewpoints()
        {
            if (viewPointGenerator == null) { numViewpoints = 0; return; }

            viewpoints = viewPointGenerator.GetViewpoints();
            numViewpoints = viewpoints != null ? viewpoints.Length : 0;
            visited = new bool[numViewpoints];
            cachedGains = new float[numViewpoints];

            Debug.Log($"[ViewPointManager] Loaded {numViewpoints} viewpoints.");
        }
    }
}
