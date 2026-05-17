// ============================================================
// ReconstructionManager.cs — FIXED
// Voxel grid, raycasting, uncertainty, ground truth.
// Uses Ignore Raycast layermask so voxel cubes never interfere.
// Uncertainty computed over occupied voxels only.
// ============================================================

using UnityEngine;

namespace ReconstructionRL
{
    public class ReconstructionManager : MonoBehaviour
    {
        [Header("Voxel Grid Settings")]
        [Range(8, 64)] public int voxelSize = 32;
        public Vector3 sceneBoundsCenter = Vector3.zero;
        public Vector3 sceneBoundsSize   = new Vector3(5f, 5f, 5f);

        [Header("Raycast Settings")]
        [Range(16, 512)] public int raysPerStep = 128;
        public float maxRayDistance = 15f;
        [Range(0.01f, 0.9f)] public float confidenceDelta = 0.45f;
        public string targetTag = "Reconstructable";

        [Header("Visualization")]
        public VoxelColorAnimator voxelAnimator;
        [Range(1, 20)] public int visualizationUpdateInterval = 1;

        // ── Private State ────────────────────────────────────────────
        private float[,,] confidence;
        private bool[,,]  groundTruthOccupancy;
        private int       totalOccupiedVoxels;
        private int       coveredVoxelCount;
        private int       stepCounter;
        private Bounds    sceneBounds;

        // Layermask: exclude Ignore Raycast layer so voxel cubes never block rays
        private int raycastMask;

        // ── Unity Lifecycle ──────────────────────────────────────────

        private void Awake()
        {
            InitializeGrid();
            raycastMask = ~LayerMask.GetMask("Ignore Raycast");
        }

        private void Start()
        {
            // Start() ensures all Awake()s (including VoxelColorAnimator.Awake) have run
            ComputeGroundTruth();

            // Tell animator which voxels are occupied so it shows only those
            if (voxelAnimator != null)
                voxelAnimator.InitOccupancy(groundTruthOccupancy, voxelSize);
        }

        // ── Public API ───────────────────────────────────────────────

        public void ResetGrid()
        {
            confidence        = new float[voxelSize, voxelSize, voxelSize];
            coveredVoxelCount = 0;
            stepCounter       = 0;

            if (voxelAnimator != null)
                voxelAnimator.ResetColors();
        }

        public void UpdateFromViewpoint(Vector3 cameraPos, Vector3 forward, Vector3 up)
        {
            if (confidence == null) InitializeGrid();

            Vector3 right = Vector3.Cross(forward, up).normalized;
            float fovH = 60f * Mathf.Deg2Rad;
            float fovV = 45f * Mathf.Deg2Rad;

            int sqrtRays = Mathf.CeilToInt(Mathf.Sqrt(raysPerStep));

            for (int i = 0; i < raysPerStep; i++)
            {
                float u = (i % sqrtRays + Random.value) / sqrtRays;
                float v = (i / sqrtRays + Random.value) / sqrtRays;

                float angleH = (u - 0.5f) * fovH;
                float angleV = (v - 0.5f) * fovV;

                Vector3 dir = (forward
                    + right * Mathf.Tan(angleH)
                    + up    * Mathf.Tan(angleV)).normalized;

                RaycastHit hit;
                if (!Physics.Raycast(cameraPos, dir, out hit, maxRayDistance, raycastMask))
                    continue;

                if (!hit.collider.CompareTag(targetTag)) continue;

                Vector3Int idx = WorldToVoxel(hit.point);
                if (!IsValidVoxel(idx)) continue;

                bool wasZero = confidence[idx.x, idx.y, idx.z] < 0.001f;

                // Exponential confidence update toward 1
                confidence[idx.x, idx.y, idx.z] +=
                    confidenceDelta * (1f - confidence[idx.x, idx.y, idx.z]);
                confidence[idx.x, idx.y, idx.z] =
                    Mathf.Clamp01(confidence[idx.x, idx.y, idx.z]);

                if (wasZero) coveredVoxelCount++;
            }

            stepCounter++;

            if (voxelAnimator != null && stepCounter % visualizationUpdateInterval == 0)
                voxelAnimator.UpdateColors(confidence, voxelSize);
        }

        // ── Metrics ─────────────────────────────────────────────────

        /// <summary>Mean uncertainty over OCCUPIED voxels only.</summary>
        public float GetMeanUncertainty()
        {
            if (confidence == null || groundTruthOccupancy == null) return 1f;
            float sum = 0f; int count = 0;
            for (int x = 0; x < voxelSize; x++)
            for (int y = 0; y < voxelSize; y++)
            for (int z = 0; z < voxelSize; z++)
                if (groundTruthOccupancy[x, y, z])
                { sum += 1f - confidence[x, y, z]; count++; }
            return count > 0 ? sum / count : 1f;
        }

        /// <summary>Fraction of occupied voxels hit at least once.</summary>
        public float GetVoxelCoverage()
        {
            if (totalOccupiedVoxels == 0 || groundTruthOccupancy == null) return 0f;
            int covered = 0;
            for (int x = 0; x < voxelSize; x++)
            for (int y = 0; y < voxelSize; y++)
            for (int z = 0; z < voxelSize; z++)
                if (groundTruthOccupancy[x, y, z] && confidence[x, y, z] > 0.001f)
                    covered++;
            return (float)covered / totalOccupiedVoxels;
        }

        /// <summary>Fraction of occupied voxels with confidence > 0.5.</summary>
        public float GetReconstructionAccuracy()
        {
            if (totalOccupiedVoxels == 0 || groundTruthOccupancy == null) return 0f;
            int correct = 0;
            for (int x = 0; x < voxelSize; x++)
            for (int y = 0; y < voxelSize; y++)
            for (int z = 0; z < voxelSize; z++)
                if (groundTruthOccupancy[x, y, z] && confidence[x, y, z] > 0.5f)
                    correct++;
            return (float)correct / totalOccupiedVoxels;
        }

        public int GetCoveredVoxelCount() => coveredVoxelCount;

        public void GetUncertaintyHistogram(float[] histogram, int bins)
        {
            if (confidence == null || groundTruthOccupancy == null || histogram == null) return;
            System.Array.Clear(histogram, 0, histogram.Length);
            int count = 0;
            for (int x = 0; x < voxelSize; x++)
            for (int y = 0; y < voxelSize; y++)
            for (int z = 0; z < voxelSize; z++)
                if (groundTruthOccupancy[x, y, z])
                {
                    float u   = 1f - confidence[x, y, z];
                    int   bin = Mathf.Min(Mathf.FloorToInt(u * bins), bins - 1);
                    histogram[bin]++;
                    count++;
                }
            if (count > 0)
                for (int i = 0; i < bins; i++) histogram[i] /= count;
        }

        public Vector3 GetUncertaintyCentroid()
        {
            if (confidence == null || groundTruthOccupancy == null) return sceneBoundsCenter;
            Vector3 sum = Vector3.zero; float weightSum = 0f;
            for (int x = 0; x < voxelSize; x++)
            for (int y = 0; y < voxelSize; y++)
            for (int z = 0; z < voxelSize; z++)
                if (groundTruthOccupancy[x, y, z])
                {
                    float u = 1f - confidence[x, y, z];
                    if (u > 0.3f)
                    {
                        sum       += VoxelToWorld(new Vector3Int(x, y, z)) * u;
                        weightSum += u;
                    }
                }
            return weightSum > 1e-6f ? sum / weightSum : sceneBoundsCenter;
        }

        public float[,,] GetConfidenceArray() => confidence;

        // ── Private ──────────────────────────────────────────────────

        private void InitializeGrid()
        {
            confidence = new float[voxelSize, voxelSize, voxelSize];
            sceneBounds = new Bounds(sceneBoundsCenter, sceneBoundsSize);
        }

        private void ComputeGroundTruth()
        {
            groundTruthOccupancy = new bool[voxelSize, voxelSize, voxelSize];
            totalOccupiedVoxels  = 0;

            Vector3 cellSize = new Vector3(
                sceneBoundsSize.x / voxelSize,
                sceneBoundsSize.y / voxelSize,
                sceneBoundsSize.z / voxelSize);
            Vector3 halfCell = cellSize * 0.45f;

            // Use Ignore Raycast mask inverted so overlap only hits ship colliders
            int overlapMask = ~LayerMask.GetMask("Ignore Raycast");

            for (int x = 0; x < voxelSize; x++)
            for (int y = 0; y < voxelSize; y++)
            for (int z = 0; z < voxelSize; z++)
            {
                Vector3 center = VoxelToWorld(new Vector3Int(x, y, z));
                Collider[] cols = Physics.OverlapBox(
                    center, halfCell, Quaternion.identity, overlapMask);

                bool occupied = false;
                for (int c = 0; c < cols.Length; c++)
                    if (cols[c].CompareTag(targetTag)) { occupied = true; break; }

                groundTruthOccupancy[x, y, z] = occupied;
                if (occupied) totalOccupiedVoxels++;
            }

            Debug.Log($"[ReconstructionManager] Ground truth: {totalOccupiedVoxels} occupied voxels.");
        }

        private Vector3Int WorldToVoxel(Vector3 worldPos)
        {
            Vector3 origin = sceneBoundsCenter - sceneBoundsSize * 0.5f;
            Vector3 local  = worldPos - origin;
            return new Vector3Int(
                Mathf.FloorToInt(local.x / sceneBoundsSize.x * voxelSize),
                Mathf.FloorToInt(local.y / sceneBoundsSize.y * voxelSize),
                Mathf.FloorToInt(local.z / sceneBoundsSize.z * voxelSize));
        }

        private Vector3 VoxelToWorld(Vector3Int idx)
        {
            Vector3 origin = sceneBoundsCenter - sceneBoundsSize * 0.5f;
            return origin + new Vector3(
                (idx.x + 0.5f) / voxelSize * sceneBoundsSize.x,
                (idx.y + 0.5f) / voxelSize * sceneBoundsSize.y,
                (idx.z + 0.5f) / voxelSize * sceneBoundsSize.z);
        }

        private bool IsValidVoxel(Vector3Int idx) =>
            idx.x >= 0 && idx.x < voxelSize &&
            idx.y >= 0 && idx.y < voxelSize &&
            idx.z >= 0 && idx.z < voxelSize;
    }
}