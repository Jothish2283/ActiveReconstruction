// ============================================================
// VoxelColorAnimator.cs — FIXED
// Only renders voxels that overlap reconstructable geometry.
// Uncertain = red, certain = green. High alpha for visibility.
// ============================================================

using UnityEngine;

namespace ReconstructionRL
{
    public class VoxelColorAnimator : MonoBehaviour
    {
        [Header("Voxel Grid")]
        public Transform voxelParent;
        public bool autoGenerateGrid = true;
        [Range(4, 64)] public int gridSize = 32;
        public Vector3 gridCenter = Vector3.zero;
        public Vector3 gridWorldSize = new Vector3(5f, 5f, 5f);

        [Header("Color Mapping")]
        public Color uncertainColor = Color.red;
        public Color certainColor   = Color.green;
        [Range(0f, 1f)] public float voxelAlpha = 0.6f;

        [Header("Filter")]
        [Tooltip("Only show voxels that overlap reconstructable geometry. Hides empty space.")]
        public string targetTag = "Reconstructable";
        [Tooltip("Only render voxels whose uncertainty exceeds this. 0 = show all occupied.")]
        [Range(0f, 1f)] public float uncertaintyThreshold = 0.0f;

        [Header("Performance")]
        [Range(1, 10)] public int updateThrottle = 2;

        // ── Private ──────────────────────────────────────────────────
        private Renderer[]             voxelRenderers;
        private MaterialPropertyBlock[] propBlocks;
        private bool[]                 isOccupied;   // precomputed: does voxel overlap ship?
        private int                    totalVoxels;
        private int                    throttleCounter;
        private bool                   initialized;

        private static readonly int ColorID = Shader.PropertyToID("_Color");

        // ── Unity Lifecycle ──────────────────────────────────────────

        private void Awake()
        {
            if (autoGenerateGrid)
                GenerateVoxelGrid();
        }

        // Called by ReconstructionManager after ground truth is computed
        public void InitOccupancy(bool[,,] groundTruth, int voxelSize)
        {
            if (!initialized || groundTruth == null) return;
            isOccupied = new bool[totalVoxels];
            int idx = 0;
            for (int x = 0; x < voxelSize; x++)
                for (int y = 0; y < voxelSize; y++)
                    for (int z = 0; z < voxelSize; z++, idx++)
                        if (idx < totalVoxels)
                            isOccupied[idx] = groundTruth[x, y, z];

            // Initially hide all non-occupied voxels
            for (int i = 0; i < totalVoxels; i++)
                if (voxelRenderers[i] != null)
                    voxelRenderers[i].enabled = isOccupied != null && isOccupied[i];
        }

        // ── Public API ───────────────────────────────────────────────

        public void UpdateColors(float[,,] confidence, int voxelSize)
        {
            if (!initialized || confidence == null) return;

            throttleCounter++;
            if (throttleCounter % updateThrottle != 0) return;

            int idx = 0;
            for (int x = 0; x < voxelSize && idx < totalVoxels; x++)
            for (int y = 0; y < voxelSize && idx < totalVoxels; y++)
            for (int z = 0; z < voxelSize && idx < totalVoxels; z++, idx++)
            {
                if (voxelRenderers[idx] == null) continue;

                // Only show occupied voxels (ship geometry)
                bool occupied = isOccupied != null && isOccupied[idx];
                if (!occupied) { voxelRenderers[idx].enabled = false; continue; }

                float conf        = Mathf.Clamp01(confidence[x, y, z]);
                float uncertainty = 1f - conf;

                // Hide if below threshold
                if (uncertainty < uncertaintyThreshold)
                {
                    voxelRenderers[idx].enabled = false;
                    continue;
                }

                voxelRenderers[idx].enabled = true;

                // Red (uncertain) → Green (certain)
                Color col = Color.Lerp(uncertainColor, certainColor, conf);
                col.a = voxelAlpha;

                propBlocks[idx].SetColor(ColorID, col);
                voxelRenderers[idx].SetPropertyBlock(propBlocks[idx]);
            }
        }

        public void ResetColors()
        {
            if (!initialized) return;
            Color c = uncertainColor; c.a = voxelAlpha;
            for (int i = 0; i < totalVoxels; i++)
            {
                if (voxelRenderers[i] == null) continue;
                bool occ = isOccupied != null && isOccupied[i];
                voxelRenderers[i].enabled = occ;
                if (!occ) continue;
                propBlocks[i].SetColor(ColorID, c);
                voxelRenderers[i].SetPropertyBlock(propBlocks[i]);
            }
        }

        // ── Private ──────────────────────────────────────────────────

        private void GenerateVoxelGrid()
        {
            if (voxelParent == null)
            {
                GameObject p = new GameObject("VoxelGrid_Auto");
                voxelParent = p.transform;
            }

            // Clear old children
            for (int i = voxelParent.childCount - 1; i >= 0; i--)
                DestroyImmediate(voxelParent.GetChild(i).gameObject);

            Vector3 cellSize = new Vector3(
                gridWorldSize.x / gridSize,
                gridWorldSize.y / gridSize,
                gridWorldSize.z / gridSize);

            Vector3 origin = gridCenter - gridWorldSize * 0.5f + cellSize * 0.5f;

            totalVoxels    = gridSize * gridSize * gridSize;
            voxelRenderers = new Renderer[totalVoxels];
            propBlocks     = new MaterialPropertyBlock[totalVoxels];
            isOccupied     = new bool[totalVoxels];   // all false until InitOccupancy()

            // Find or create transparent material once
            Material mat = BuildTransparentMaterial();

            int idx = 0;
            for (int x = 0; x < gridSize; x++)
            for (int y = 0; y < gridSize; y++)
            for (int z = 0; z < gridSize; z++, idx++)
            {
                Vector3 pos = origin + new Vector3(
                    x * cellSize.x, y * cellSize.y, z * cellSize.z);

                GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
                cube.name = $"V{x}_{y}_{z}";
                cube.transform.SetParent(voxelParent, false);
                cube.transform.position  = pos;
                cube.transform.localScale = cellSize * 0.88f;

                // Must use DestroyImmediate so collider is gone before ComputeGroundTruth
                DestroyImmediate(cube.GetComponent<Collider>());

                // Put on Ignore Raycast layer so raycasts never hit voxel cubes
                cube.layer = LayerMask.NameToLayer("Ignore Raycast");

                Renderer r = cube.GetComponent<Renderer>();
                r.sharedMaterial = mat;
                r.enabled = false;   // hidden until InitOccupancy() marks it occupied

                voxelRenderers[idx] = r;
                propBlocks[idx]     = new MaterialPropertyBlock();

                // Set initial red color
                Color initCol = uncertainColor; initCol.a = voxelAlpha;
                propBlocks[idx].SetColor(ColorID, initCol);
                r.SetPropertyBlock(propBlocks[idx]);
            }

            initialized = true;
            Debug.Log($"[VoxelColorAnimator] Generated {totalVoxels} voxel cubes.");
        }

        private Material BuildTransparentMaterial()
        {
            Material mat = new Material(Shader.Find("Standard"));
            mat.SetFloat("_Mode", 3);
            mat.SetInt("_SrcBlend",  (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetInt("_DstBlend",  (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            mat.SetInt("_ZWrite",    0);
            mat.DisableKeyword("_ALPHATEST_ON");
            mat.EnableKeyword("_ALPHABLEND_ON");
            mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            mat.renderQueue = 3000;
            return mat;
        }
    }
}