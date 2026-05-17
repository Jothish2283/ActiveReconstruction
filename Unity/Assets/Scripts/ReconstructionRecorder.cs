// ============================================================
// ReconstructionRecorder.cs
// In-game UI overlay showing live reconstruction curves.
// Plots: uncertainty, coverage, accuracy vs episode step.
// Also supports screenshot capture for paper figures.
// Pure Unity GUI — no external dependencies required.
// ============================================================

using System.Collections.Generic;
using UnityEngine;

namespace ReconstructionRL
{
    /// <summary>
    /// Renders live graphs of reconstruction metrics as Unity OnGUI overlay.
    /// Tracks multiple run modes (RL, Random, Sequential) for visual comparison.
    /// Supports screenshot export to Application.persistentDataPath.
    /// </summary>
    public class ReconstructionRecorder : MonoBehaviour
    {
        // ── Inspector Variables ──────────────────────────────────────
        [Header("Dependencies")]
        public ReconstructionManager reconstructionManager;
        public MetricsLogger metricsLogger;

        [Header("UI Layout")]
        [Tooltip("Position and size of the overlay panel (normalized 0-1 screen coords).")]
        public Rect panelRect = new Rect(0.01f, 0.01f, 0.35f, 0.45f);

        [Tooltip("Height of each mini graph within panel (pixels).")]
        public int graphHeight = 80;

        [Tooltip("If true, show overlay during play.")]
        public bool showOverlay = true;

        [Header("Colors")]
        public Color rlColor = Color.cyan;
        public Color randomColor = Color.red;
        public Color sequentialColor = Color.yellow;
        public Color bgColor = new Color(0f, 0f, 0f, 0.7f);
        public Color gridColor = new Color(0.3f, 0.3f, 0.3f, 0.5f);

        [Header("Screenshot")]
        [Tooltip("Key to capture screenshot.")]
        public KeyCode screenshotKey = KeyCode.F12;

        [Tooltip("Screenshot file prefix.")]
        public string screenshotPrefix = "ReconSnapshot";

        // ── Private State ────────────────────────────────────────────
        // Circular buffers for each metric (max 200 samples)
        private const int MAX_SAMPLES = 200;

        private List<float> rlUncertainty = new List<float>(MAX_SAMPLES);
        private List<float> rlCoverage = new List<float>(MAX_SAMPLES);
        private List<float> rlAccuracy = new List<float>(MAX_SAMPLES);

        private List<float> rndUncertainty = new List<float>(MAX_SAMPLES);
        private List<float> rndCoverage = new List<float>(MAX_SAMPLES);

        private List<float> seqUncertainty = new List<float>(MAX_SAMPLES);
        private List<float> seqCoverage = new List<float>(MAX_SAMPLES);

        private Texture2D graphTex;
        private Texture2D bgTex;
        private GUIStyle labelStyle;
        private bool stylesInitialized = false;

        private int screenshotCount = 0;
        private float sampleTimer = 0f;
        private float sampleInterval = 0.5f; // sample every 0.5s

        // ── Unity Lifecycle ──────────────────────────────────────────

        private void Awake()
        {
            graphTex = new Texture2D(1, 1);
            graphTex.SetPixel(0, 0, Color.white);
            graphTex.Apply();

            bgTex = new Texture2D(1, 1);
            bgTex.SetPixel(0, 0, bgColor);
            bgTex.Apply();
        }

        private void Update()
        {
            // Sample current reconstruction state
            sampleTimer += Time.deltaTime;
            if (sampleTimer >= sampleInterval && reconstructionManager != null)
            {
                sampleTimer = 0f;
                float u = reconstructionManager.GetMeanUncertainty();
                float cov = reconstructionManager.GetVoxelCoverage();
                float acc = reconstructionManager.GetReconstructionAccuracy();

                // Route to correct run mode list
                MetricsLogger.RunMode mode = metricsLogger != null
                    ? metricsLogger.mode
                    : MetricsLogger.RunMode.RL;

                AddSample(mode, u, cov, acc);
            }

            // Screenshot capture
            if (Input.GetKeyDown(screenshotKey))
                CaptureScreenshot();
        }

        private void OnGUI()
        {
            if (!showOverlay) return;
            if (!stylesInitialized) InitStyles();

            // Convert normalized rect to pixel rect
            Rect pixelPanel = new Rect(
                panelRect.x * Screen.width,
                panelRect.y * Screen.height,
                panelRect.width * Screen.width,
                panelRect.height * Screen.height
            );

            // Background
            GUI.DrawTexture(pixelPanel, bgTex);

            float y = pixelPanel.y + 5f;
            float x = pixelPanel.x + 5f;
            float w = pixelPanel.width - 10f;

            // Title
            GUI.Label(new Rect(x, y, w, 20f), "  Reconstruction Monitor", labelStyle);
            y += 22f;

            // Graph: Mean Uncertainty
            DrawGraphLabel(x, y, "Mean Uncertainty (↓ better)");
            y += 16f;
            DrawMultiLineGraph(
                new Rect(x, y, w, graphHeight),
                rlUncertainty, rlColor,
                rndUncertainty, randomColor,
                seqUncertainty, sequentialColor,
                0f, 1f
            );
            y += graphHeight + 4f;

            // Graph: Coverage
            DrawGraphLabel(x, y, "Voxel Coverage (↑ better)");
            y += 16f;
            DrawMultiLineGraph(
                new Rect(x, y, w, graphHeight),
                rlCoverage, rlColor,
                rndCoverage, randomColor,
                seqCoverage, sequentialColor,
                0f, 1f
            );
            y += graphHeight + 4f;

            // Legend
            float lx = x;
            DrawLegendItem(ref lx, y, "RL", rlColor);
            DrawLegendItem(ref lx, y, "Random", randomColor);
            DrawLegendItem(ref lx, y, "Sequential", sequentialColor);

            y += 20f;
            // Live stats
            if (reconstructionManager != null)
            {
                GUI.Label(new Rect(x, y, w, 18f),
                    $"  U={reconstructionManager.GetMeanUncertainty():F3}  " +
                    $"Cov={reconstructionManager.GetVoxelCoverage():F3}  " +
                    $"Acc={reconstructionManager.GetReconstructionAccuracy():F3}",
                    labelStyle);
            }

            // F12 hint
            GUI.Label(new Rect(x, pixelPanel.yMax - 18f, w, 16f), "  [F12] Screenshot", labelStyle);
        }

        // ── Public API ───────────────────────────────────────────────

        /// <summary>
        /// Add a metric sample manually (e.g. from baseline scripts).
        /// </summary>
        public void AddSample(MetricsLogger.RunMode mode, float uncertainty, float coverage, float accuracy)
        {
            switch (mode)
            {
                case MetricsLogger.RunMode.RL:
                    AddToList(rlUncertainty, uncertainty);
                    AddToList(rlCoverage, coverage);
                    AddToList(rlAccuracy, accuracy);
                    break;
                case MetricsLogger.RunMode.Random:
                    AddToList(rndUncertainty, uncertainty);
                    AddToList(rndCoverage, coverage);
                    break;
                case MetricsLogger.RunMode.Sequential:
                    AddToList(seqUncertainty, uncertainty);
                    AddToList(seqCoverage, coverage);
                    break;
            }
        }

        /// <summary>
        /// Clear all graph data.
        /// </summary>
        public void ClearData()
        {
            rlUncertainty.Clear(); rlCoverage.Clear(); rlAccuracy.Clear();
            rndUncertainty.Clear(); rndCoverage.Clear();
            seqUncertainty.Clear(); seqCoverage.Clear();
        }

        /// <summary>
        /// Capture screenshot to persistentDataPath.
        /// </summary>
        public void CaptureScreenshot()
        {
            string path = System.IO.Path.Combine(
                "D:/Downloads_IDM/ActiveReconstruction_Final/Assets",
                $"{screenshotPrefix}_{screenshotCount:D4}.png"
            );
            ScreenCapture.CaptureScreenshot(path);
            screenshotCount++;
            Debug.Log($"[ReconstructionRecorder] Screenshot saved: {path}");
        }

        // ── Private Helpers ──────────────────────────────────────────

        private void AddToList(List<float> list, float value)
        {
            if (list.Count >= MAX_SAMPLES)
                list.RemoveAt(0);
            list.Add(value);
        }

        private void DrawGraphLabel(float x, float y, string text)
        {
            GUI.Label(new Rect(x, y, 300f, 16f), "  " + text, labelStyle);
        }

        private void DrawLegendItem(ref float x, float y, string label, Color color)
        {
            GUI.color = color;
            GUI.DrawTexture(new Rect(x, y + 3f, 14f, 10f), graphTex);
            GUI.color = Color.white;
            GUI.Label(new Rect(x + 16f, y, 80f, 18f), label, labelStyle);
            x += 80f;
        }

        /// <summary>
        /// Draw up to three overlapping line graphs in one rect.
        /// </summary>
        private void DrawMultiLineGraph(
            Rect rect,
            List<float> data1, Color col1,
            List<float> data2, Color col2,
            List<float> data3, Color col3,
            float minVal, float maxVal)
        {
            // Background
            GUI.color = new Color(0.1f, 0.1f, 0.1f, 0.8f);
            GUI.DrawTexture(rect, graphTex);
            GUI.color = Color.white;

            // Grid lines at 0.25, 0.5, 0.75
            for (int g = 1; g <= 3; g++)
            {
                float gy = rect.yMax - (g / 4f) * rect.height;
                GUI.color = gridColor;
                GUI.DrawTexture(new Rect(rect.x, gy, rect.width, 1f), graphTex);
            }
            GUI.color = Color.white;

            // Draw each dataset
            DrawLineGraph(rect, data1, col1, minVal, maxVal);
            DrawLineGraph(rect, data2, col2, minVal, maxVal);
            DrawLineGraph(rect, data3, col3, minVal, maxVal);
        }

        private void DrawLineGraph(Rect rect, List<float> data, Color color, float minVal, float maxVal)
        {
            if (data == null || data.Count < 2) return;
            float range = Mathf.Max(maxVal - minVal, 1e-6f);
            float xStep = rect.width / (data.Count - 1);

            GUI.color = color;
            for (int i = 1; i < data.Count; i++)
            {
                float x0 = rect.x + (i - 1) * xStep;
                float x1 = rect.x + i * xStep;
                float y0 = rect.yMax - ((data[i - 1] - minVal) / range) * rect.height;
                float y1 = rect.yMax - ((data[i] - minVal) / range) * rect.height;

                // Draw thick pixel line
                DrawLine(new Vector2(x0, y0), new Vector2(x1, y1), 2f);
            }
            GUI.color = Color.white;
        }

        private void DrawLine(Vector2 from, Vector2 to, float thickness)
        {
            float dx = to.x - from.x;
            float dy = to.y - from.y;
            float len = Mathf.Sqrt(dx * dx + dy * dy);
            if (len < 0.5f) return;

            float angle = Mathf.Atan2(dy, dx) * Mathf.Rad2Deg;
            GUIUtility.RotateAroundPivot(angle, from);
            GUI.DrawTexture(new Rect(from.x, from.y - thickness * 0.5f, len, thickness), graphTex);
            GUIUtility.RotateAroundPivot(-angle, from);
        }

        private void InitStyles()
        {
            labelStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 11,
                normal = { textColor = Color.white }
            };
            stylesInitialized = true;
        }
    }
}
