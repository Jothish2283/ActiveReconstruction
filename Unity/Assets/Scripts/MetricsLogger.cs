// ============================================================
// MetricsLogger.cs
// Logs per-episode metrics to CSV for RL, Random, and Sequential modes.
// Columns: Episode, TotalReward, Coverage, MeanUncertainty,
//          ReconstructionAccuracy, ImageCount, RuntimeSeconds, Mode
// ============================================================

using System;
using System.IO;
using System.Text;
using UnityEngine;

namespace ReconstructionRL
{
    /// <summary>
    /// Writes per-episode reconstruction metrics to CSV.
    /// Switch Mode enum to produce correctly named output files.
    /// </summary>
    public class MetricsLogger : MonoBehaviour
    {
        // ── Enums ────────────────────────────────────────────────────

        public enum RunMode { RL, Random, Sequential }

        // ── Inspector Variables ──────────────────────────────────────
        [Header("Mode")]
        [Tooltip("Selects which CSV file to write (RL/Random/Sequential).")]
        public RunMode mode = RunMode.RL;

        [Header("Logging Settings")]
        [Tooltip("Folder (relative to Application.dataPath) for CSV output.")]
        public string logFolder = "Metrics";

        [Tooltip("Flush CSV to disk every N episodes (0 = every episode).")]
        [Range(0, 50)]
        public int flushInterval = 5;

        [Tooltip("If true, overwrite existing CSV on start. If false, append.")]
        public bool overwriteOnStart = true;

        [Header("Runtime Display")]
        [Tooltip("If true, print latest metrics row to console.")]
        public bool logToConsole = true;

        // ── Private State ────────────────────────────────────────────
        private string csvPath;
        private StreamWriter writer;
        private int episodeCount = 0;
        // Add this inside MetricsLogger class, after episodeCount field:
        public int EpisodeCount => episodeCount;
        private bool initialized = false;
        private StringBuilder sb = new StringBuilder(256);

        // CSV column header
        private const string CSV_HEADER =
            "Episode,TotalReward,Coverage,MeanUncertainty,ReconstructionAccuracy," +
            "ImageCount,RuntimeSeconds,Mode,Timestamp";

        // ── Unity Lifecycle ──────────────────────────────────────────

        private void Awake()
        {
            InitCSV();
        }

        private void OnApplicationQuit()
        {
            CloseWriter();
        }

        private void OnDisable()
        {
            CloseWriter();
        }

        // ── Public API ───────────────────────────────────────────────

        /// <summary>
        /// Log one episode's metrics. Called by CameraAgent at episode end.
        /// </summary>
        public void LogEpisode(
            float totalReward,
            float coverage,
            float meanUncertainty,
            float reconstructionAccuracy,
            int imageCount,
            float runtimeSeconds)
        {
            if (!initialized) InitCSV();
            if (writer == null) return;

            episodeCount++;

            sb.Clear();
            sb.Append(episodeCount).Append(',');
            sb.Append(totalReward.ToString("F4")).Append(',');
            sb.Append(coverage.ToString("F4")).Append(',');
            sb.Append(meanUncertainty.ToString("F4")).Append(',');
            sb.Append(reconstructionAccuracy.ToString("F4")).Append(',');
            sb.Append(imageCount).Append(',');
            sb.Append(runtimeSeconds.ToString("F2")).Append(',');
            sb.Append(mode.ToString()).Append(',');
            sb.Append(DateTime.Now.ToString("HH:mm:ss"));

            writer.WriteLine(sb.ToString());

            if (flushInterval == 0 || episodeCount % flushInterval == 0)
                writer.Flush();

            if (logToConsole)
                Debug.Log($"[MetricsLogger][{mode}] Ep {episodeCount}: " +
                          $"R={totalReward:F3} Cov={coverage:F3} U={meanUncertainty:F3} " +
                          $"Acc={reconstructionAccuracy:F3} imgs={imageCount}");
        }

        /// <summary>
        /// Switch mode at runtime (e.g. when switching baselines).
        /// Closes current writer and opens new file.
        /// </summary>
        public void SetMode(RunMode newMode)
        {
            CloseWriter();
            mode = newMode;
            episodeCount = 0;
            InitCSV();
        }

        /// <summary>
        /// Get path of current CSV file.
        /// </summary>
        public string GetCSVPath() => csvPath;

        // ── Private Helpers ──────────────────────────────────────────

        private void InitCSV()
        {
            try
            {
                // Build output directory
                string baseDir = Application.isEditor
                    ? Path.Combine("D:/Downloads_IDM/ActiveReconstruction_Final/Assets", logFolder)
                    : Path.Combine("D:/Downloads_IDM/ActiveReconstruction_Final/Assets", logFolder);

                Directory.CreateDirectory(baseDir);

                // File named by mode
                string fileName = $"{mode}_metrics.csv";
                csvPath = Path.Combine(baseDir, fileName);

                // Open writer
                FileMode fileMode = (overwriteOnStart || !File.Exists(csvPath))
                    ? FileMode.Create
                    : FileMode.Append;

                writer = new StreamWriter(new FileStream(csvPath, fileMode, FileAccess.Write, FileShare.Read));

                // Write header only when creating fresh
                if (fileMode == FileMode.Create)
                    writer.WriteLine(CSV_HEADER);

                initialized = true;
                Debug.Log($"[MetricsLogger] Writing to: {csvPath}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[MetricsLogger] Failed to init CSV: {e.Message}");
                initialized = false;
            }
        }

        private void CloseWriter()
        {
            if (writer != null)
            {
                writer.Flush();
                writer.Close();
                writer = null;
            }
        }

        /// <summary>
        /// Read any metrics CSV and print one-line summary. Call from anywhere.
        /// </summary>
        public static void PrintSummaryFromCSV(string csvPath, string label)
        {
            if (!System.IO.File.Exists(csvPath))
            {
                Debug.Log($"[MetricsLogger] No file at {csvPath}");
                return;
            }
            try
            {
                string[] lines = System.IO.File.ReadAllLines(csvPath);
                if (lines.Length < 2) { Debug.Log($"[{label}] No data rows."); return; }

                float r=0, cov=0, u=0, acc=0; int n=0;
                for (int i = 1; i < lines.Length; i++)
                {
                    string[] c = lines[i].Split(',');
                    if (c.Length < 5) continue;
                    float.TryParse(c[1], out float rv); float.TryParse(c[2], out float cv);
                    float.TryParse(c[3], out float uv); float.TryParse(c[4], out float av);
                    r+=rv; cov+=cv; u+=uv; acc+=av; n++;
                }
                if (n == 0) return;
                Debug.Log($"══ {label} SUMMARY ({n} eps) ══  " +
                        $"Reward={r/n:F3}  Coverage={cov/n:F3}  " +
                        $"Uncertainty={u/n:F3}  Accuracy={acc/n:F3}");
            }
            catch (System.Exception e) { Debug.LogError($"Summary error: {e.Message}"); }
        }

        /// <summary>Force flush writer to disk immediately (call before reading CSV).</summary>
        public void FlushNow()
        {
            if (writer != null) writer.Flush();
        }
        /// <summary>
        /// Close writer, read CSV contents as string array, reopen writer in append mode.
        /// Safe way to read CSV while MetricsLogger is active.
        /// </summary>
        public string[] ReadCSVLines()
        {
            // Close writer temporarily
            if (writer != null) { writer.Flush(); writer.Close(); writer = null; }

            string[] lines = System.Array.Empty<string>();
            try
            {
                if (File.Exists(csvPath))
                    lines = File.ReadAllLines(csvPath);
            }
            catch (System.Exception e)
            {
                Debug.LogError("[MetricsLogger] ReadCSVLines error: " + e.Message);
            }

            // Reopen in append mode so logging continues
            try
            {
                writer = new StreamWriter(
                    new FileStream(csvPath, FileMode.Append, FileAccess.Write, FileShare.Read));
            }
            catch (System.Exception e)
            {
                Debug.LogError("[MetricsLogger] Failed to reopen writer: " + e.Message);
            }

            return lines;
        }
    }
}
