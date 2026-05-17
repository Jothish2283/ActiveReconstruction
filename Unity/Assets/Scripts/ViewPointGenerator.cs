// ============================================================
// ViewPointGenerator.cs
// Generates evenly distributed viewpoints on a sphere using
// the Fibonacci lattice (golden angle method).
// Deterministic: same N always gives same viewpoints.
// ============================================================

using UnityEngine;

namespace ReconstructionRL
{
    /// <summary>
    /// Generates N viewpoints uniformly distributed on a sphere
    /// using the Fibonacci/golden-angle spiral method.
    /// Provides deterministic, reproducible viewpoint sets.
    /// </summary>
    public class ViewPointGenerator : MonoBehaviour
    {
        // ── Inspector Variables ──────────────────────────────────────
        [Header("Generation Settings")]
        [Tooltip("Number of viewpoints to generate on the sphere.")]
        [Range(8, 128)]
        public int numViewpoints = 32;

        [Tooltip("Radius of the viewpoint sphere around the scene center.")]
        [Range(1f, 20f)]
        public float orbitRadius = 5f;

        [Tooltip("Center of the orbit sphere in world space.")]
        public Vector3 targetPosition = Vector3.zero;

        [Tooltip("If true, auto-generate viewpoints on Awake.")]
        public bool generateOnAwake = true;

        [Tooltip("If true, draw Gizmos for viewpoints in Editor.")]
        public bool showGizmos = true;

        // ── Private State ────────────────────────────────────────────
        private Vector3[] generatedViewpoints;

        // ── Unity Lifecycle ──────────────────────────────────────────

        private void Awake()
        {
            if (generateOnAwake)
                GenerateViewpoints();
        }

        private void OnDrawGizmos()
        {
            if (!showGizmos || generatedViewpoints == null) return;
            Gizmos.color = Color.cyan;
            for (int i = 0; i < generatedViewpoints.Length; i++)
            {
                Gizmos.DrawSphere(generatedViewpoints[i], 0.1f);
                Gizmos.DrawLine(targetPosition, generatedViewpoints[i]);
            }
        }

        // ── Public API ───────────────────────────────────────────────

        /// <summary>
        /// Generate Fibonacci sphere viewpoints. Must be called before GetViewpoints().
        /// </summary>
        public void GenerateViewpoints()
        {
            generatedViewpoints = GenerateFibonacciSphere(numViewpoints, orbitRadius, targetPosition);
            Debug.Log($"[ViewPointGenerator] Generated {numViewpoints} Fibonacci sphere viewpoints at radius {orbitRadius}.");
        }

        /// <summary>
        /// Returns the array of generated viewpoint positions.
        /// </summary>
        public Vector3[] GetViewpoints()
        {
            if (generatedViewpoints == null)
                GenerateViewpoints();
            return generatedViewpoints;
        }

        /// <summary>
        /// Returns a single viewpoint by index.
        /// </summary>
        public Vector3 GetViewpoint(int index)
        {
            if (generatedViewpoints == null) GenerateViewpoints();
            return generatedViewpoints[Mathf.Clamp(index, 0, generatedViewpoints.Length - 1)];
        }

        /// <summary>
        /// Returns count of generated viewpoints.
        /// </summary>
        public int Count => generatedViewpoints != null ? generatedViewpoints.Length : 0;

        // ── Static Utility ───────────────────────────────────────────

        /// <summary>
        /// Fibonacci sphere algorithm: distributes N points uniformly on a sphere.
        /// Based on golden angle φ = π(3 − √5) ≈ 2.39996 radians.
        /// Fully deterministic — no random seed required.
        /// </summary>
        public static Vector3[] GenerateFibonacciSphere(int n, float radius, Vector3 center)
        {
            Vector3[] points = new Vector3[n];
            float goldenAngle = Mathf.PI * (3f - Mathf.Sqrt(5f)); // ≈ 2.399963

            for (int i = 0; i < n; i++)
            {
                // y goes from 1 to -1 (top to bottom of sphere)
                float y = 1f - (i / (float)(n - 1)) * 2f;
                float radiusAtY = Mathf.Sqrt(Mathf.Max(0f, 1f - y * y)); // sin of inclination
                float theta = goldenAngle * i; // azimuthal angle

                float x = Mathf.Cos(theta) * radiusAtY;
                float z = Mathf.Sin(theta) * radiusAtY;

                points[i] = center + new Vector3(x, y, z) * radius;
            }

            return points;
        }

        /// <summary>
        /// Regenerate with new parameters at runtime.
        /// </summary>
        public void Regenerate(int newCount, float newRadius)
        {
            numViewpoints = newCount;
            orbitRadius = newRadius;
            GenerateViewpoints();
        }
    }
}
