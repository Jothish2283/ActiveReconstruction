// ============================================================
// CameraOrbit.cs
// Smooth camera movement between viewpoints on a sphere.
// Used for inference/visualization; training uses instant teleport.
// ============================================================

using UnityEngine;

namespace ReconstructionRL
{
    /// <summary>
    /// Handles smooth interpolated camera movement to target viewpoints.
    /// During training (high timescale), the agent bypasses this via direct
    /// transform assignment. During inference, this provides smooth orbit.
    /// </summary>
    public class CameraOrbit : MonoBehaviour
    {
        // ── Inspector Variables ──────────────────────────────────────
        [Header("Orbit Settings")]
        [Tooltip("Center of the scene that camera always looks at.")]
        public Transform sceneTarget;

        [Tooltip("Speed of interpolation (higher = faster movement).")]
        [Range(1f, 20f)]
        public float orbitSpeed = 5f;

        [Tooltip("If true, camera instantly snaps to target (use during training).")]
        public bool instantMove = false;

        [Header("Smoothing")]
        [Tooltip("Lerp factor for position interpolation per frame.")]
        [Range(0.01f, 1f)]
        public float positionLerpFactor = 0.15f;

        [Tooltip("Slerp factor for rotation smoothing.")]
        [Range(0.01f, 1f)]
        public float rotationSlerpFactor = 0.15f;

        // ── Private State ────────────────────────────────────────────
        private Vector3 targetPosition;
        private Quaternion targetRotation;
        private bool hasTarget = false;
        private bool reachedTarget = true;

        // ── Unity Lifecycle ──────────────────────────────────────────

        private void Awake()
        {
            targetPosition = transform.position;
            targetRotation = transform.rotation;

            // Auto-find scene target if not assigned
            if (sceneTarget == null)
            {
                GameObject go = GameObject.FindWithTag("Reconstructable");
                if (go != null)
                    sceneTarget = go.transform;
            }
        }

        private void LateUpdate()
        {
            if (!hasTarget || reachedTarget) return;

            if (instantMove || Time.timeScale > 5f)
            {
                // Fast training mode: instant snap
                transform.position = targetPosition;
                transform.rotation = targetRotation;
                reachedTarget = true;
                return;
            }

            // Smooth interpolation for inference/visualization
            transform.position = Vector3.Lerp(
                transform.position, targetPosition,
                positionLerpFactor * orbitSpeed * Time.deltaTime * 10f
            );

            transform.rotation = Quaternion.Slerp(
                transform.rotation, targetRotation,
                rotationSlerpFactor * orbitSpeed * Time.deltaTime * 10f
            );

            // Check arrival
            if (Vector3.Distance(transform.position, targetPosition) < 0.01f)
            {
                transform.position = targetPosition;
                transform.rotation = targetRotation;
                reachedTarget = true;
            }
        }

        // ── Public API ───────────────────────────────────────────────

        /// <summary>
        /// Set a new target position. Camera will smoothly move there.
        /// </summary>
        public void SetTargetPosition(Vector3 position)
        {
            targetPosition = position;

            // Always look at scene center
            Vector3 lookDir = (sceneTarget != null ? sceneTarget.position : Vector3.zero) - position;
            if (lookDir != Vector3.zero)
                targetRotation = Quaternion.LookRotation(lookDir.normalized, Vector3.up);

            hasTarget = true;
            reachedTarget = false;
        }

        /// <summary>
        /// Instantly snap to position without animation.
        /// </summary>
        public void SetPositionImmediate(Vector3 position)
        {
            targetPosition = position;
            transform.position = position;

            Vector3 lookDir = (sceneTarget != null ? sceneTarget.position : Vector3.zero) - position;
            if (lookDir != Vector3.zero)
            {
                targetRotation = Quaternion.LookRotation(lookDir.normalized, Vector3.up);
                transform.rotation = targetRotation;
            }

            hasTarget = true;
            reachedTarget = true;
        }

        /// <summary>
        /// Returns true when camera has reached its target.
        /// </summary>
        public bool HasReachedTarget() => reachedTarget;

        /// <summary>
        /// Returns current target position.
        /// </summary>
        public Vector3 GetTargetPosition() => targetPosition;

        /// <summary>
        /// Force look-at update (call after direct transform changes).
        /// </summary>
        public void ForceLookAtTarget()
        {
            if (sceneTarget == null) return;
            transform.LookAt(sceneTarget.position);
        }
    }
}
