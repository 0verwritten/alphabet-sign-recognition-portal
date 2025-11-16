import { useMutation } from "@tanstack/react-query"

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

interface Landmark {
  x: number
  y: number
  z: number
}

interface HandPose {
  landmarks: Landmark[]
  bounding_box?: {
    x_min: number
    y_min: number
    x_max: number
    y_max: number
  }
}

export interface ASLRecognitionResponse {
  letter: string
  confidence: number
  hand_pose?: HandPose
}

export interface ASLRecognitionError {
  detail?: string
  error?: string
}

/**
 * Predict ASL sign from an image
 */
export async function predictASLSign(imageBlob: Blob): Promise<ASLRecognitionResponse> {
  const formData = new FormData()
  formData.append("file", imageBlob, "sign.jpg")

  const response = await fetch(`${API_BASE_URL}/api/v1/asl/recognitions`, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({})) as ASLRecognitionError

    if (response.status === 422) {
      throw new Error(errorData.detail || "No hand detected in the image. Please ensure your hand is clearly visible.")
    } else if (response.status === 503) {
      throw new Error("ASL recognition model is not ready. Please contact the administrator.")
    } else if (response.status === 400) {
      throw new Error(errorData.detail || "Invalid image format. Please upload a valid image file.")
    } else {
      throw new Error(errorData.detail || errorData.error || "Failed to recognize ASL sign")
    }
  }

  return response.json()
}

/**
 * React Query hook for ASL recognition
 */
export function useASLRecognition() {
  return useMutation({
    mutationFn: predictASLSign,
    onError: (error) => {
      console.error("ASL Recognition Error:", error)
    },
  })
}
