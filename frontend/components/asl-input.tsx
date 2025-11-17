"use client"

import type React from "react"

import { useState, useRef, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Upload, Camera, X, Image as ImageIcon, RefreshCw } from "lucide-react"

interface ASLInputProps {
  onImageCapture: (imageBlob: Blob) => void
  onImageClear?: () => void
  isProcessing: boolean
  externalImageSrc?: string | null
  externalImageBlob?: Blob | null
}

const MAX_FILE_SIZE = 5 * 1024 * 1024 // 5MB
const MAX_IMAGE_DIMENSION = 1024

async function compressImage(file: File): Promise<Blob> {
  return new Promise((resolve, reject) => {
    // Check file size first
    if (file.size > MAX_FILE_SIZE) {
      reject(
        new Error(
          `Image too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Please use an image smaller than 5MB.`
        )
      )
      return
    }

    const img = new Image()
    const reader = new FileReader()

    reader.onload = (e) => {
      img.src = e.target?.result as string
    }

    img.onload = () => {
      const canvas = document.createElement("canvas")
      let { width, height } = img

      // Resize if too large
      if (width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION) {
        if (width > height) {
          height = (height / width) * MAX_IMAGE_DIMENSION
          width = MAX_IMAGE_DIMENSION
        } else {
          width = (width / height) * MAX_IMAGE_DIMENSION
          height = MAX_IMAGE_DIMENSION
        }
      }

      canvas.width = width
      canvas.height = height
      const ctx = canvas.getContext("2d")!
      ctx.drawImage(img, 0, 0, width, height)

      canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(new Error("Image compression failed"))
            return
          }
          resolve(blob)
        },
        "image/jpeg",
        0.9
      )
    }

    img.onerror = () => reject(new Error("Failed to load image"))
    reader.onerror = () => reject(new Error("Failed to read file"))
    reader.readAsDataURL(file)
  })
}

export function ASLInput({
  onImageCapture,
  onImageClear,
  isProcessing,
  externalImageSrc,
  externalImageBlob
}: ASLInputProps) {
  const [mode, setMode] = useState<"upload" | "camera" | null>(null)
  const [imageSrc, setImageSrc] = useState<string | null>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [currentBlob, setCurrentBlob] = useState<Blob | null>(null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const isMountedRef = useRef(true)
  const blobUrlRef = useRef<string | null>(null)

  // Cleanup blob URLs on unmount or when imageSrc changes
  useEffect(() => {
    return () => {
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current)
        blobUrlRef.current = null
      }
    }
  }, [imageSrc])

  // Cleanup camera stream on unmount
  useEffect(() => {
    isMountedRef.current = true

    return () => {
      isMountedRef.current = false
      if (stream) {
        stream.getTracks().forEach((track) => {
          track.stop()
          console.log(`Stopped ${track.kind} track on unmount`)
        })
      }
    }
  }, [stream])

  // Keyboard shortcuts for camera mode
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (mode === "camera" && !imageSrc && (e.key === " " || e.key === "Enter")) {
        e.preventDefault()
        captureImage()
      }
    }

    if (mode === "camera" && !imageSrc) {
      window.addEventListener("keydown", handleKeyPress)
      return () => window.removeEventListener("keydown", handleKeyPress)
    }
  }, [mode, imageSrc])

  // Sync external image from history
  useEffect(() => {
    if (externalImageSrc && externalImageBlob) {
      setImageSrc(externalImageSrc)
      setCurrentBlob(externalImageBlob)
      setMode("upload")
    }
  }, [externalImageSrc, externalImageBlob])

  // Attach stream to video element when both are available
  useEffect(() => {
    const attachStreamToVideo = async () => {
      if (stream && videoRef.current && mode === "camera") {
        videoRef.current.srcObject = stream
        try {
          await videoRef.current.play()
          console.log("Video stream attached and playing")
        } catch (playError) {
          console.error("Video play error:", playError)
          setError("Failed to start video playback. Please try again.")
        }
      }
    }

    attachStreamToVideo()
  }, [stream, mode])

  const createAndTrackBlobUrl = useCallback((blob: Blob) => {
    // Revoke previous URL if exists
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current)
    }
    const url = URL.createObjectURL(blob)
    blobUrlRef.current = url
    return url
  }, [])

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setError(null)

    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (JPG, PNG, WebP)")
      e.target.value = ""
      return
    }

    try {
      const compressed = await compressImage(file)
      const url = createAndTrackBlobUrl(compressed)
      setImageSrc(url)
      setCurrentBlob(compressed)
      setMode("upload")
      onImageCapture(compressed)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to process image")
      e.target.value = ""
    }
  }

  const startCamera = async () => {
    setError(null)

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { min: 320, ideal: 640, max: 1920 },
          height: { min: 240, ideal: 480, max: 1080 },
        },
        audio: false,
      })
      setStream(mediaStream)
      setMode("camera")
      // Note: The stream will be attached to the video element by the useEffect
    } catch (error) {
      console.error("Camera access error:", error)
      setError(
        "Unable to access camera. Please check permissions and ensure no other app is using the camera."
      )
    }
  }

  const captureImage = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) {
      console.error("Video or canvas ref not available")
      return
    }

    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext("2d")

    if (!context) {
      console.error("Canvas context not available")
      return
    }

    // Check if video has valid dimensions
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.error("Video not ready - invalid dimensions")
      setError("Camera not ready yet. Please wait a moment and try again.")
      return
    }

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw the current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Convert canvas to blob with mounted check
    canvas.toBlob(
      (blob) => {
        // Guard against unmounted component
        if (!isMountedRef.current || !blob) return

        const url = createAndTrackBlobUrl(blob)
        setImageSrc(url)
        setCurrentBlob(blob)
        onImageCapture(blob)

        // Stop camera stream
        if (stream) {
          stream.getTracks().forEach((track) => track.stop())
          setStream(null)
        }
      },
      "image/jpeg",
      0.9
    )
  }, [stream, onImageCapture, createAndTrackBlobUrl])

  const reset = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
    }
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current)
      blobUrlRef.current = null
    }
    setImageSrc(null)
    setCurrentBlob(null)
    setMode(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }

    // Notify parent to clear results
    onImageClear?.()
  }, [stream, onImageClear])

  const handleRefresh = useCallback(() => {
    if (currentBlob) {
      onImageCapture(currentBlob)
    }
  }, [currentBlob, onImageCapture])

  return (
    <Card className="flex flex-col overflow-hidden bg-card">
      <div className="border-b border-border bg-muted/30 px-4 py-3">
        <h2 className="font-semibold text-card-foreground">ASL Sign Input</h2>
        <p className="text-sm text-muted-foreground">Upload an image or use your camera to capture a sign</p>
      </div>

      <div className="flex flex-1 flex-col p-4">
        {error && (
          <div
            className="mb-4 rounded-lg border border-destructive bg-destructive/10 p-3 text-sm text-destructive"
            role="alert"
            aria-live="assertive"
          >
            {error}
          </div>
        )}

        {!mode && (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <div className="flex flex-col gap-3 sm:flex-row">
              <Button
                size="lg"
                onClick={() => fileInputRef.current?.click()}
                disabled={isProcessing}
                className="gap-2"
                aria-label="Upload image file"
                aria-busy={isProcessing}
              >
                <Upload className="h-5 w-5" aria-hidden="true" />
                Upload Image
              </Button>

              <Button
                size="lg"
                variant="outline"
                onClick={startCamera}
                disabled={isProcessing}
                className="gap-2 bg-transparent"
                aria-label="Start camera to capture sign"
                aria-busy={isProcessing}
              >
                <Camera className="h-5 w-5" aria-hidden="true" />
                Use Camera
              </Button>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="hidden"
              aria-label="Upload ASL sign image"
              id="file-upload-input"
            />

            <label htmlFor="file-upload-input" className="sr-only">
              Upload an image of an ASL alphabet sign
            </label>

            <p className="text-center text-sm text-muted-foreground">
              Supported formats: JPG, PNG, WebP (max 5MB)
            </p>
          </div>
        )}

        {mode === "camera" && !imageSrc && (
          <div className="flex flex-1 flex-col gap-4">
            <div className="relative flex-1 overflow-hidden rounded-lg bg-black">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="h-full w-full object-cover"
                aria-label="Camera preview for capturing ASL sign"
              />
              <canvas ref={canvasRef} className="hidden" />
            </div>

            <div className="flex gap-2">
              <Button
                onClick={captureImage}
                className="flex-1"
                size="lg"
                aria-label="Capture image from camera (Press Space or Enter)"
              >
                <ImageIcon className="mr-2 h-5 w-5" aria-hidden="true" />
                Capture Sign
              </Button>

              <Button onClick={reset} variant="outline" size="lg" aria-label="Cancel and go back">
                <X className="h-5 w-5" aria-hidden="true" />
              </Button>
            </div>

            <p className="text-center text-xs text-muted-foreground">Tip: Press Space or Enter to capture</p>
          </div>
        )}

        {imageSrc && (
          <div className="flex flex-1 flex-col gap-4">
            <div className="relative flex-1 overflow-hidden rounded-lg bg-black/10">
              <img src={imageSrc} alt="Captured ASL sign" className="h-full w-full object-contain max-h-96" />
            </div>

            <div className="flex gap-2">
              <Button
                onClick={handleRefresh}
                variant="default"
                size="lg"
                className="flex-1 gap-2"
                disabled={isProcessing || !currentBlob}
                aria-label="Re-analyze this image"
                aria-busy={isProcessing}
              >
                <RefreshCw className={`h-5 w-5 ${isProcessing ? 'animate-spin' : ''}`} aria-hidden="true" />
                {isProcessing ? 'Analyzing...' : 'Re-analyze'}
              </Button>

              <Button
                onClick={reset}
                variant="outline"
                size="lg"
                className="gap-2 bg-transparent"
                disabled={isProcessing}
                aria-label="Clear image and start over"
              >
                <X className="h-5 w-5" aria-hidden="true" />
              </Button>
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}
