"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Upload, Video, X } from "lucide-react"

interface VideoInputProps {
  onVideoProcess: (videoBlob: Blob) => void
  isProcessing: boolean
}

export function VideoInput({ onVideoProcess, isProcessing }: VideoInputProps) {
  const [mode, setMode] = useState<"upload" | "camera" | null>(null)
  const [videoSrc, setVideoSrc] = useState<string | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [stream, setStream] = useState<MediaStream | null>(null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith("video/")) {
      const url = URL.createObjectURL(file)
      setVideoSrc(url)
      setMode("upload")
      onVideoProcess(file)
    }
  }

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      })
      setStream(mediaStream)
      setMode("camera")

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }
    } catch (error) {
      console.error("[v0] Camera access error:", error)
      alert("Unable to access camera. Please check permissions.")
    }
  }

  const startRecording = () => {
    if (!stream) return

    chunksRef.current = []
    const mediaRecorder = new MediaRecorder(stream)
    mediaRecorderRef.current = mediaRecorder

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        chunksRef.current.push(e.data)
      }
    }

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" })
      const url = URL.createObjectURL(blob)
      setVideoSrc(url)
      onVideoProcess(blob)

      // Stop camera stream
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
    }

    mediaRecorder.start()
    setIsRecording(true)
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const reset = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
    }
    if (videoSrc) {
      URL.revokeObjectURL(videoSrc)
    }
    setVideoSrc(null)
    setMode(null)
    setIsRecording(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  return (
    <Card className="flex flex-col overflow-hidden bg-card">
      <div className="border-b border-border bg-muted/30 px-4 py-3">
        <h2 className="font-semibold text-card-foreground">Video Input</h2>
        <p className="text-sm text-muted-foreground">Upload a video or use your camera</p>
      </div>

      <div className="flex flex-1 flex-col p-4">
        {!mode && (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <div className="flex flex-col gap-3 sm:flex-row">
              <Button size="lg" onClick={() => fileInputRef.current?.click()} disabled={isProcessing} className="gap-2">
                <Upload className="h-5 w-5" />
                Upload Video
              </Button>

              <Button
                size="lg"
                variant="outline"
                onClick={startCamera}
                disabled={isProcessing}
                className="gap-2 bg-transparent"
              >
                <Video className="h-5 w-5" />
                Use Camera
              </Button>
            </div>

            <input ref={fileInputRef} type="file" accept="video/*" onChange={handleFileUpload} className="hidden" />

            <p className="text-center text-sm text-muted-foreground">Supported formats: MP4, WebM, MOV</p>
          </div>
        )}

        {mode === "camera" && !videoSrc && (
          <div className="flex flex-1 flex-col gap-4">
            <div className="relative flex-1 overflow-hidden rounded-lg bg-black">
              <video ref={videoRef} autoPlay muted playsInline className="h-full w-full object-cover" />
            </div>

            <div className="flex gap-2">
              {!isRecording ? (
                <Button onClick={startRecording} className="flex-1" size="lg">
                  Start Recording
                </Button>
              ) : (
                <Button onClick={stopRecording} variant="destructive" className="flex-1" size="lg">
                  Stop Recording
                </Button>
              )}

              <Button onClick={reset} variant="outline" size="lg">
                <X className="h-5 w-5" />
              </Button>
            </div>
          </div>
        )}

        {videoSrc && (
          <div className="flex flex-1 flex-col gap-4">
            <div className="relative flex-1 overflow-hidden rounded-lg bg-black">
              <video src={videoSrc} controls className="h-full w-full object-cover" />
            </div>

            <Button onClick={reset} variant="outline" size="lg" className="gap-2 bg-transparent">
              <X className="h-5 w-5" />
              Clear & Start Over
            </Button>
          </div>
        )}
      </div>
    </Card>
  )
}
