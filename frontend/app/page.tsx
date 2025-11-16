"use client"

import { useState } from "react"
import { ASLInput } from "@/components/asl-input"
import { ASLRecognitionDisplay } from "@/components/asl-recognition-display"
import { useASLRecognition, type ASLRecognitionResponse } from "@/lib/api-client"
import { Hand } from "lucide-react"

export default function Home() {
  const [recognitionResult, setRecognitionResult] = useState<ASLRecognitionResponse | null>(null)
  const { mutate: recognizeSign, isPending, error } = useASLRecognition()

  const handleImageCapture = async (imageBlob: Blob) => {
    // Reset previous result
    setRecognitionResult(null)

    // Call the recognition API
    recognizeSign(imageBlob, {
      onSuccess: (data) => {
        setRecognitionResult(data)
      },
    })
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
              <Hand className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground">ASL Alphabet Recognition Portal</h1>
              <p className="text-sm text-muted-foreground">
                Real-time American Sign Language recognition using AI
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content - Split View */}
      <main className="container mx-auto p-4">
        <div className="grid gap-4 lg:grid-cols-2 lg:gap-6">
          {/* Left Side - Image Input */}
          <ASLInput onImageCapture={handleImageCapture} isProcessing={isPending} />

          {/* Right Side - Recognition Display */}
          <ASLRecognitionDisplay
            result={recognitionResult}
            isRecognizing={isPending}
            error={error?.message || null}
          />
        </div>

        {/* Instructions Section */}
        <div className="mt-6 rounded-lg border border-border bg-card p-6">
          <h2 className="text-lg font-semibold mb-4">How to Use</h2>
          <div className="grid gap-4 md:grid-cols-3">
            <div>
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground font-bold mb-2">
                1
              </div>
              <h3 className="font-semibold mb-1">Capture or Upload</h3>
              <p className="text-sm text-muted-foreground">
                Use your camera to capture a sign or upload an existing image of an ASL alphabet letter (A-Z).
              </p>
            </div>
            <div>
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground font-bold mb-2">
                2
              </div>
              <h3 className="font-semibold mb-1">AI Analysis</h3>
              <p className="text-sm text-muted-foreground">
                Our AI model detects your hand pose using MediaPipe and classifies the sign with a PyTorch neural network.
              </p>
            </div>
            <div>
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground font-bold mb-2">
                3
              </div>
              <h3 className="font-semibold mb-1">Get Results</h3>
              <p className="text-sm text-muted-foreground">
                View the recognized letter with confidence score and detailed hand pose information.
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-card mt-8">
        <div className="container mx-auto px-4 py-6 text-center text-sm text-muted-foreground">
          <p>Powered by MediaPipe, PyTorch, FastAPI, and React</p>
        </div>
      </footer>
    </div>
  )
}
