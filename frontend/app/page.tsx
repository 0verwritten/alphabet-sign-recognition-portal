"use client"

import { useState } from "react"
import { ASLInput } from "@/components/asl-input"
import { ASLRecognitionDisplay } from "@/components/asl-recognition-display"
import { HistoryPanel } from "@/components/history-panel"
import { useASLRecognition, type ASLRecognitionResponse } from "@/lib/api-client"
import { useRecognitionHistory } from "@/hooks/use-recognition-history"
import { Hand, History } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function Home() {
  const [recognitionResult, setRecognitionResult] = useState<ASLRecognitionResponse | null>(null)
  const [currentImageBlob, setCurrentImageBlob] = useState<Blob | null>(null)
  const [currentImageSrc, setCurrentImageSrc] = useState<string | null>(null)
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)

  const { mutate: recognizeSign, isPending, error } = useASLRecognition()
  const { history, addToHistory, removeFromHistory, clearHistory } = useRecognitionHistory()

  const handleImageCapture = async (imageBlob: Blob) => {
    // Store the current image blob
    setCurrentImageBlob(imageBlob)

    // Clear external image source (used for history items)
    setCurrentImageSrc(null)

    // Reset previous result
    setRecognitionResult(null)

    // Call the recognition API
    recognizeSign(imageBlob, {
      onSuccess: (data) => {
        setRecognitionResult(data)
        // Add to history
        addToHistory(data, imageBlob)
      },
    })
  }

  const handleImageClear = () => {
    setRecognitionResult(null)
    setCurrentImageBlob(null)
    setCurrentImageSrc(null)
  }

  const handleHistoryItemClick = (item: ReturnType<typeof useRecognitionHistory>['history'][0]) => {
    // Convert data URL back to blob
    fetch(item.imageDataUrl)
      .then(res => res.blob())
      .then(blob => {
        setCurrentImageBlob(blob)
        setCurrentImageSrc(item.imageDataUrl)
        setRecognitionResult({
          letter: item.letter,
          confidence: item.confidence,
          hand_pose: item.handPose,
        })
        setIsHistoryOpen(false)
      })
      .catch(err => console.error("Failed to load history item:", err))
  }

  const handleClearHistory = () => {
    if (confirm("Are you sure you want to clear all recognition history?")) {
      clearHistory()
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between gap-3">
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

            {/* History Toggle Button */}
            <Button
              variant="outline"
              size="default"
              onClick={() => setIsHistoryOpen(!isHistoryOpen)}
              className="gap-2"
              aria-label={isHistoryOpen ? "Close history" : "Open history"}
            >
              <History className="h-5 w-5" />
              <span className="hidden sm:inline">History</span>
              {history.length > 0 && (
                <span className="flex h-5 min-w-5 items-center justify-center rounded-full bg-primary px-1.5 text-xs text-primary-foreground">
                  {history.length}
                </span>
              )}
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content - Split View */}
      <main className="container mx-auto p-4">
        <div className="grid gap-4 lg:grid-cols-2 lg:gap-6">
          {/* Left Side - Image Input */}
          <ASLInput
            onImageCapture={handleImageCapture}
            onImageClear={handleImageClear}
            isProcessing={isPending}
            externalImageSrc={currentImageSrc}
            externalImageBlob={currentImageBlob}
          />

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

      {/* Overlay for mobile */}
      {isHistoryOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 sm:hidden"
          onClick={() => setIsHistoryOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* History Panel */}
      <HistoryPanel
        history={history}
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        onItemClick={handleHistoryItemClick}
        onItemDelete={removeFromHistory}
        onClearAll={handleClearHistory}
      />
    </div>
  )
}
