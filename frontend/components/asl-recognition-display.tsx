"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Loader2, Hand, AlertCircle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

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

interface ASLRecognitionResult {
  letter: string
  confidence: number
  hand_pose?: HandPose
}

interface ASLRecognitionDisplayProps {
  result: ASLRecognitionResult | null
  isRecognizing: boolean
  error: string | null
}

export function ASLRecognitionDisplay({ result, isRecognizing, error }: ASLRecognitionDisplayProps) {
  return (
    <Card className="flex flex-col overflow-hidden bg-card">
      <div className="border-b border-border bg-muted/30 px-4 py-3">
        <h2 className="font-semibold text-card-foreground">Recognition Result</h2>
        <p className="text-sm text-muted-foreground">Real-time ASL alphabet recognition</p>
      </div>

      <div className="flex flex-1 flex-col p-6">
        {isRecognizing ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <Loader2 className="h-12 w-12 animate-spin text-primary" />
            <p className="text-center text-muted-foreground">Analyzing hand pose and recognizing sign...</p>
          </div>
        ) : error ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          </div>
        ) : result ? (
          <div className="flex flex-1 flex-col gap-6">
            {/* Main Recognition Result */}
            <div className="flex flex-col items-center justify-center gap-4 rounded-lg bg-gradient-to-br from-primary/10 to-primary/5 p-8">
              <p className="text-sm font-medium text-muted-foreground">Recognized Letter</p>
              <div className="text-8xl font-bold text-primary">{result.letter}</div>
              <Badge variant="secondary" className="text-base px-4 py-1">
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </Badge>
            </div>

            {/* Hand Pose Information */}
            {result.hand_pose && (
              <div className="rounded-lg border border-border bg-muted/30 p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Hand className="h-5 w-5 text-primary" />
                  <h3 className="font-semibold text-sm">Hand Pose Detected</h3>
                </div>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-muted-foreground">Landmarks</p>
                    <p className="font-mono font-semibold">{result.hand_pose.landmarks.length} points</p>
                  </div>
                  {result.hand_pose.bounding_box && (
                    <div>
                      <p className="text-muted-foreground">Detection Area</p>
                      <p className="font-mono font-semibold">
                        {(result.hand_pose.bounding_box.x_max - result.hand_pose.bounding_box.x_min).toFixed(2)} ×{" "}
                        {(result.hand_pose.bounding_box.y_max - result.hand_pose.bounding_box.y_min).toFixed(2)}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Confidence Bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Prediction Confidence</span>
                <span className="font-medium">{(result.confidence * 100).toFixed(2)}%</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full bg-primary transition-all duration-500"
                  style={{ width: `${result.confidence * 100}%` }}
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center gap-4 text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
              <Hand className="h-8 w-8 text-muted-foreground" />
            </div>
            <div>
              <p className="font-medium text-foreground">No recognition yet</p>
              <p className="text-sm text-muted-foreground">
                Upload or capture an ASL sign to see the recognition result
              </p>
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}
