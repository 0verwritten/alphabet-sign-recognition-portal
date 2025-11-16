"use client"

import { Card } from "@/components/ui/card"
import { Loader2, MessageSquare } from "lucide-react"

interface TranslationDisplayProps {
  translationText: string
  isTranslating: boolean
}

export function TranslationDisplay({ translationText, isTranslating }: TranslationDisplayProps) {
  return (
    <Card className="flex flex-col overflow-hidden bg-card">
      <div className="border-b border-border bg-muted/30 px-4 py-3">
        <h2 className="font-semibold text-card-foreground">Translation</h2>
        <p className="text-sm text-muted-foreground">Real-time translation results</p>
      </div>

      <div className="flex flex-1 flex-col p-6">
        {isTranslating ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <Loader2 className="h-12 w-12 animate-spin text-primary" />
            <p className="text-center text-muted-foreground">Processing video and generating translation...</p>
          </div>
        ) : translationText ? (
          <div className="flex flex-1 flex-col gap-4">
            <div className="flex-1 rounded-lg bg-muted/50 p-6">
              <p className="text-lg leading-relaxed text-foreground">{translationText}</p>
            </div>
          </div>
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center gap-4 text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
              <MessageSquare className="h-8 w-8 text-muted-foreground" />
            </div>
            <div>
              <p className="font-medium text-foreground">No translation yet</p>
              <p className="text-sm text-muted-foreground">Upload or record a video to see the translation</p>
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}
