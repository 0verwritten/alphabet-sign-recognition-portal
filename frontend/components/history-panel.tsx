"use client"

import type React from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Trash2, Clock, X } from "lucide-react"
import type { RecognitionHistoryItem } from "@/hooks/use-recognition-history"

interface HistoryPanelProps {
  history: RecognitionHistoryItem[]
  onItemClick?: (item: RecognitionHistoryItem) => void
  onItemDelete?: (id: string) => void
  onClearAll?: () => void
  onClose?: () => void
  isOpen: boolean
}

export function HistoryPanel({
  history,
  onItemClick,
  onItemDelete,
  onClearAll,
  onClose,
  isOpen,
}: HistoryPanelProps) {
  if (!isOpen) return null

  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return "Just now"
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`

    return date.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: date.getFullYear() !== now.getFullYear() ? "numeric" : undefined,
    })
  }

  return (
    <div className="fixed inset-y-0 right-0 z-50 w-full sm:w-96 bg-background border-l border-border shadow-2xl">
      <div className="flex h-full flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border bg-muted/30 px-4 py-3">
          <div>
            <h2 className="font-semibold text-lg">Recognition History</h2>
            <p className="text-xs text-muted-foreground">
              {history.length} {history.length === 1 ? "item" : "items"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {history.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onClearAll}
                className="gap-2"
                aria-label="Clear all history"
              >
                <Trash2 className="h-4 w-4" />
                Clear All
              </Button>
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              aria-label="Close history panel"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>
        </div>

        {/* History List */}
        <ScrollArea className="flex-1">
          {history.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full p-8 text-center">
              <Clock className="h-12 w-12 text-muted-foreground/50 mb-4" />
              <p className="text-sm text-muted-foreground">No recognition history yet</p>
              <p className="text-xs text-muted-foreground mt-1">
                Recognized signs will appear here
              </p>
            </div>
          ) : (
            <div className="p-4 space-y-3">
              {history.map((item) => (
                <Card
                  key={item.id}
                  className="group relative overflow-hidden transition-all hover:shadow-md cursor-pointer"
                  onClick={() => onItemClick?.(item)}
                >
                  <div className="flex gap-3 p-3">
                    {/* Thumbnail */}
                    <div className="relative flex-shrink-0 w-20 h-20 rounded-lg overflow-hidden bg-black/5">
                      <img
                        src={item.imageDataUrl}
                        alt={`ASL sign ${item.letter}`}
                        className="w-full h-full object-cover"
                      />
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="text-2xl font-bold">{item.letter}</span>
                            <Badge
                              variant={item.confidence >= 0.8 ? "default" : "secondary"}
                              className="text-xs"
                            >
                              {(item.confidence * 100).toFixed(0)}%
                            </Badge>
                          </div>
                          <p className="text-xs text-muted-foreground mt-1 flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {formatTimestamp(item.timestamp)}
                          </p>
                        </div>

                        {/* Delete Button */}
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity"
                          onClick={(e) => {
                            e.stopPropagation()
                            onItemDelete?.(item.id)
                          }}
                          aria-label={`Delete ${item.letter} from history`}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>

                      {/* Hand Pose Info */}
                      {item.handPose && (
                        <p className="text-xs text-muted-foreground mt-2">
                          {item.handPose.landmarks.length} landmarks detected
                        </p>
                      )}
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </ScrollArea>
      </div>
    </div>
  )
}
