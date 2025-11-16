"use client"

import { useState, useEffect, useCallback } from "react"
import type { ASLRecognitionResponse } from "@/lib/api-client"

export interface RecognitionHistoryItem {
  id: string
  timestamp: number
  letter: string
  confidence: number
  imageDataUrl: string
  handPose?: ASLRecognitionResponse["hand_pose"]
}

const STORAGE_KEY = "asl-recognition-history"
const MAX_HISTORY_ITEMS = 50

/**
 * Custom hook for managing ASL recognition history with localStorage persistence
 */
export function useRecognitionHistory() {
  const [history, setHistory] = useState<RecognitionHistoryItem[]>([])
  const [isLoaded, setIsLoaded] = useState(false)

  // Load history from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        const parsed = JSON.parse(stored) as RecognitionHistoryItem[]
        setHistory(parsed)
      }
    } catch (error) {
      console.error("Failed to load history from localStorage:", error)
    } finally {
      setIsLoaded(true)
    }
  }, [])

  // Save history to localStorage whenever it changes
  useEffect(() => {
    if (isLoaded) {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
      } catch (error) {
        console.error("Failed to save history to localStorage:", error)
      }
    }
  }, [history, isLoaded])

  /**
   * Add a new recognition result to history
   */
  const addToHistory = useCallback(
    async (result: ASLRecognitionResponse, imageBlob: Blob) => {
      try {
        // Convert blob to data URL for storage
        const reader = new FileReader()
        const dataUrl = await new Promise<string>((resolve, reject) => {
          reader.onload = () => resolve(reader.result as string)
          reader.onerror = reject
          reader.readAsDataURL(imageBlob)
        })

        const newItem: RecognitionHistoryItem = {
          id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          timestamp: Date.now(),
          letter: result.letter,
          confidence: result.confidence,
          imageDataUrl: dataUrl,
          handPose: result.hand_pose,
        }

        setHistory((prev) => {
          const updated = [newItem, ...prev]
          // Keep only the most recent MAX_HISTORY_ITEMS
          return updated.slice(0, MAX_HISTORY_ITEMS)
        })
      } catch (error) {
        console.error("Failed to add item to history:", error)
      }
    },
    []
  )

  /**
   * Remove a specific item from history
   */
  const removeFromHistory = useCallback((id: string) => {
    setHistory((prev) => prev.filter((item) => item.id !== id))
  }, [])

  /**
   * Clear all history
   */
  const clearHistory = useCallback(() => {
    setHistory([])
  }, [])

  /**
   * Get a specific history item by ID
   */
  const getHistoryItem = useCallback(
    (id: string) => {
      return history.find((item) => item.id === id)
    },
    [history]
  )

  return {
    history,
    isLoaded,
    addToHistory,
    removeFromHistory,
    clearHistory,
    getHistoryItem,
  }
}
