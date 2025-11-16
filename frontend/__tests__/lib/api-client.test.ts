import { predictASLSign, useASLRecognition } from '@/lib/api-client'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactNode } from 'react'

// Mock fetch globally
global.fetch = jest.fn()

describe('API Client', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  describe('predictASLSign', () => {
    it('should successfully predict ASL sign', async () => {
      const mockResponse = {
        letter: 'A',
        confidence: 0.95,
        hand_pose: {
          landmarks: [{ x: 0.5, y: 0.5, z: 0 }],
          bounding_box: {
            x_min: 0.2,
            y_min: 0.3,
            x_max: 0.8,
            y_max: 0.9,
          },
        },
      }

      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      })

      const imageBlob = new Blob(['test'], { type: 'image/jpeg' })
      const result = await predictASLSign(imageBlob)

      expect(result).toEqual(mockResponse)
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/asl/predict',
        expect.objectContaining({
          method: 'POST',
          body: expect.any(FormData),
        })
      )
    })

    it('should throw error for 422 (no hand detected)', async () => {
      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 422,
        json: async () => ({ detail: 'No hand detected' }),
      })

      const imageBlob = new Blob(['test'], { type: 'image/jpeg' })

      await expect(predictASLSign(imageBlob)).rejects.toThrow(
        'No hand detected'
      )
    })

    it('should throw error for 503 (model not ready)', async () => {
      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 503,
        json: async () => ({ detail: 'Model not ready' }),
      })

      const imageBlob = new Blob(['test'], { type: 'image/jpeg' })

      await expect(predictASLSign(imageBlob)).rejects.toThrow(
        'ASL recognition model is not ready'
      )
    })

    it('should throw error for 400 (invalid image)', async () => {
      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'Invalid image format' }),
      })

      const imageBlob = new Blob(['test'], { type: 'image/jpeg' })

      await expect(predictASLSign(imageBlob)).rejects.toThrow(
        'Invalid image format'
      )
    })

    it('should handle generic errors', async () => {
      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({ error: 'Internal server error' }),
      })

      const imageBlob = new Blob(['test'], { type: 'image/jpeg' })

      await expect(predictASLSign(imageBlob)).rejects.toThrow(
        'Internal server error'
      )
    })
  })

  describe('useASLRecognition hook', () => {
    const createWrapper = () => {
      const queryClient = new QueryClient({
        defaultOptions: {
          queries: { retry: false },
          mutations: { retry: false },
        },
      })

      return ({ children }: { children: ReactNode }) => (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      )
    }

    it('should successfully mutate and return result', async () => {
      const mockResponse = {
        letter: 'B',
        confidence: 0.88,
      }

      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      })

      const { result } = renderHook(() => useASLRecognition(), {
        wrapper: createWrapper(),
      })

      const imageBlob = new Blob(['test'], { type: 'image/jpeg' })
      result.current.mutate(imageBlob)

      await waitFor(() => expect(result.current.isSuccess).toBe(true))

      expect(result.current.data).toEqual(mockResponse)
    })

    it('should handle mutation error', async () => {
      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 422,
        json: async () => ({ detail: 'No hand detected' }),
      })

      const { result } = renderHook(() => useASLRecognition(), {
        wrapper: createWrapper(),
      })

      const imageBlob = new Blob(['test'], { type: 'image/jpeg' })
      result.current.mutate(imageBlob)

      await waitFor(() => expect(result.current.isError).toBe(true))

      expect(result.current.error).toBeDefined()
      expect(result.current.error?.message).toContain('No hand detected')
    })

    it('should track pending state', async () => {
      ;(global.fetch as jest.Mock).mockImplementationOnce(
        () =>
          new Promise((resolve) =>
            setTimeout(
              () =>
                resolve({
                  ok: true,
                  json: async () => ({ letter: 'C', confidence: 0.92 }),
                }),
              100
            )
          )
      )

      const { result } = renderHook(() => useASLRecognition(), {
        wrapper: createWrapper(),
      })

      const imageBlob = new Blob(['test'], { type: 'image/jpeg' })
      result.current.mutate(imageBlob)

      expect(result.current.isPending).toBe(true)

      await waitFor(() => expect(result.current.isSuccess).toBe(true))

      expect(result.current.isPending).toBe(false)
    })
  })
})
