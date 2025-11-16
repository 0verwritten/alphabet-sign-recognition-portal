# ASL Recognition Portal - Frontend

Modern React frontend for the ASL Alphabet Recognition Portal, built with Next.js, shadcn UI, and TanStack Query.

## Features

- **Real-time Recognition**: Upload or capture ASL signs for instant recognition
- **Modern UI**: Built with shadcn UI components and Tailwind CSS
- **State Management**: TanStack Query for efficient API state management
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Camera Integration**: Capture signs directly from your webcam
- **Hand Pose Visualization**: Display detailed hand landmark information

## Tech Stack

- **Framework**: Next.js 15 (App Router)
- **UI Library**: shadcn UI (Radix UI primitives)
- **Styling**: Tailwind CSS with CSS variables
- **State Management**: TanStack Query v5
- **Icons**: Lucide React
- **TypeScript**: Full type safety

## Getting Started

### Prerequisites

- Node.js 18+ or use Fedora toolbox with `toolbox enter node`
- pnpm (recommended) or npm
- Backend API running on `http://localhost:8000`

### Installation

1. Install dependencies:

```bash
pnpm install
```

2. Configure environment variables:

```bash
cp .env.local.example .env.local
```

Edit `.env.local` and set your backend API URL:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Development

Run the development server:

```bash
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

Build for production:

```bash
pnpm build
```

Start production server:

```bash
pnpm start
```

## Project Structure

```
frontend/
├── app/                    # Next.js app router
│   ├── layout.tsx         # Root layout with QueryProvider
│   ├── page.tsx           # Main ASL recognition page
│   └── globals.css        # Global styles
├── components/            # React components
│   ├── ui/               # shadcn UI components
│   ├── asl-input.tsx     # Image/camera capture component
│   ├── asl-recognition-display.tsx  # Results display
│   └── query-provider.tsx # TanStack Query setup
├── lib/                   # Utilities and API clients
│   ├── api-client.ts     # FastAPI client with hooks
│   └── utils.ts          # Helper functions
└── public/               # Static assets
```

## Key Components

### ASLInput

Handles image capture via upload or webcam:
- File upload with image validation
- Camera access and snapshot capture
- Canvas-based image processing

### ASLRecognitionDisplay

Shows recognition results:
- Large letter display with confidence score
- Hand pose landmark information
- Bounding box dimensions
- Visual confidence meter

### API Client

TanStack Query integration:
- `useASLRecognition()` hook for mutations
- Automatic error handling
- Type-safe API responses

## API Integration

The frontend communicates with the FastAPI backend at `/api/v1/asl/predict`:

**Request:**
- Method: POST
- Body: FormData with image file
- Content-Type: multipart/form-data

**Response:**
```typescript
{
  letter: string          // Recognized letter (A-Z)
  confidence: number      // 0.0 - 1.0
  hand_pose?: {
    landmarks: Array<{x: number, y: number, z: number}>
    bounding_box?: {
      x_min: number
      y_min: number
      x_max: number
      y_max: number
    }
  }
}
```

**Error Codes:**
- 400: Invalid image format
- 422: No hand detected
- 503: Model not ready

## Customization

### Theming

The app uses CSS variables for theming. Modify `app/globals.css` to customize colors:

```css
:root {
  --primary: ...
  --secondary: ...
  /* etc */
}
```

### Adding shadcn Components

Add new shadcn UI components:

```bash
pnpx shadcn@latest add [component-name]
```

## Development Notes

- The app uses Next.js App Router with client components
- TanStack Query handles all API state management
- Images are converted to Blob format before API submission
- Camera stream is properly cleaned up on component unmount

## Environment Variables

- `NEXT_PUBLIC_API_URL`: Backend API base URL (required)

## Browser Compatibility

- Modern browsers with WebRTC support for camera access
- ES2020+ JavaScript features
- CSS Grid and Flexbox support

## Troubleshooting

**Camera not working:**
- Check browser permissions
- Ensure HTTPS in production (required for getUserMedia)
- Verify camera is not in use by another application

**API connection failed:**
- Verify backend is running on configured port
- Check CORS settings in backend
- Ensure `NEXT_PUBLIC_API_URL` is correct

**Build errors:**
- Clear `.next` directory: `rm -rf .next`
- Reinstall dependencies: `rm -rf node_modules pnpm-lock.yaml && pnpm install`

## License

MIT
