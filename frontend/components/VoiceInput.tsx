'use client'
import { Mic } from 'lucide-react'

export default function VoiceInput({ onResult }: { onResult: (text: string) => void }) {
  const handleVoice = () => {
    const recognition = new (window as any).webkitSpeechRecognition()
    recognition.lang = 'en-US'
    recognition.start()
    recognition.onresult = (e: any) => {
      const text = e.results[0][0].transcript
      onResult(text)
    }
  }

  return (
    <button type="button" onClick={handleVoice} className="p-2 border rounded flex items-center">
      <Mic className="w-4 h-4 mr-1" /> Voice
    </button>
  )
}
