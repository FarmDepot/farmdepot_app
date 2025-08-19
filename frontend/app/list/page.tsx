'use client'
import { useState } from 'react'
import { postProduct } from '../api/backend'
import VoiceInput from '@/components/VoiceInput'

export default function ListPage() {
  const [form, setForm] = useState({ name: '', price: '', location: '' })
  const [photo, setPhoto] = useState<File | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    await postProduct({ ...form, photo })
    alert('Product submitted!')
  }

  return (
    <form onSubmit={handleSubmit} className="grid gap-4 max-w-md">
      <label>
        Product Name
        <input
          value={form.name}
          onChange={(e) => setForm({ ...form, name: e.target.value })}
          className="p-2 border rounded w-full"
        />
      </label>
      <VoiceInput onResult={(text) => setForm({ ...form, name: text })} />

      <label>
        Price
        <input
          type="number"
          value={form.price}
          onChange={(e) => setForm({ ...form, price: e.target.value })}
          className="p-2 border rounded w-full"
        />
      </label>

      <label>
        Location (LGA)
        <input
          value={form.location}
          onChange={(e) => setForm({ ...form, location: e.target.value })}
          className="p-2 border rounded w-full"
        />
      </label>

      <label>
        Photo
        <input type="file" onChange={(e) => setPhoto(e.target.files?.[0] || null)} />
      </label>

      <button className="bg-green-600 text-white p-2 rounded">Submit</button>
    </form>
  )
}
