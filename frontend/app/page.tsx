'use client'
import { useEffect, useState } from 'react'
import { getProducts } from './api/backend'
import ProductCard from '@/components/ProductCard'
import VoiceInput from '@/components/VoiceInput'

export default function HomePage() {
  const [products, setProducts] = useState<any[]>([])
  const [query, setQuery] = useState('')

  useEffect(() => {
    getProducts(query).then(setProducts)
  }, [query])

  return (
    <div>
      <div className="flex gap-2 mb-4">
        <input
          type="text"
          placeholder="Search products..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="p-2 border rounded flex-grow"
        />
        <VoiceInput onResult={setQuery} />
      </div>
      <div className="grid gap-4">
        {products.map((p) => (
          <ProductCard key={p.id} product={p} />
        ))}
      </div>
    </div>
  )
}
