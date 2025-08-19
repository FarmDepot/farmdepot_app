import Image from 'next/image'

export default function ProductCard({ product }: { product: any }) {
  return (
    <div className="p-4 border rounded shadow-sm bg-white">
      {product.photo && (
        <Image
          src={product.photo}
          alt={product.name}
          width={200}
          height={150}
          className="rounded mb-2"
        />
      )}
      <h2 className="font-bold">{product.name}</h2>
      <p>₦{product.price}</p>
      <p className="text-sm text-gray-600">{product.location}</p>
    </div>
  )
}
