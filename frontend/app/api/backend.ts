const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'

export async function getProducts(query: string) {
  const res = await fetch(`${BACKEND_URL}/products?search=${query}`)
  return res.json()
}

export async function postProduct(product: any) {
  const formData = new FormData()
  for (let key in product) {
    if (product[key]) formData.append(key, product[key])
  }
  const res = await fetch(`${BACKEND_URL}/products`, {
    method: 'POST',
    body: formData,
  })
  return res.json()
}
