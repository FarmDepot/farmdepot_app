import './globals.css'
import Image from 'next/image'
import Link from 'next/link'

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-900">
        <header className="flex items-center p-4 shadow bg-white">
          <Image src="/farmdepot-logo.png" alt="FarmDepot" width={40} height={40} />
          <h1 className="ml-2 font-bold text-lg">FarmDepot</h1>
          <nav className="ml-auto flex space-x-4">
            <Link href="/">Home</Link>
            <Link href="/list">Post Ad</Link>
          </nav>
        </header>
        <main className="p-6">{children}</main>
      </body>
    </html>
  )
}
